from __future__ import annotations

from pathlib import Path
from typing import Any

from rewardhack_interp.activations import load_activation_artifact, load_activation_tensors
from rewardhack_interp.cohorts import classify_cohort
from rewardhack_interp.config import PatchConfig
from rewardhack_interp.gym_integration import build_environment, reward_metrics_from_trajectory
from rewardhack_interp.io import read_json, read_jsonl, write_json
from rewardhack_interp.modeling import load_qwen_model
from rewardhack_interp.tracking import start_wandb_run
from rewardhack_interp.types import PatchTrialArtifact, TrajectoryArtifact


def run_patch_experiment(config: PatchConfig) -> PatchTrialArtifact:
    with start_wandb_run(
        wandb_config=config.wandb,
        run_name=Path(config.output_path).stem,
        job_type="activation_patching",
        config_payload=config.model_dump(mode="json"),
    ) as tracker:
        source_record = load_rollout_record(config.source_rollout_path, config.source_trace_id)
        donor_record = load_rollout_record(config.donor_rollout_path, config.donor_trace_id)
        source_activation = load_activation_artifact(config.source_activation_path)
        donor_activation = load_activation_artifact(config.donor_activation_path)
        _ = source_activation
        donor_tensors = load_activation_tensors(donor_activation)

        model_bundle = load_qwen_model(config.model)
        environment = build_environment(config.environment)
        source_task = environment.sample_task(seed=source_record.task.task_seed)

        patched_completion = generate_with_activation_patch(
            model=model_bundle.model,
            tokenizer=model_bundle.tokenizer,
            prompt_token_ids=source_record.generation.prompt_token_ids,
            donor_tensors=donor_tensors,
            donor_prompt_token_count=donor_activation.prompt_token_count,
            layer_names=config.layer_names,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        patched_trajectory = environment.evaluate_output(
            source_task,
            patched_completion,
            policy_id=config.model.policy_id,
            annotations={"causal_patch": True, "donor_trace_id": donor_record.trace_id},
        )
        patched_metrics = reward_metrics_from_trajectory(patched_trajectory)
        patched_cohort = classify_cohort(patched_metrics)

        artifact = PatchTrialArtifact(
            source_trace_id=source_record.trace_id,
            donor_trace_id=donor_record.trace_id,
            mode=config.mode,
            layer_names=list(config.layer_names),
            baseline_completion=source_record.generation.completion_text,
            patched_completion=patched_completion,
            baseline_official_reward=source_record.reward_metrics.official_reward,
            baseline_oracle_reward=source_record.reward_metrics.oracle_reward,
            patched_official_reward=patched_metrics.official_reward,
            patched_oracle_reward=patched_metrics.oracle_reward,
            baseline_cohort=source_record.cohort,
            patched_cohort=patched_cohort,
        )
        write_json(config.output_path, artifact.model_dump(mode="json"))
        tracker.log_summary(artifact.model_dump(mode="json"))
        tracker.log_path(config.output_path, artifact_type="patch-trial")
        return artifact


def load_rollout_record(path: Path, trace_id: str | None) -> TrajectoryArtifact:
    if path.suffix.lower() == ".json":
        return TrajectoryArtifact.model_validate(read_json(path))
    rows = [TrajectoryArtifact.model_validate(row) for row in read_jsonl(path)]
    if trace_id is None:
        if len(rows) != 1:
            raise ValueError("Provide trace_id when loading from a multi-row JSONL rollout file.")
        return rows[0]
    for row in rows:
        if row.trace_id == trace_id:
            return row
    raise KeyError(f"Trace id {trace_id!r} was not found in {path}.")


def generate_with_activation_patch(
    *,
    model: Any,
    tokenizer: Any,
    prompt_token_ids: list[int],
    donor_tensors: dict[str, Any],
    donor_prompt_token_count: int,
    layer_names: list[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    import torch

    device = next(model.parameters()).device
    generated_ids = list(prompt_token_ids)
    for _step in range(max_new_tokens):
        input_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
        hooks = register_patch_hooks(
            model=model,
            layer_names=layer_names,
            donor_tensors=donor_tensors,
            source_prompt_token_count=len(prompt_token_ids),
            donor_prompt_token_count=donor_prompt_token_count,
        )
        try:
            with torch.no_grad():
                logits = model(input_ids=input_ids, use_cache=False).logits[:, -1, :]
        finally:
            for hook in hooks:
                hook.remove()

        if do_sample:
            probabilities = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
            next_token = sample_top_p(probabilities, top_p=top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token_id = int(next_token[0].item())
        if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
            break
        generated_ids.append(next_token_id)
    completion_ids = generated_ids[len(prompt_token_ids) :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def register_patch_hooks(
    *,
    model: Any,
    layer_names: list[str],
    donor_tensors: dict[str, Any],
    source_prompt_token_count: int,
    donor_prompt_token_count: int,
) -> list[Any]:
    hooks: list[Any] = []
    layer_map = {name: module for name, module in model.named_modules()}
    for layer_name in layer_names:
        module_name = resolve_patch_module_name(layer_name)
        if module_name not in layer_map or layer_name not in donor_tensors:
            continue
        donor_tensor = donor_tensors[layer_name]

        def patch_hook(
            _module: Any,
            _args: Any,
            output: Any,
            *,
            cached_donor: Any = donor_tensor,
        ) -> Any:
            return patch_output(
                output=output,
                donor_tensor=cached_donor,
                source_prompt_token_count=source_prompt_token_count,
                donor_prompt_token_count=donor_prompt_token_count,
            )

        hooks.append(layer_map[module_name].register_forward_hook(patch_hook))
    return hooks


def resolve_patch_module_name(layer_name: str) -> str:
    if layer_name.startswith("hidden_state.layer_"):
        layer_index = int(layer_name.split("_")[-1])
        return f"model.layers.{layer_index}"
    if layer_name.startswith("module."):
        return layer_name.removeprefix("module.")
    return layer_name


def patch_output(
    *,
    output: Any,
    donor_tensor: Any,
    source_prompt_token_count: int,
    donor_prompt_token_count: int,
) -> Any:
    import torch

    source_index = max(source_prompt_token_count - 1, 0)
    donor_index = max(donor_prompt_token_count - 1, 0)

    if torch.is_tensor(output):
        patched = output.clone()
        if patched.ndim >= 3 and donor_tensor.ndim >= 2:
            patched[:, source_index, :] = donor_tensor[donor_index].to(
                device=patched.device,
                dtype=patched.dtype,
            )
        return patched
    if isinstance(output, tuple) and output:
        first = output[0]
        if torch.is_tensor(first):
            patched_first = first.clone()
            if patched_first.ndim >= 3 and donor_tensor.ndim >= 2:
                patched_first[:, source_index, :] = donor_tensor[donor_index].to(
                    device=patched_first.device,
                    dtype=patched_first.dtype,
                )
            return (patched_first, *tuple(output[1:]))
    return output


def sample_top_p(probabilities: Any, *, top_p: float) -> Any:
    import torch

    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
    cutoff = cumulative > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_probabilities = sorted_probabilities.masked_fill(cutoff, 0.0)
    sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(sorted_probabilities, num_samples=1)
    return sorted_indices.gather(-1, sampled)
