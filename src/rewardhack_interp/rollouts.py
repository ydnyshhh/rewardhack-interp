from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

from rewardhack_interp.activations import ActivationCaptureRunner
from rewardhack_interp.cohorts import classify_cohort
from rewardhack_interp.config import RolloutConfig
from rewardhack_interp.gym_integration import (
    build_environment,
    build_mech_interp_row,
    reward_metrics_from_trajectory,
    task_to_reference,
)
from rewardhack_interp.io import append_jsonl, write_json
from rewardhack_interp.modeling import load_qwen_model
from rewardhack_interp.types import GenerationRecord, TrajectoryArtifact
from rewardhack_interp.utils.paths import ensure_parent_dir, to_path_string
from rewardhack_interp.utils.seeding import set_global_seed


def collect_rollouts(config: RolloutConfig) -> dict[str, Any]:
    set_global_seed(config.environment.start_seed)
    output_path = ensure_parent_dir(config.output_path)
    if output_path.exists():
        output_path.unlink()

    activation_manifest_path = config.activation_manifest_path
    if config.activation_capture.enabled:
        activation_manifest_path = activation_manifest_path or output_path.with_name(
            f"{output_path.stem}.activations.jsonl"
        )
        activation_manifest_path = ensure_parent_dir(activation_manifest_path)
        if activation_manifest_path.exists():
            activation_manifest_path.unlink()

    environment = build_environment(config.environment)
    model_bundle = load_qwen_model(config.model)
    activation_runner = (
        ActivationCaptureRunner(model_bundle, config.activation_capture)
        if config.activation_capture.enabled
        else None
    )

    run_id = f"{config.run_name}-{uuid4().hex[:10]}"
    counts: Counter[str] = Counter()
    rows_written = 0

    from tqdm.auto import tqdm

    for task_seed in tqdm(config.environment.resolved_task_seeds(), desc="rollouts"):
        task = environment.sample_task(seed=task_seed)
        task_reference = task_to_reference(task, config.environment, task_seed)
        generated_samples = model_bundle.generate(task_reference.prompt, config.sampling)
        for completion_index, sample in enumerate(generated_samples):
            trajectory = environment.evaluate_output(
                task,
                sample.text,
                steps=[
                    {"role": "user", "content": task_reference.prompt},
                    {
                        "role": "assistant",
                        "content": sample.text,
                        "completion_index": completion_index,
                        "decoded_tokens": sample.decoded_completion_tokens,
                    },
                ],
                policy_id=config.model.policy_id,
                annotations={
                    "task_seed": task_seed,
                    "completion_index": completion_index,
                    "rendered_prompt": sample.rendered_prompt,
                    "sampling": config.sampling.model_dump(mode="json"),
                },
            )
            reward_metrics = reward_metrics_from_trajectory(trajectory)
            cohort = classify_cohort(reward_metrics)
            mech_row = build_mech_interp_row(trajectory)

            trajectory_payload = trajectory.to_dict(
                include_hidden_task_metadata=config.include_hidden_task_metadata
            )
            trajectory_payload["runtime"]["tokens_in"] = len(sample.prompt_token_ids)
            trajectory_payload["runtime"]["tokens_out"] = len(sample.completion_token_ids)

            record = TrajectoryArtifact(
                run_id=run_id,
                trace_id=str(mech_row["trace_id"]),
                policy_id=config.model.policy_id,
                model_name_or_path=config.model.model_name_or_path,
                revision=config.model.revision,
                adapter_name_or_path=config.model.adapter_name_or_path,
                task=task_reference,
                generation=GenerationRecord(
                    completion_index=completion_index,
                    rendered_prompt=sample.rendered_prompt,
                    completion_text=sample.text,
                    prompt_token_ids=sample.prompt_token_ids,
                    completion_token_ids=sample.completion_token_ids,
                    decoded_completion_tokens=sample.decoded_completion_tokens,
                    token_logprobs=sample.token_logprobs,
                    prompt_token_count=len(sample.prompt_token_ids),
                    completion_token_count=len(sample.completion_token_ids),
                    finish_reason=sample.finish_reason,
                    sampling=config.sampling.model_dump(mode="json"),
                ),
                reward_metrics=reward_metrics,
                cohort=cohort,
                trajectory=trajectory_payload,
                mech_interp_row=mech_row,
            )

            if activation_runner is not None and activation_manifest_path is not None:
                activation_artifact = activation_runner.capture_from_rollout_record(record)
                record.activation_artifact_path = str(
                    Path(activation_artifact.tensor_path).with_suffix(".json")
                )
                append_jsonl(
                    activation_manifest_path,
                    [activation_artifact.model_dump(mode="json")],
                )

            append_jsonl(output_path, [record.model_dump(mode="json")])
            counts[record.cohort.value] += 1
            rows_written += 1

    summary = {
        "run_id": run_id,
        "output_path": to_path_string(output_path),
        "activation_manifest_path": (
            to_path_string(activation_manifest_path) if activation_manifest_path else None
        ),
        "rows_written": rows_written,
        "counts_by_cohort": dict(sorted(counts.items())),
    }
    write_json(output_path.with_name(f"{output_path.stem}.summary.json"), summary)
    return summary
