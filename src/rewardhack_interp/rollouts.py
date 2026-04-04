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
    evaluate_task_output,
)
from rewardhack_interp.io import append_jsonl, write_json
from rewardhack_interp.modeling import load_qwen_model
from rewardhack_interp.tracking import start_wandb_run
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
    official_reward_total = 0.0
    oracle_reward_total = 0.0
    verifier_gap_total = 0.0
    false_pass_count = 0

    from tqdm.auto import tqdm

    with start_wandb_run(
        wandb_config=config.wandb,
        run_name=config.run_name,
        job_type="rollout_collection",
        config_payload=config.model_dump(mode="json"),
    ) as tracker:
        for task_seed in tqdm(config.environment.resolved_task_seeds(), desc="rollouts"):
            task = environment.sample_task(seed=task_seed)
            generated_samples = model_bundle.generate(task.prompt, config.sampling)
            for completion_index, sample in enumerate(generated_samples):
                evaluated_output = evaluate_task_output(
                    environment=environment,
                    spec=config.environment,
                    task=task,
                    task_seed=task_seed,
                    completion=sample.text,
                    policy_id=config.model.policy_id,
                    include_hidden_task_metadata=config.include_hidden_task_metadata,
                    steps=[
                        {"role": "user", "content": task.prompt},
                        {
                            "role": "assistant",
                            "content": sample.text,
                            "completion_index": completion_index,
                            "decoded_tokens": sample.decoded_completion_tokens,
                        },
                    ],
                    annotations={
                        "task_seed": task_seed,
                        "completion_index": completion_index,
                        "rendered_prompt": sample.rendered_prompt,
                        "sampling": config.sampling.model_dump(mode="json"),
                    },
                )
                cohort = classify_cohort(evaluated_output.reward_metrics)
                trajectory_payload = evaluated_output.trajectory_payload
                trajectory_payload["runtime"]["tokens_in"] = len(sample.prompt_token_ids)
                trajectory_payload["runtime"]["tokens_out"] = len(sample.completion_token_ids)

                record = TrajectoryArtifact(
                    run_id=run_id,
                    trace_id=str(evaluated_output.mech_interp_row["trace_id"]),
                    policy_id=config.model.policy_id,
                    model_name_or_path=config.model.model_name_or_path,
                    revision=config.model.revision,
                    adapter_name_or_path=config.model.adapter_name_or_path,
                    task=evaluated_output.task_reference,
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
                    reward_metrics=evaluated_output.reward_metrics,
                    normalized_rollout=evaluated_output.normalized_rollout,
                    cohort=cohort,
                    trajectory=trajectory_payload,
                    mech_interp_row=evaluated_output.mech_interp_row,
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
                official_reward_total += record.reward_metrics.official_reward
                oracle_reward_total += record.reward_metrics.oracle_reward
                verifier_gap_total += record.reward_metrics.verifier_gap
                false_pass_count += int(record.reward_metrics.false_pass)

        summary = {
            "run_id": run_id,
            "output_path": to_path_string(output_path),
            "activation_manifest_path": (
                to_path_string(activation_manifest_path) if activation_manifest_path else None
            ),
            "rows_written": rows_written,
            "counts_by_cohort": dict(sorted(counts.items())),
            "mean_official_reward": (
                official_reward_total / rows_written if rows_written > 0 else 0.0
            ),
            "mean_oracle_reward": (
                oracle_reward_total / rows_written if rows_written > 0 else 0.0
            ),
            "mean_verifier_gap": (
                verifier_gap_total / rows_written if rows_written > 0 else 0.0
            ),
            "false_pass_rate": (false_pass_count / rows_written if rows_written > 0 else 0.0),
        }
        summary_path = output_path.with_name(f"{output_path.stem}.summary.json")
        write_json(summary_path, summary)
        tracker.log_summary(summary)
        tracker.log_path(output_path, artifact_type="rollout-jsonl", metadata=summary)
        tracker.log_path(summary_path, artifact_type="rollout-summary", metadata=summary)
        if activation_manifest_path is not None:
            tracker.log_path(
                activation_manifest_path,
                artifact_type="activation-manifest",
                metadata={"rows_written": rows_written},
            )
        return summary
