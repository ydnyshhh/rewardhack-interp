from __future__ import annotations

import re
from pathlib import Path

from rewardhack_interp.config import (
    CheckpointComparisonConfig,
    ModelConfig,
    RolloutConfig,
)
from rewardhack_interp.io import load_models, write_json
from rewardhack_interp.rollouts import collect_rollouts
from rewardhack_interp.tracking import start_wandb_run
from rewardhack_interp.types import (
    CheckpointComparisonArtifact,
    CheckpointSummary,
    TrajectoryArtifact,
)
from rewardhack_interp.utils.paths import ensure_dir, to_path_string


def compare_checkpoints(config: CheckpointComparisonConfig) -> CheckpointComparisonArtifact:
    ensure_dir(config.per_checkpoint_rollout_dir)
    with start_wandb_run(
        wandb_config=config.wandb,
        run_name=Path(config.output_path).stem,
        job_type="checkpoint_comparison",
        config_payload=config.model_dump(mode="json"),
    ) as tracker:
        summaries: list[CheckpointSummary] = []
        for model_config in config.checkpoints:
            model_slug = slugify_model_name(
                model_config.adapter_name_or_path or model_config.model_name_or_path
            )
            rollout_path = Path(config.per_checkpoint_rollout_dir) / f"{model_slug}.jsonl"
            collect_rollouts(
                RolloutConfig(
                    run_name=f"compare-{model_slug}",
                    environment=config.environment,
                    model=model_config,
                    sampling=config.sampling,
                    output_path=rollout_path,
                    wandb=None,
                )
            )
            records = load_models(rollout_path, TrajectoryArtifact)
            summaries.append(
                build_checkpoint_summary(
                    records=records,
                    model_config=model_config,
                    rollout_path=rollout_path,
                )
            )

        artifact = CheckpointComparisonArtifact(
            environment_name=config.environment.name,
            environment_profile=config.environment.profile,
            task_seeds=config.environment.resolved_task_seeds(),
            summaries=summaries,
        )
        write_json(config.output_path, artifact.model_dump(mode="json"))
        tracker.log_summary(
            {
                "environment_name": artifact.environment_name,
                "environment_profile": artifact.environment_profile,
                "num_checkpoints": len(artifact.summaries),
            }
        )
        tracker.log_path(config.output_path, artifact_type="checkpoint-comparison")
        return artifact


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def build_checkpoint_summary(
    *,
    records: list[TrajectoryArtifact],
    model_config: ModelConfig,
    rollout_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    phase_label: str | None = None,
    split_name: str | None = None,
    global_step: int | None = None,
) -> CheckpointSummary:
    official_rewards = [record.reward_metrics.official_reward for record in records]
    oracle_rewards = [record.reward_metrics.oracle_reward for record in records]
    verifier_gaps = [record.reward_metrics.verifier_gap for record in records]
    false_passes = [record.reward_metrics.false_pass for record in records]
    counts: dict[str, int] = {}
    for record in records:
        counts[record.cohort.value] = counts.get(record.cohort.value, 0) + 1

    return CheckpointSummary(
        model_name_or_path=model_config.model_name_or_path,
        adapter_name_or_path=model_config.adapter_name_or_path,
        checkpoint_path=(
            to_path_string(checkpoint_path) if checkpoint_path is not None else None
        ),
        phase_label=phase_label,
        split_name=split_name,
        global_step=global_step,
        rollout_path=to_path_string(rollout_path) if rollout_path is not None else None,
        mean_official_reward=mean(official_rewards),
        mean_oracle_reward=mean(oracle_rewards),
        mean_verifier_gap=mean(verifier_gaps),
        false_pass_rate=(float(sum(false_passes) / len(false_passes)) if false_passes else 0.0),
        counts_by_cohort=dict(sorted(counts.items())),
    )


def slugify_model_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()
