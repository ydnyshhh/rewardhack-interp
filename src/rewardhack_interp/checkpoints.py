from __future__ import annotations

import re
from pathlib import Path

from rewardhack_interp.config import CheckpointComparisonConfig, RolloutConfig
from rewardhack_interp.io import load_models, write_json
from rewardhack_interp.rollouts import collect_rollouts
from rewardhack_interp.types import (
    CheckpointComparisonArtifact,
    CheckpointSummary,
    TrajectoryArtifact,
)
from rewardhack_interp.utils.paths import ensure_dir


def compare_checkpoints(config: CheckpointComparisonConfig) -> CheckpointComparisonArtifact:
    ensure_dir(config.per_checkpoint_rollout_dir)
    summaries: list[CheckpointSummary] = []
    for model_config in config.checkpoints:
        model_slug = _slugify_model_name(
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
            )
        )
        records = load_models(rollout_path, TrajectoryArtifact)
        official_rewards = [record.reward_metrics.official_reward for record in records]
        oracle_rewards = [record.reward_metrics.oracle_reward for record in records]
        verifier_gaps = [record.reward_metrics.verifier_gap for record in records]
        false_passes = [record.reward_metrics.false_pass for record in records]
        counts: dict[str, int] = {}
        for record in records:
            counts[record.cohort.value] = counts.get(record.cohort.value, 0) + 1
        summaries.append(
            CheckpointSummary(
                model_name_or_path=model_config.model_name_or_path,
                adapter_name_or_path=model_config.adapter_name_or_path,
                mean_official_reward=_mean(official_rewards),
                mean_oracle_reward=_mean(oracle_rewards),
                mean_verifier_gap=_mean(verifier_gaps),
                false_pass_rate=(
                    float(sum(false_passes) / len(false_passes))
                    if false_passes
                    else 0.0
                ),
                counts_by_cohort=dict(sorted(counts.items())),
            )
        )

    artifact = CheckpointComparisonArtifact(
        environment_name=config.environment.name,
        environment_profile=config.environment.profile,
        task_seeds=config.environment.resolved_task_seeds(),
        summaries=summaries,
    )
    write_json(config.output_path, artifact.model_dump(mode="json"))
    return artifact


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _slugify_model_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()
