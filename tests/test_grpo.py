from __future__ import annotations

from pathlib import Path

import pytest

from rewardhack_interp.config import (
    EnvironmentSpec,
    GRPOHeldoutEvaluationConfig,
    GRPORunConfig,
    ModelConfig,
)
from rewardhack_interp.rl.grpo import build_grpo_evaluation_targets, dataset_task_seeds


def build_grpo_config(tmp_path: Path) -> GRPORunConfig:
    return GRPORunConfig(
        run_name="demo-grpo",
        output_dir=tmp_path / "checkpoints",
        environment=EnvironmentSpec(
            name="code/spec-overfit",
            profile="high",
            start_seed=0,
            num_tasks=4,
        ),
        model=ModelConfig(
            model_name_or_path="Qwen/Qwen3-1.7B",
            torch_dtype="bfloat16",
            device_map="auto",
        ),
        dataset_size=6,
        evaluation=GRPOHeldoutEvaluationConfig(
            environment=EnvironmentSpec(
                name="code/spec-overfit",
                profile="high",
                start_seed=100,
                num_tasks=3,
            ),
            output_path=tmp_path / "heldout.json",
            per_checkpoint_rollout_dir=tmp_path / "heldout-rollouts",
        ),
    )


def test_dataset_task_seeds_extend_past_environment_range(tmp_path: Path) -> None:
    config = build_grpo_config(tmp_path)
    assert dataset_task_seeds(config) == [0, 1, 2, 3, 4, 5]


def test_grpo_config_rejects_overlapping_heldout_eval_seeds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="disjoint"):
        GRPORunConfig(
            run_name="demo-grpo",
            output_dir=tmp_path / "checkpoints",
            environment=EnvironmentSpec(
                name="code/spec-overfit",
                profile="high",
                start_seed=0,
                num_tasks=4,
            ),
            model=ModelConfig(
            model_name_or_path="Qwen/Qwen3-1.7B",
                torch_dtype="bfloat16",
                device_map="auto",
            ),
            dataset_size=4,
            evaluation=GRPOHeldoutEvaluationConfig(
                environment=EnvironmentSpec(
                    name="code/spec-overfit",
                    profile="high",
                    start_seed=2,
                    num_tasks=4,
                )
            ),
        )


def test_build_grpo_evaluation_targets_orders_phases(tmp_path: Path) -> None:
    output_dir = tmp_path / "checkpoints"
    output_dir.mkdir()
    (output_dir / "checkpoint-50").mkdir()
    (output_dir / "checkpoint-25").mkdir()

    config = build_grpo_config(tmp_path)
    config = config.model_copy(update={"output_dir": output_dir})
    targets = build_grpo_evaluation_targets(config)

    assert [target.phase_label for target in targets] == [
        "pre_training",
        "checkpoint_25",
        "checkpoint_50",
        "post_training_final",
    ]
    assert targets[1].model_config.adapter_name_or_path is not None
    assert targets[1].rollout_path.name == "checkpoint_25.jsonl"
