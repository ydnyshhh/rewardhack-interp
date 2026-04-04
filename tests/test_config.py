from __future__ import annotations

from pathlib import Path

import pytest

from rewardhack_interp.config import ExperimentOneConfig, RolloutConfig, load_config


def test_load_rollout_config_from_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "rollout.toml"
    config_path.write_text(
        """
run_name = "demo"
output_path = "artifacts/rollouts/demo.jsonl"

[environment]
name = "code/spec-overfit"
profile = "high"
start_seed = 10
num_tasks = 3

[model]
model_name_or_path = "Qwen/Qwen3-1.7B-Instruct"
torch_dtype = "bfloat16"
device_map = "auto"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path, RolloutConfig)
    assert config.run_name == "demo"
    assert config.environment.resolved_task_seeds() == [10, 11, 12]
    assert config.model.policy_id == "Qwen/Qwen3-1.7B-Instruct"


def test_experiment_one_config_rejects_overlapping_seed_splits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        ExperimentOneConfig(
            rollout_path=tmp_path / "rollouts.jsonl",
            activation_manifest_path=tmp_path / "activations.jsonl",
            train_task_seeds=[0, 1, 2],
            eval_task_seeds=[2, 3, 4],
        )
