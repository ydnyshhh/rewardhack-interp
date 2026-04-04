from __future__ import annotations

from pathlib import Path

from rewardhack_interp.config import RolloutConfig, load_config


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
