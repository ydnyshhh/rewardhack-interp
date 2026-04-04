from __future__ import annotations

import tomllib
from pathlib import Path


def load_toml(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_probe_example_uses_rollout_example_manifest_path() -> None:
    rollout_config = load_toml(Path("configs/examples/rollout_qwen3.toml"))
    probe_config = load_toml(Path("configs/examples/probe_false_pass_vs_true_pass.toml"))

    rollout_manifest_path = rollout_config["activation_manifest_path"]
    probe_manifest_path = probe_config["features"]["activation_manifest_path"]

    assert rollout_manifest_path == probe_manifest_path


def test_experiment_one_example_configs_exist() -> None:
    assert Path("configs/examples/experiment1_rollout_patch_verification.toml").is_file()
    assert Path("configs/examples/experiment1_patch_verification.toml").is_file()


def test_code_rollout_example_disables_thinking_and_uses_strict_prompt() -> None:
    rollout_config = load_toml(Path("configs/examples/rollout_qwen3.toml"))

    model_config = rollout_config["model"]
    assert model_config["enable_thinking"] is False
    assert "<think>" in model_config["system_prompt"]
    assert "Output only valid Python code" in model_config["system_prompt"]


def test_pilot_rollout_example_is_deterministic_for_interface_debugging() -> None:
    rollout_config = load_toml(Path("configs/examples/rollout_qwen3_pilot.toml"))

    sampling_config = rollout_config["sampling"]
    assert sampling_config["num_completions"] == 1
    assert sampling_config["do_sample"] is False
