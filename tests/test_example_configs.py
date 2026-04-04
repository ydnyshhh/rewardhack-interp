from __future__ import annotations

import tomllib
from pathlib import Path


def load_toml(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_capture_example_uses_subset_output_path() -> None:
    subset_config = load_toml(Path("configs/examples/subset_rewardhack_vs_failure.toml"))
    capture_config = load_toml(Path("configs/examples/capture_rewardhack_vs_failure.toml"))

    subset_output_path = subset_config["output_path"]
    capture_rollout_path = capture_config["rollout_path"]

    assert subset_output_path == capture_rollout_path


def test_probe_example_uses_capture_example_manifest_path() -> None:
    capture_config = load_toml(Path("configs/examples/capture_rewardhack_vs_failure.toml"))
    probe_config = load_toml(Path("configs/examples/probe_rewardhack_vs_failure.toml"))

    capture_manifest_path = capture_config["activation_manifest_path"]
    probe_manifest_path = probe_config["activation_manifest_path"]

    assert capture_manifest_path == probe_manifest_path


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


def test_explicit_rollout_example_disables_activation_capture_by_default() -> None:
    rollout_config = load_toml(Path("configs/examples/rollout_qwen3_4b_spec_overfit.toml"))

    assert rollout_config["activation_capture"]["enabled"] is False
