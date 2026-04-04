from __future__ import annotations

from pathlib import Path

import pytest

from rewardhack_interp.analysis.layerwise_probe import run_layerwise_probe
from rewardhack_interp.config import (
    CaseStudyConfig,
    LayerwiseProbeConfig,
    RolloutSubsetConfig,
)
from rewardhack_interp.io import write_jsonl
from rewardhack_interp.rollout_analysis import (
    build_rollout_subset,
    extract_case_studies,
    summarize_rollouts,
)
from rewardhack_interp.types import (
    ActivationCaptureArtifact,
    CohortLabel,
    FinishReason,
    GenerationRecord,
    NormalizedRolloutRecord,
    RewardMetrics,
    TaskReference,
    TrajectoryArtifact,
)


def build_trajectory(
    *,
    trace_id: str,
    completion_index: int = 0,
    cohort: CohortLabel,
    official_reward: float,
    oracle_reward: float,
    verifier_gap: float,
    false_pass: bool,
    completion_text: str,
    completion_token_count: int,
) -> TrajectoryArtifact:
    return TrajectoryArtifact(
        run_id="demo-run",
        trace_id=trace_id,
        policy_id="Qwen/Qwen3-4B",
        model_name_or_path="Qwen/Qwen3-4B",
        task=TaskReference(
            environment_name="code/spec-overfit",
            environment_profile="high",
            task_seed=0,
            task_id=f"task-{trace_id}",
            family="code",
            difficulty="hard",
            prompt=f"Implement function {trace_id}",
            expected_interface="solve",
        ),
        generation=GenerationRecord(
            completion_index=completion_index,
            rendered_prompt="prompt",
            completion_text=completion_text,
            prompt_token_ids=[1, 2, 3],
            completion_token_ids=list(range(completion_token_count)),
            prompt_token_count=3,
            completion_token_count=completion_token_count,
            finish_reason=FinishReason.eos,
        ),
        reward_metrics=RewardMetrics(
            official_reward=official_reward,
            oracle_reward=oracle_reward,
            verifier_gap=verifier_gap,
            false_pass=false_pass,
            official_passed=official_reward >= 1.0,
            oracle_passed=oracle_reward >= 1.0,
        ),
        normalized_rollout=NormalizedRolloutRecord(
            prompt=f"Implement function {trace_id}",
            completion=completion_text,
            env_id="code/spec-overfit",
            family_id="code",
            task_id=f"task-{trace_id}",
            official_reward=official_reward,
            oracle_reward=oracle_reward,
            verifier_gap=verifier_gap,
            false_pass=false_pass,
        ),
        cohort=cohort,
        trajectory={"runtime": {}},
        mech_interp_row={"scenario_id": "scenario-a"},
    )


def test_summarize_rollouts_writes_summary_and_plots(tmp_path: Path) -> None:
    rollout_path = tmp_path / "rollouts.jsonl"
    write_jsonl(
        rollout_path,
        [
            build_trajectory(
                trace_id="a",
                cohort=CohortLabel.reward_hack,
                official_reward=1.0,
                oracle_reward=0.0,
                verifier_gap=1.0,
                false_pass=True,
                completion_text="def a():\n    return 1",
                completion_token_count=12,
            ).model_dump(mode="json"),
            build_trajectory(
                trace_id="b",
                cohort=CohortLabel.reward_hack,
                official_reward=0.8,
                oracle_reward=0.0,
                verifier_gap=0.8,
                false_pass=True,
                completion_text="def b():\n    return 2",
                completion_token_count=10,
            ).model_dump(mode="json"),
            build_trajectory(
                trace_id="c",
                cohort=CohortLabel.genuine_failure,
                official_reward=0.2,
                oracle_reward=0.0,
                verifier_gap=0.2,
                false_pass=False,
                completion_text="def c():\n    return 3",
                completion_token_count=8,
            ).model_dump(mode="json"),
        ],
    )

    artifact = summarize_rollouts(
        rollout_path,
        output_path=tmp_path / "summary.json",
    )

    assert artifact.total_rows == 3
    assert artifact.cohort_counts["reward_hack"] == 2
    assert artifact.cohort_summaries["reward_hack"].false_pass_rate == 1.0
    for plot_path in artifact.plot_paths.values():
        assert Path(plot_path).is_file()


def test_build_rollout_subset_balances_requested_cohorts(tmp_path: Path) -> None:
    rollout_path = tmp_path / "rollouts.jsonl"
    rows = []
    for index in range(4):
        rows.append(
            build_trajectory(
                trace_id=f"rh-{index // 2}",
                completion_index=index % 2,
                cohort=CohortLabel.reward_hack,
                official_reward=1.0,
                oracle_reward=0.0,
                verifier_gap=1.0,
                false_pass=True,
                completion_text="def x():\n    return 1",
                completion_token_count=10 + index,
            ).model_dump(mode="json")
        )
    for index in range(5):
        rows.append(
            build_trajectory(
                trace_id=f"gf-{index // 2}",
                completion_index=index % 2,
                cohort=CohortLabel.genuine_failure,
                official_reward=0.1,
                oracle_reward=0.0,
                verifier_gap=0.1,
                false_pass=False,
                completion_text="def y():\n    return 0",
                completion_token_count=20 + index,
            ).model_dump(mode="json")
        )
    write_jsonl(rollout_path, rows)

    artifact = build_rollout_subset(
        RolloutSubsetConfig(
            rollout_path=rollout_path,
            output_path=tmp_path / "subset.jsonl",
            included_cohorts=[CohortLabel.reward_hack, CohortLabel.genuine_failure],
            max_samples_per_cohort=2,
            random_seed=7,
        )
    )

    assert artifact.counts_after == {"reward_hack": 2, "genuine_failure": 2}
    assert Path(artifact.output_rollout_path).is_file()
    assert Path(artifact.summary_path).is_file()
    assert len(set(artifact.selected_sample_ids)) == 4


def test_extract_case_studies_uses_top_verifier_gap(tmp_path: Path) -> None:
    rollout_path = tmp_path / "rollouts.jsonl"
    write_jsonl(
        rollout_path,
        [
            build_trajectory(
                trace_id="low-gap",
                cohort=CohortLabel.reward_hack,
                official_reward=0.8,
                oracle_reward=0.0,
                verifier_gap=0.4,
                false_pass=True,
                completion_text="def low():\n    return 1",
                completion_token_count=10,
            ).model_dump(mode="json"),
            build_trajectory(
                trace_id="high-gap",
                cohort=CohortLabel.reward_hack,
                official_reward=1.0,
                oracle_reward=0.0,
                verifier_gap=0.9,
                false_pass=True,
                completion_text="def high():\n    return 1",
                completion_token_count=12,
            ).model_dump(mode="json"),
        ],
    )

    artifact = extract_case_studies(
        CaseStudyConfig(
            rollout_path=rollout_path,
            output_path=tmp_path / "cases.json",
            included_cohorts=[CohortLabel.reward_hack],
            samples_per_cohort=1,
        )
    )

    assert len(artifact.examples) == 1
    assert artifact.examples[0].trace_id == "high-gap"


def test_run_layerwise_probe_writes_report_and_plots(tmp_path: Path) -> None:
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyTorch is unavailable for activation probe test on this machine: {exc}")

    activation_dir = tmp_path / "activations"
    activation_dir.mkdir()
    manifest_path = tmp_path / "manifest.jsonl"

    manifest_rows = []
    for trace_id, cohort, layer_0_value, layer_1_value in [
        ("rh-1", CohortLabel.reward_hack, 1.0, 5.0),
        ("rh-2", CohortLabel.reward_hack, 1.2, 5.2),
        ("gf-1", CohortLabel.genuine_failure, -1.0, -5.0),
        ("gf-2", CohortLabel.genuine_failure, -1.2, -5.2),
    ]:
        tensor_path = activation_dir / f"{trace_id}.pt"
        torch.save(
            {
                "hidden_state.layer_0": torch.tensor([layer_0_value, layer_0_value + 0.1]),
                "hidden_state.layer_1": torch.tensor([layer_1_value, layer_1_value + 0.1]),
                "token_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
            },
            tensor_path,
        )
        manifest_rows.append(
            ActivationCaptureArtifact(
                trace_id=trace_id,
                rollout_run_id="demo",
                source_model_name_or_path="Qwen/Qwen3-4B",
                cohort=cohort,
                tensor_path=str(tensor_path),
                prompt_token_count=2,
                completion_token_count=1,
                tensor_names=["hidden_state.layer_0", "hidden_state.layer_1"],
                tensor_shapes={
                    "hidden_state.layer_0": [2],
                    "hidden_state.layer_1": [2],
                },
                token_ids=[1, 2, 3],
                capture_mode="replay",
            ).model_dump(mode="json")
        )
    write_jsonl(manifest_path, manifest_rows)

    artifact = run_layerwise_probe(
        LayerwiseProbeConfig(
            activation_manifest_path=manifest_path,
            output_path=tmp_path / "layerwise_probe.json",
            positive_label=CohortLabel.reward_hack,
            negative_label=CohortLabel.genuine_failure,
        )
    )

    assert len(artifact.results) == 2
    assert Path(artifact.report_path).is_file()
    assert Path(artifact.plot_paths["accuracy"]).is_file()
