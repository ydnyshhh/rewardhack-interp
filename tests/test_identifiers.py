from __future__ import annotations

from rewardhack_interp.types import (
    CohortLabel,
    FinishReason,
    GenerationRecord,
    NormalizedRolloutRecord,
    RewardMetrics,
    TaskReference,
    TrajectoryArtifact,
)
from rewardhack_interp.utils.identifiers import safe_artifact_stem, trajectory_sample_id


def build_record(trace_id: str, completion_index: int) -> TrajectoryArtifact:
    return TrajectoryArtifact(
        run_id="demo",
        trace_id=trace_id,
        policy_id="Qwen/Qwen3-4B",
        model_name_or_path="Qwen/Qwen3-4B",
        task=TaskReference(
            environment_name="code/spec-overfit",
            environment_profile="high",
            task_seed=0,
            task_id="task-0",
            family="code",
            difficulty="hard",
            prompt="Implement solve",
            expected_interface="solve",
        ),
        generation=GenerationRecord(
            completion_index=completion_index,
            rendered_prompt="prompt",
            completion_text="def solve():\n    return 1",
            prompt_token_ids=[1, 2],
            completion_token_ids=[3, 4],
            prompt_token_count=2,
            completion_token_count=2,
            finish_reason=FinishReason.eos,
        ),
        reward_metrics=RewardMetrics(
            official_reward=1.0,
            oracle_reward=0.0,
            verifier_gap=1.0,
            false_pass=True,
            official_passed=True,
            oracle_passed=False,
        ),
        normalized_rollout=NormalizedRolloutRecord(
            prompt="Implement solve",
            completion="def solve():\n    return 1",
            env_id="code/spec-overfit",
            family_id="code",
            task_id="task-0",
            official_reward=1.0,
            oracle_reward=0.0,
            verifier_gap=1.0,
            false_pass=True,
        ),
        cohort=CohortLabel.reward_hack,
        trajectory={"runtime": {}},
        mech_interp_row={"trace_id": trace_id},
    )


def test_trajectory_sample_id_uses_completion_index() -> None:
    first = trajectory_sample_id(build_record("trace:abc", 0))
    second = trajectory_sample_id(build_record("trace:abc", 1))

    assert first != second
    assert first.endswith("__completion_0")
    assert second.endswith("__completion_1")


def test_safe_artifact_stem_removes_windows_unsafe_characters() -> None:
    assert safe_artifact_stem("trace:abc__completion_1") == "trace_abc__completion_1"
