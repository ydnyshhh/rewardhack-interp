from rewardhack_interp.gym_integration import build_normalized_rollout_record
from rewardhack_interp.types import RewardMetrics, TaskReference


def test_build_normalized_rollout_record() -> None:
    task_reference = TaskReference(
        environment_name="code/spec-overfit",
        environment_profile="high",
        task_seed=7,
        task_id="code/spec-overfit:7",
        family="code",
        difficulty="medium",
        prompt="Solve the task.",
        expected_interface="python",
    )
    reward_metrics = RewardMetrics(
        official_reward=1.0,
        oracle_reward=0.25,
        verifier_gap=0.75,
        false_pass=True,
        official_passed=True,
        oracle_passed=False,
        exploit_labels=["spec-overfit"],
    )

    normalized = build_normalized_rollout_record(
        task_reference=task_reference,
        completion="return 42",
        reward_metrics=reward_metrics,
    )

    assert normalized.prompt == "Solve the task."
    assert normalized.completion == "return 42"
    assert normalized.env_id == "code/spec-overfit"
    assert normalized.family_id == "code"
    assert normalized.task_id == "code/spec-overfit:7"
    assert normalized.official_reward == 1.0
    assert normalized.oracle_reward == 0.25
    assert normalized.verifier_gap == 0.75
    assert normalized.false_pass is True
    assert normalized.exploit_labels == ["spec-overfit"]
