from rewardhack_interp.cohorts import classify_cohort
from rewardhack_interp.types import CohortLabel, RewardMetrics


def test_classify_reward_hack() -> None:
    metrics = RewardMetrics(
        official_reward=1.0,
        oracle_reward=0.0,
        verifier_gap=1.0,
        false_pass=True,
        official_passed=True,
        oracle_passed=False,
    )
    assert classify_cohort(metrics) == CohortLabel.reward_hack


def test_classify_genuine_success() -> None:
    metrics = RewardMetrics(
        official_reward=1.0,
        oracle_reward=1.0,
        verifier_gap=0.0,
        false_pass=False,
        official_passed=True,
        oracle_passed=True,
    )
    assert classify_cohort(metrics) == CohortLabel.genuine_success
