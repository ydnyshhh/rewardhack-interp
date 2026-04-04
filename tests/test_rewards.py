from rewardhack_interp.rl.rewards import scalarize_reward
from rewardhack_interp.types import RewardMetrics, RewardSignal


def build_metrics() -> RewardMetrics:
    return RewardMetrics(
        official_reward=1.0,
        oracle_reward=0.2,
        verifier_gap=0.8,
        false_pass=True,
        official_passed=True,
        oracle_passed=False,
    )


def test_gap_aware_reward_penalizes_gap() -> None:
    metrics = build_metrics()
    reward = scalarize_reward(
        metrics,
        reward_signal=RewardSignal.gap_aware,
        gap_penalty=0.5,
        false_pass_penalty=0.0,
    )
    assert reward == 0.6


def test_anti_hack_reward_penalizes_false_pass() -> None:
    metrics = build_metrics()
    reward = scalarize_reward(
        metrics,
        reward_signal=RewardSignal.anti_hack,
        gap_penalty=0.5,
        false_pass_penalty=0.25,
    )
    assert reward == -0.45
