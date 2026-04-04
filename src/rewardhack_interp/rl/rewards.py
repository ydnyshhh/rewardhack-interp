from __future__ import annotations

from typing import Any

from rewardhack_interp.config import GRPORunConfig
from rewardhack_interp.gym_integration import build_environment, reward_metrics_from_trajectory
from rewardhack_interp.io import append_jsonl
from rewardhack_interp.types import RewardMetrics, RewardSignal


def scalarize_reward(
    metrics: RewardMetrics,
    *,
    reward_signal: RewardSignal,
    gap_penalty: float,
    false_pass_penalty: float,
) -> float:
    if reward_signal == RewardSignal.official:
        return metrics.official_reward
    if reward_signal == RewardSignal.oracle:
        return metrics.oracle_reward
    if reward_signal == RewardSignal.gap_aware:
        return metrics.official_reward - gap_penalty * max(metrics.verifier_gap, 0.0)
    if reward_signal == RewardSignal.anti_hack:
        return (
            metrics.oracle_reward
            - gap_penalty * max(metrics.verifier_gap, 0.0)
            - (false_pass_penalty if metrics.false_pass else 0.0)
        )
    raise ValueError(f"Unsupported reward signal {reward_signal!r}.")


def make_reward_function(config: GRPORunConfig) -> Any:
    environment = build_environment(config.environment)

    def reward_func(completions: list[Any], task_seed: list[int], **kwargs: Any) -> list[float]:
        rewards: list[float] = []
        reward_rows: list[dict[str, Any]] = []
        for completion, seed in zip(completions, task_seed, strict=True):
            completion_text = _completion_to_text(completion)
            task = environment.sample_task(seed=int(seed))
            trajectory = environment.evaluate_output(
                task,
                completion_text,
                policy_id=config.model.policy_id,
                annotations={"grpo": True, "task_seed": int(seed)},
            )
            metrics = reward_metrics_from_trajectory(trajectory)
            reward_value = scalarize_reward(
                metrics,
                reward_signal=config.reward_signal,
                gap_penalty=config.gap_penalty,
                false_pass_penalty=config.false_pass_penalty,
            )
            rewards.append(float(reward_value))
            reward_rows.append(
                {
                    "task_seed": int(seed),
                    "completion": completion_text,
                    "reward_signal": config.reward_signal.value,
                    "reward": float(reward_value),
                    "official_reward": metrics.official_reward,
                    "oracle_reward": metrics.oracle_reward,
                    "verifier_gap": metrics.verifier_gap,
                    "false_pass": metrics.false_pass,
                    "official_passed": metrics.official_passed,
                    "oracle_passed": metrics.oracle_passed,
                }
            )
        if config.reward_trace_output is not None:
            append_jsonl(config.reward_trace_output, reward_rows)
        return rewards

    return reward_func


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        fragments: list[str] = []
        for item in completion:
            if isinstance(item, dict) and "content" in item:
                fragments.append(str(item["content"]))
            else:
                fragments.append(str(item))
        return "".join(fragments)
    return str(completion)
