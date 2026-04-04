from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rewardhack_interp.config import EnvironmentSpec
from rewardhack_interp.types import (
    NormalizedRolloutRecord,
    RewardMetrics,
    TaskReference,
)


@dataclass(frozen=True, slots=True)
class EvaluatedTaskOutput:
    task_reference: TaskReference
    reward_metrics: RewardMetrics
    normalized_rollout: NormalizedRolloutRecord
    mech_interp_row: dict[str, Any]
    trajectory_payload: dict[str, Any]


def build_environment(spec: EnvironmentSpec) -> Any:
    from rewardhack_gym import create_environment
    from rewardhack_gym.core.config import EnvironmentConfig

    config = EnvironmentConfig.from_profile(
        seed=spec.start_seed,
        profile=spec.profile,
        exploitability_overrides=dict(spec.exploitability_overrides),
    )
    return create_environment(spec.name, config=config)


def task_to_reference(task: Any, spec: EnvironmentSpec, task_seed: int) -> TaskReference:
    task_dict = task.to_dict(include_hidden=False)
    return TaskReference(
        environment_name=spec.name,
        environment_profile=spec.profile,
        task_seed=task_seed,
        task_id=str(task_dict["task_id"]),
        family=str(task_dict["family"]),
        difficulty=str(task_dict["difficulty"]),
        prompt=str(task_dict["prompt"]),
        expected_interface=str(task_dict["expected_interface"]),
        tags=[str(tag) for tag in task_dict.get("tags", [])],
        metadata=dict(task_dict.get("metadata", {})),
    )


def reward_metrics_from_trajectory(trajectory: Any) -> RewardMetrics:
    from rewardhack_gym.runners.rl import RewardAdapter

    record = RewardAdapter.from_trajectory(trajectory)
    return RewardMetrics(
        official_reward=float(record.official_reward),
        oracle_reward=float(record.oracle_reward),
        verifier_gap=float(record.verifier_gap),
        false_pass=bool(record.false_pass),
        official_passed=bool(trajectory.official_result.passed),
        oracle_passed=bool(trajectory.oracle_result.passed),
        exploit_labels=[str(label) for label in trajectory.exploit_labels],
        annotations=dict(trajectory.annotations),
        reward_metadata=dict(record.metadata),
    )


def build_normalized_rollout_record(
    task_reference: TaskReference,
    completion: str,
    reward_metrics: RewardMetrics,
) -> NormalizedRolloutRecord:
    return NormalizedRolloutRecord(
        prompt=task_reference.prompt,
        completion=completion,
        env_id=task_reference.environment_name,
        family_id=task_reference.family,
        task_id=task_reference.task_id,
        official_reward=reward_metrics.official_reward,
        oracle_reward=reward_metrics.oracle_reward,
        verifier_gap=reward_metrics.verifier_gap,
        false_pass=reward_metrics.false_pass,
        exploit_labels=list(reward_metrics.exploit_labels),
    )


def build_mech_interp_row(trajectory: Any) -> dict[str, Any]:
    from rewardhack_gym import build_mech_interp_record

    return build_mech_interp_record(trajectory).to_dict()


def evaluate_task_output(
    *,
    environment: Any,
    spec: EnvironmentSpec,
    task: Any,
    task_seed: int,
    completion: str,
    policy_id: str,
    include_hidden_task_metadata: bool = False,
    steps: list[dict[str, Any]] | None = None,
    annotations: dict[str, Any] | None = None,
) -> EvaluatedTaskOutput:
    task_reference = task_to_reference(task, spec, task_seed)
    trajectory = environment.evaluate_output(
        task,
        completion,
        steps=steps,
        policy_id=policy_id,
        annotations=annotations,
    )
    reward_metrics = reward_metrics_from_trajectory(trajectory)
    normalized_rollout = build_normalized_rollout_record(
        task_reference=task_reference,
        completion=completion,
        reward_metrics=reward_metrics,
    )
    trajectory_payload = trajectory.to_dict(
        include_hidden_task_metadata=include_hidden_task_metadata
    )
    return EvaluatedTaskOutput(
        task_reference=task_reference,
        reward_metrics=reward_metrics,
        normalized_rollout=normalized_rollout,
        mech_interp_row=build_mech_interp_row(trajectory),
        trajectory_payload=trajectory_payload,
    )
