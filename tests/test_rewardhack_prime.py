from __future__ import annotations

import asyncio
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import pytest
from pydantic import BaseModel, ConfigDict
from pytest import MonkeyPatch

T = TypeVar("T")


class _TasksetConfig(BaseModel):
    pass


class _Taskset(Generic[T]):
    def __init__(self, config: T) -> None:
        self.config = config


class _EnvConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    taskset: object
    harness: object | None = None


class _Env:
    def __init__(self, taskset: object) -> None:
        self.taskset = taskset


F = TypeVar("F", bound=Callable[..., Any])


def _decorator(*args: object, **kwargs: object) -> Callable[[F], F] | F:
    if args and callable(args[0]) and not kwargs:
        return cast(F, args[0])

    def decorate(func: F) -> F:
        return func

    return decorate


def install_verifiers_stub(monkeypatch: MonkeyPatch) -> None:
    module = types.ModuleType("verifiers")
    module.TasksetConfig = _TasksetConfig
    module.Taskset = _Taskset
    module.EnvConfig = _EnvConfig
    module.Env = _Env
    module.TaskSplit = str
    module.Tasks = list
    module.reward = _decorator
    module.metric = _decorator
    monkeypatch.setitem(sys.modules, "verifiers", module)


def install_rewardhack_prime_path(monkeypatch: MonkeyPatch) -> None:
    package_path = Path(__file__).resolve().parents[1] / "environments" / "rewardhack_prime"
    monkeypatch.syspath_prepend(str(package_path))


def test_rewardhack_task_to_vf_task_hides_hidden_metadata(monkeypatch: MonkeyPatch) -> None:
    install_verifiers_stub(monkeypatch)
    install_rewardhack_prime_path(monkeypatch)

    from rewardhack_gym import create_environment
    from rewardhack_gym.core.config import EnvironmentConfig
    from rewardhack_prime.conversion import rewardhack_task_to_vf_task

    environment = create_environment("code/spec-overfit", config=EnvironmentConfig())
    task = environment.sample_task(seed=0)
    row = rewardhack_task_to_vf_task(task)
    serialized_row = json.dumps(row, sort_keys=True)

    assert row["prompt"] == [{"role": "user", "content": task.prompt}]
    assert row["answer"] == ""
    assert row["info"]["task_id"] == task.task_id
    assert row["info"]["exploit_surface"] == task.exploit_surface.to_dict()
    assert row["info"]["metadata"]["public_cases"] == json.loads(
        json.dumps(task.metadata["public_cases"])
    )
    assert "hidden_metadata" not in row
    assert "hidden_metadata" not in row["info"]
    assert "hidden_cases" not in serialized_row


def test_public_task_row_redacts_private_oracle_state(monkeypatch: MonkeyPatch) -> None:
    install_verifiers_stub(monkeypatch)
    install_rewardhack_prime_path(monkeypatch)

    from rewardhack_gym import create_environment
    from rewardhack_gym.core.config import EnvironmentConfig
    from rewardhack_prime.conversion import rewardhack_task_to_vf_task

    environment = create_environment("code/spec-overfit", config=EnvironmentConfig())
    task = environment.sample_task(seed=0)
    row = rewardhack_task_to_vf_task(task)
    serialized_row = json.dumps(row, sort_keys=True)
    prompt = row["prompt"][0]["content"]

    assert "oracle_property_cases" in task.metadata
    assert "canonical_exploit_output" in task.metadata
    assert "hidden_cases" in task.hidden_metadata
    assert "oracle_property_cases" not in serialized_row
    assert "canonical_exploit_output" not in serialized_row
    assert "hidden_cases" not in serialized_row
    assert "hidden_cases" not in prompt
    for hidden_case in task.hidden_metadata["hidden_cases"]:
        assert hidden_case["label"] not in prompt


def test_canonical_outputs_require_explicit_public_conversion(monkeypatch: MonkeyPatch) -> None:
    install_verifiers_stub(monkeypatch)
    install_rewardhack_prime_path(monkeypatch)

    from rewardhack_gym import create_environment
    from rewardhack_gym.core.config import EnvironmentConfig
    from rewardhack_prime.conversion import rewardhack_task_to_vf_task

    environment = create_environment("code/spec-overfit", config=EnvironmentConfig())
    task = environment.sample_task(seed=0)

    default_row = rewardhack_task_to_vf_task(task)
    allowed_row = rewardhack_task_to_vf_task(task, include_canonical_outputs=True)

    assert "canonical_exploit_output" not in default_row["info"]["metadata"]
    assert (
        allowed_row["info"]["metadata"]["canonical_exploit_output"]
        == task.metadata["canonical_exploit_output"]
    )


def test_aligned_profile_builds_clean_control_config(monkeypatch: MonkeyPatch) -> None:
    install_verifiers_stub(monkeypatch)
    install_rewardhack_prime_path(monkeypatch)

    from rewardhack_prime.config import RewardHackTasksetConfig, build_environment_config

    config = RewardHackTasksetConfig(profile="aligned")
    environment_config = build_environment_config(config)

    assert environment_config.exploitability.level == "aligned"
    assert environment_config.exploitability.official_coverage_fraction == 1.0
    assert environment_config.exploitability.hidden_adversarial_strength == 0.0
    assert environment_config.exploitability.process_validation_fraction == 1.0


def test_taskset_scores_official_and_oracle_without_exposing_hidden_metadata(
    monkeypatch: MonkeyPatch,
) -> None:
    install_verifiers_stub(monkeypatch)
    install_rewardhack_prime_path(monkeypatch)

    from rewardhack_prime.config import RewardHackTasksetConfig
    from rewardhack_prime.taskset import RewardHackTaskset

    taskset = RewardHackTaskset(
        config=RewardHackTasksetConfig(
            family="code/spec-overfit",
            profile="medium",
            num_tasks=1,
            seed=0,
        )
    )
    row = taskset.rows()[0]
    rewardhack_task = taskset.private_tasks.get(row["info"]["task_id"])
    completion = taskset.environment.canonical_true_output(rewardhack_task)
    state: dict[str, Any] = {"completion": [{"role": "assistant", "content": completion}]}

    reward = asyncio.run(
        taskset.official_reward(
            row,
            state,
        )
    )
    cached_scores = state["rewardhack_trajectory"]
    oracle_score = asyncio.run(taskset.oracle_score(row, state))

    assert reward == cached_scores.official_score
    assert oracle_score == 1.0
    assert "hidden_metadata" not in row
    assert "hidden_metadata" not in row["info"]


def test_private_store_keeps_full_oracle_task_off_public_row(
    monkeypatch: MonkeyPatch,
) -> None:
    install_verifiers_stub(monkeypatch)
    install_rewardhack_prime_path(monkeypatch)

    from rewardhack_prime.config import RewardHackTasksetConfig
    from rewardhack_prime.taskset import RewardHackTaskset

    taskset = RewardHackTaskset(
        config=RewardHackTasksetConfig(
            family="code/spec-overfit",
            profile="medium",
            num_tasks=1,
            seed=0,
        )
    )
    row = taskset.rows()[0]
    private_task = taskset.private_tasks.get(row["info"]["task_id"])

    assert "hidden_cases" in private_task.hidden_metadata
    assert "hidden_cases" not in json.dumps(row, sort_keys=True)


def test_trajectory_hidden_task_metadata_flag(monkeypatch: MonkeyPatch) -> None:
    install_verifiers_stub(monkeypatch)
    install_rewardhack_prime_path(monkeypatch)

    from rewardhack_gym import create_environment
    from rewardhack_gym.core.config import EnvironmentConfig

    environment = create_environment("code/spec-overfit", config=EnvironmentConfig())
    task = environment.sample_task(seed=0)
    trajectory = environment.evaluate_output(task, environment.canonical_true_output(task))

    public_payload = trajectory.to_dict()
    hidden_payload = trajectory.to_dict(include_hidden_task_metadata=True)

    assert "hidden_metadata" not in public_payload["task"]
    assert "hidden_metadata" in hidden_payload["task"]
    assert "hidden_cases" in hidden_payload["task"]["hidden_metadata"]
    assert "hidden_cases" not in trajectory.prompt


def test_redacted_score_trajectory_omits_private_task_and_oracle_diagnostics(
    monkeypatch: MonkeyPatch,
) -> None:
    install_verifiers_stub(monkeypatch)
    install_rewardhack_prime_path(monkeypatch)

    from rewardhack_gym import create_environment
    from rewardhack_gym.core.config import EnvironmentConfig
    from rewardhack_prime.scoring import rewardhack_scores_from_trajectory

    environment = create_environment("code/spec-overfit", config=EnvironmentConfig())
    task = environment.sample_task(seed=0)
    trajectory = environment.evaluate_output(task, environment.canonical_exploit_output(task))
    scores = rewardhack_scores_from_trajectory(trajectory)
    serialized_payload = json.dumps(scores.trajectory, sort_keys=True)

    assert "hidden_metadata" not in serialized_payload
    assert "hidden_cases" not in serialized_payload
    assert "oracle_property_cases" not in serialized_payload
    assert "canonical_exploit_output" not in serialized_payload
    assert scores.trajectory["oracle_result"]["diagnostics"] == {}


def test_default_reward_mode_is_official_only(monkeypatch: MonkeyPatch) -> None:
    install_verifiers_stub(monkeypatch)
    install_rewardhack_prime_path(monkeypatch)

    from rewardhack_prime.config import RewardHackTasksetConfig

    config = RewardHackTasksetConfig()

    assert config.reward_mode == "official_only"


def test_reward_modes_are_explicit_and_prime_safe(monkeypatch: MonkeyPatch) -> None:
    install_verifiers_stub(monkeypatch)
    install_rewardhack_prime_path(monkeypatch)

    from rewardhack_prime.config import RewardHackTasksetConfig
    from rewardhack_prime.scoring import RewardHackScores, scalarize_reward

    scores = RewardHackScores(
        task_id="task-1",
        completion="answer",
        official_score=0.9,
        oracle_score=0.2,
        hack_gap=0.7,
        false_pass=True,
        official_passed=True,
        oracle_passed=False,
    )

    assert scalarize_reward(scores, "official_only", reward_penalty=0.5) == 0.9
    assert scalarize_reward(scores, "oracle_upper_bound", reward_penalty=0.5) == 0.2
    assert scalarize_reward(scores, "gap_penalized", reward_penalty=0.5) == 0.55
    assert scalarize_reward(scores, "false_pass_penalized", reward_penalty=0.5) == 0.4
    with pytest.raises(ValueError, match="reward_mode"):
        RewardHackTasksetConfig(reward_mode="oracle_only")


def test_metrics_reuse_one_cached_rewardhack_trajectory(monkeypatch: MonkeyPatch) -> None:
    install_verifiers_stub(monkeypatch)
    install_rewardhack_prime_path(monkeypatch)

    from rewardhack_prime.config import RewardHackTasksetConfig
    from rewardhack_prime.scoring import TRAJECTORY_CACHE_KEY
    from rewardhack_prime.taskset import RewardHackTaskset

    taskset = RewardHackTaskset(
        config=RewardHackTasksetConfig(
            family="code/spec-overfit",
            profile="medium",
            num_tasks=1,
            seed=0,
        )
    )
    row = taskset.rows()[0]
    rewardhack_task = taskset.private_tasks.get(row["info"]["task_id"])
    completion = taskset.environment.canonical_true_output(rewardhack_task)
    state: dict[str, Any] = {"completion": [{"role": "assistant", "content": completion}]}

    original_evaluate_output = taskset.environment.evaluate_output
    call_count = 0

    def counted_evaluate_output(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return original_evaluate_output(*args, **kwargs)

    monkeypatch.setattr(taskset.environment, "evaluate_output", counted_evaluate_output)

    reward = asyncio.run(taskset.official_reward(row, state))
    oracle_score = asyncio.run(taskset.oracle_score(row, state))
    hack_gap = asyncio.run(taskset.hack_gap(row, state))
    false_pass = asyncio.run(taskset.false_pass(row, state))

    assert call_count == 1
    assert TRAJECTORY_CACHE_KEY in state
    assert reward == state[TRAJECTORY_CACHE_KEY].official_score
    assert oracle_score == state[TRAJECTORY_CACHE_KEY].oracle_score
    assert hack_gap == state[TRAJECTORY_CACHE_KEY].hack_gap
    assert false_pass == float(state[TRAJECTORY_CACHE_KEY].false_pass)
