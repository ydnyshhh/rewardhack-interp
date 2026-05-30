from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import verifiers as vf

from rewardhack_prime.config import RewardHackTasksetConfig, build_environment_config
from rewardhack_prime.conversion import rewardhack_task_to_vf_task
from rewardhack_prime.scoring import RewardHackScores, scalarize_reward, score_state
from rewardhack_prime.store import PrivateTaskStore


class RewardHackTaskset(vf.Taskset):
    config: RewardHackTasksetConfig

    def __init__(self, config: RewardHackTasksetConfig) -> None:
        super().__init__(config=config)
        self._environment: Any | None = None
        self.private_tasks = PrivateTaskStore()

    @property
    def environment(self) -> Any:
        if self._environment is None:
            from rewardhack_gym import create_environment

            self._environment = create_environment(
                self.config.family,
                config=build_environment_config(self.config),
            )
        return self._environment

    def rows(self) -> list[dict[str, Any]]:
        return list(self.load_tasks())

    def load_eval_tasks(self) -> vf.Tasks:
        return self.load_tasks(split=self.config.split)

    def load_tasks(self, split: vf.TaskSplit | None = None) -> vf.Tasks:
        if split is not None and str(split) != self.config.split:
            return []

        rows: list[dict[str, Any]] = []
        self.private_tasks.clear()
        for offset in range(self.config.num_tasks):
            seed = self.config.seed + offset
            task = self.environment.sample_task(seed=seed)
            row = rewardhack_task_to_vf_task(task)
            row["split"] = self.config.split
            row["max_turns"] = 1
            rows.append(row)
            self.private_tasks.add(task)
        return rows

    def _resolve_rewardhack_task(self, task: Mapping[str, Any]) -> Any:
        task_id = _extract_task_id(task)
        if task_id is None:
            raise KeyError("Verifiers task is missing info.task_id.")
        try:
            return self.private_tasks.get(task_id)
        except KeyError:
            self.rows()
        try:
            return self.private_tasks.get(task_id)
        except KeyError as exc:
            raise KeyError(
                f"RewardHack task {task_id!r} is not in this taskset cache. "
                "Make sure the task row came from this RewardHackTaskset instance."
            ) from exc

    async def _score(self, task: Mapping[str, Any], state: Any) -> RewardHackScores:
        rewardhack_task = self._resolve_rewardhack_task(task)
        return await score_state(
            environment=self.environment,
            task=rewardhack_task,
            state=state,
        )

    @vf.reward(weight=1.0)
    async def official_reward(self, task: Mapping[str, Any], state: Any) -> float:
        scores = await self._score(task, state)
        return scalarize_reward(
            scores,
            self.config.reward_mode,
            reward_penalty=self.config.reward_penalty,
        )

    @vf.metric
    async def oracle_score(self, task: Mapping[str, Any], state: Any) -> float:
        if not self.config.include_oracle_metrics:
            return 0.0
        scores = await self._score(task, state)
        return scores.oracle_score

    @vf.metric
    async def hack_gap(self, task: Mapping[str, Any], state: Any) -> float:
        if not self.config.include_oracle_metrics:
            return 0.0
        scores = await self._score(task, state)
        return scores.hack_gap

    @vf.metric
    async def false_pass(self, task: Mapping[str, Any], state: Any) -> float:
        if not self.config.include_oracle_metrics:
            return 0.0
        scores = await self._score(task, state)
        return float(scores.false_pass)


def _extract_task_id(task: Mapping[str, Any]) -> str | None:
    info = task.get("info", {})
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except json.JSONDecodeError:
            return None
    if not isinstance(info, Mapping):
        return None

    direct_task_id = info.get("task_id")
    if direct_task_id is not None:
        return str(direct_task_id)

    nested_task = info.get("task")
    if isinstance(nested_task, Mapping):
        nested_info = nested_task.get("info", {})
        if isinstance(nested_info, Mapping) and nested_info.get("task_id") is not None:
            return str(nested_info["task_id"])
    return None
