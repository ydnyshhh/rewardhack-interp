from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import verifiers as vf

from rewardhack_prime.config import RewardHackTasksetConfig
from rewardhack_prime.taskset import RewardHackTaskset


def load_taskset(config: RewardHackTasksetConfig) -> RewardHackTaskset:
    return RewardHackTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    taskset = load_taskset(_coerce_taskset_config(config.taskset))
    return vf.Env(taskset=taskset)


def _coerce_taskset_config(config: Any) -> RewardHackTasksetConfig:
    if isinstance(config, RewardHackTasksetConfig):
        return config
    if isinstance(config, Mapping):
        return RewardHackTasksetConfig.model_validate(dict(config))
    model_dump = getattr(config, "model_dump", None)
    if callable(model_dump):
        return RewardHackTasksetConfig.model_validate(model_dump())
    raise TypeError(
        "config.taskset must be a RewardHackTasksetConfig or a mapping that can validate as one."
    )
