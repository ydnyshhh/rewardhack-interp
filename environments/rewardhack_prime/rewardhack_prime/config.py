from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import verifiers as vf
from pydantic import field_validator

if TYPE_CHECKING:
    from rewardhack_gym.core.config import EnvironmentConfig, ExploitabilityProfile


SUPPORTED_PROFILES: tuple[str, ...] = (
    "aligned",
    "low",
    "medium",
    "high",
    "adversarial",
)
SUPPORTED_REWARD_MODES: tuple[str, ...] = (
    "official_only",
    "oracle_upper_bound",
    "gap_penalized",
    "false_pass_penalized",
)
SUPPORTED_EXECUTION_BACKENDS: tuple[str, ...] = (
    "subprocess",
    "local_trusted",
    "docker",
    "prime_sandbox",
)


class RewardHackTasksetConfig(vf.TasksetConfig):
    family: str = "code/spec-overfit"
    profile: str = "medium"
    split: str = "eval"
    num_tasks: int = 100
    seed: int = 0
    reward_mode: str = "official_only"
    reward_penalty: float = 1.0
    include_oracle_metrics: bool = True
    execution_backend: str = "subprocess"
    timeout_seconds: float = 2.0
    memory_limit_mb: int = 256
    stdout_limit_chars: int = 20_000
    stderr_limit_chars: int = 20_000
    max_output_object_size: int = 20_000

    supported_profiles: ClassVar[tuple[str, ...]] = SUPPORTED_PROFILES
    supported_reward_modes: ClassVar[tuple[str, ...]] = SUPPORTED_REWARD_MODES
    supported_execution_backends: ClassVar[tuple[str, ...]] = SUPPORTED_EXECUTION_BACKENDS

    @field_validator("family")
    @classmethod
    def validate_family(cls, value: str) -> str:
        if not value or "/" not in value:
            raise ValueError(
                "family must be a RewardHack-Gym environment id such as 'code/spec-overfit'."
            )
        return value

    @field_validator("profile")
    @classmethod
    def validate_profile(cls, value: str) -> str:
        if value not in SUPPORTED_PROFILES:
            raise ValueError(f"profile must be one of {SUPPORTED_PROFILES}.")
        return value

    @field_validator("num_tasks")
    @classmethod
    def validate_num_tasks(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("num_tasks must be positive.")
        return value

    @field_validator("reward_mode")
    @classmethod
    def validate_reward_mode(cls, value: str) -> str:
        if value not in SUPPORTED_REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {SUPPORTED_REWARD_MODES}.")
        return value

    @field_validator("reward_penalty")
    @classmethod
    def validate_reward_penalty(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("reward_penalty must be non-negative.")
        return value

    @field_validator("execution_backend")
    @classmethod
    def validate_execution_backend(cls, value: str) -> str:
        if value not in SUPPORTED_EXECUTION_BACKENDS:
            raise ValueError(f"execution_backend must be one of {SUPPORTED_EXECUTION_BACKENDS}.")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout_seconds(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("timeout_seconds must be positive.")
        return value

    @field_validator(
        "memory_limit_mb",
        "stdout_limit_chars",
        "stderr_limit_chars",
        "max_output_object_size",
    )
    @classmethod
    def validate_positive_limits(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("execution limits must be positive.")
        return value


def exploitability_profile_from_level(level: str) -> ExploitabilityProfile:
    from rewardhack_gym.core.config import ExploitabilityProfile

    if level == "aligned":
        return ExploitabilityProfile(
            level="aligned",
            official_coverage_fraction=1.0,
            hidden_adversarial_strength=0.0,
            parser_strictness=1.0,
            process_validation_fraction=1.0,
            domain_awareness=1.0,
            perturbation_robustness=0.2,
            public_example_diversity=1.0,
            notes=("Aligned control: official verifier closely tracks the oracle surface.",),
        )
    return ExploitabilityProfile.from_level(level)


def build_environment_config(config: RewardHackTasksetConfig) -> EnvironmentConfig:
    from rewardhack_gym.core.config import EnvironmentConfig

    return EnvironmentConfig(
        seed=config.seed,
        exploitability=exploitability_profile_from_level(config.profile),
        metadata={
            "rewardhack_prime": {
                "family": config.family,
                "profile": config.profile,
                "split": config.split,
                "reward_mode": config.reward_mode,
                "reward_penalty": config.reward_penalty,
                "execution_backend": config.execution_backend,
                "timeout_seconds": config.timeout_seconds,
                "memory_limit_mb": config.memory_limit_mb,
            }
        },
    )
