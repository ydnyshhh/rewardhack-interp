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
    "oracle_only",
    "gap_aware",
    "anti_hack",
)


class RewardHackTasksetConfig(vf.TasksetConfig):
    family: str = "code/spec-overfit"
    profile: str = "medium"
    split: str = "eval"
    num_tasks: int = 100
    seed: int = 0
    reward_mode: str = "official_only"
    include_oracle_metrics: bool = True

    supported_profiles: ClassVar[tuple[str, ...]] = SUPPORTED_PROFILES
    supported_reward_modes: ClassVar[tuple[str, ...]] = SUPPORTED_REWARD_MODES

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
            }
        },
    )
