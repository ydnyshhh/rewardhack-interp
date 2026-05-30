from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rewardhack_gym.core.models import Task

PRIVATE_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "hidden",
        "hidden_cases",
        "hidden_metadata",
        "oracle_cases",
        "oracle_property_cases",
        "oracle_invariant_cases",
        "canonical_true_output",
        "canonical_exploit_output",
    }
)
CANONICAL_OUTPUT_KEYS: frozenset[str] = frozenset(
    {
        "canonical_true_output",
        "canonical_exploit_output",
    }
)


def public_task_metadata(
    metadata: Mapping[str, Any],
    *,
    include_canonical_outputs: bool = False,
) -> dict[str, Any]:
    return {
        str(key): _redact_private_metadata_value(
            value,
            include_canonical_outputs=include_canonical_outputs,
        )
        for key, value in metadata.items()
        if not _is_private_metadata_key(
            str(key),
            include_canonical_outputs=include_canonical_outputs,
        )
    }


def public_task_info(
    task: Task,
    *,
    include_canonical_outputs: bool = False,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "family": task.family,
        "difficulty": task.difficulty,
        "expected_interface": task.expected_interface,
        "tags": list(task.tags),
        "metadata": public_task_metadata(
            task.metadata,
            include_canonical_outputs=include_canonical_outputs,
        ),
        "exploit_surface": task.exploit_surface.to_dict(),
    }


def public_task_payload(
    task: Task,
    *,
    include_canonical_outputs: bool = False,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "family": task.family,
        "prompt": task.prompt,
        "expected_interface": task.expected_interface,
        "difficulty": task.difficulty,
        "metadata": public_task_metadata(
            task.metadata,
            include_canonical_outputs=include_canonical_outputs,
        ),
        "exploit_surface": task.exploit_surface.to_dict(),
        "tags": list(task.tags),
    }


def rewardhack_task_to_vf_task(
    task: Task,
    *,
    include_canonical_outputs: bool = False,
) -> dict[str, Any]:
    return {
        "prompt": [{"role": "user", "content": task.prompt}],
        "answer": "",
        "info": public_task_info(
            task,
            include_canonical_outputs=include_canonical_outputs,
        ),
    }


def _redact_private_metadata_value(
    value: Any,
    *,
    include_canonical_outputs: bool,
) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _redact_private_metadata_value(
                item,
                include_canonical_outputs=include_canonical_outputs,
            )
            for key, item in value.items()
            if not _is_private_metadata_key(
                str(key),
                include_canonical_outputs=include_canonical_outputs,
            )
        }
    if isinstance(value, tuple):
        return [
            _redact_private_metadata_value(
                item,
                include_canonical_outputs=include_canonical_outputs,
            )
            for item in value
        ]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [
            _redact_private_metadata_value(
                item,
                include_canonical_outputs=include_canonical_outputs,
            )
            for item in value
        ]
    return value


def _is_private_metadata_key(
    key: str,
    *,
    include_canonical_outputs: bool,
) -> bool:
    normalized = key.lower()
    if include_canonical_outputs and normalized in CANONICAL_OUTPUT_KEYS:
        return False
    return normalized in PRIVATE_METADATA_KEYS
