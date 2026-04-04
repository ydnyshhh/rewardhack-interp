from __future__ import annotations

import re

from rewardhack_interp.types import ActivationCaptureArtifact, TrajectoryArtifact


def derive_sample_id(*, trace_id: str, completion_index: int) -> str:
    return f"{trace_id}__completion_{completion_index}"


def trajectory_sample_id(record: TrajectoryArtifact) -> str:
    if record.sample_id:
        return record.sample_id
    return derive_sample_id(
        trace_id=record.trace_id,
        completion_index=record.generation.completion_index,
    )


def activation_sample_id(artifact: ActivationCaptureArtifact) -> str:
    if artifact.sample_id:
        return artifact.sample_id
    return derive_sample_id(
        trace_id=artifact.trace_id,
        completion_index=artifact.completion_index or 0,
    )


def safe_artifact_stem(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "sample"
