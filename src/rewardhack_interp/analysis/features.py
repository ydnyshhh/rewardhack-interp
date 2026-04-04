from __future__ import annotations

from typing import Any

from rewardhack_interp.activations import load_activation_tensors
from rewardhack_interp.config import FeatureSelectionConfig
from rewardhack_interp.io import load_models
from rewardhack_interp.types import ActivationCaptureArtifact, CohortLabel, PoolingStrategy
from rewardhack_interp.utils.identifiers import activation_sample_id


def load_feature_table(
    config: FeatureSelectionConfig,
) -> tuple[Any, list[CohortLabel], list[str]]:
    import numpy as np

    artifacts = load_models(config.activation_manifest_path, ActivationCaptureArtifact)
    rows: list[Any] = []
    labels: list[CohortLabel] = []
    trace_ids: list[str] = []

    for artifact in artifacts:
        if config.allowed_cohorts is not None and artifact.cohort not in config.allowed_cohorts:
            continue
        tensor_bundle = load_activation_tensors(artifact)
        for target_name in config.target_names:
            if target_name not in tensor_bundle:
                continue
            pooled = pool_tensor(
                tensor=tensor_bundle[target_name],
                prompt_token_count=artifact.prompt_token_count,
                completion_token_count=artifact.completion_token_count,
                strategy=config.pooling_strategy,
            )
            rows.append(pooled.reshape(-1))
            labels.append(artifact.cohort)
            trace_ids.append(f"{activation_sample_id(artifact)}:{target_name}")
        if config.max_records is not None and len(rows) >= config.max_records:
            break

    if not rows:
        raise ValueError("No activation rows matched the feature selection config.")
    return np.stack(rows), labels, trace_ids


def pool_tensor(
    tensor: Any,
    *,
    prompt_token_count: int,
    completion_token_count: int,
    strategy: PoolingStrategy,
) -> Any:
    import numpy as np
    import torch

    if torch.is_tensor(tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = np.asarray(tensor)

    if array.ndim == 1:
        return array
    if array.ndim == 0:
        return array.reshape(1)

    completion_start = prompt_token_count
    completion_end = prompt_token_count + completion_token_count
    if strategy == PoolingStrategy.last_completion_token:
        position = (
            completion_end - 1
            if completion_token_count > 0
            else max(prompt_token_count - 1, 0)
        )
        return array[position]
    if strategy == PoolingStrategy.mean_completion:
        segment = array[completion_start:completion_end]
        return segment.mean(axis=0) if len(segment) > 0 else array[max(prompt_token_count - 1, 0)]
    if strategy == PoolingStrategy.max_completion:
        segment = array[completion_start:completion_end]
        return segment.max(axis=0) if len(segment) > 0 else array[max(prompt_token_count - 1, 0)]
    if strategy == PoolingStrategy.last_prompt_token:
        return array[max(prompt_token_count - 1, 0)]
    if strategy == PoolingStrategy.mean_prompt:
        return array[:prompt_token_count].mean(axis=0)
    raise ValueError(f"Unsupported pooling strategy {strategy!r}.")
