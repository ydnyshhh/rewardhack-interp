from __future__ import annotations

import numpy as np

from rewardhack_interp.analysis.features import pool_tensor
from rewardhack_interp.types import PoolingStrategy


def test_last_completion_pooling() -> None:
    tensor = np.arange(15, dtype=float).reshape(5, 3)
    pooled = pool_tensor(
        tensor=tensor,
        prompt_token_count=2,
        completion_token_count=3,
        strategy=PoolingStrategy.last_completion_token,
    )
    assert pooled.tolist() == [12.0, 13.0, 14.0]


def test_mean_completion_pooling() -> None:
    tensor = np.arange(15, dtype=float).reshape(5, 3)
    pooled = pool_tensor(
        tensor=tensor,
        prompt_token_count=2,
        completion_token_count=3,
        strategy=PoolingStrategy.mean_completion,
    )
    assert pooled.tolist() == [9.0, 10.0, 11.0]
