import numpy as np
import pytest

from rewardhack_interp.experiments.experiment_one import (
    build_probe_splits,
    clustering_purity,
    parse_layer_index,
    summarize_matched_pair_rows,
    summarize_metric_distribution,
)
from rewardhack_interp.types import ExperimentOneSplitStrategy


def test_parse_layer_index_for_hidden_states() -> None:
    assert parse_layer_index("hidden_state.layer_0") == 0
    assert parse_layer_index("hidden_state.layer_17") == 17
    assert parse_layer_index("hidden_state.embed") == -1


def test_clustering_purity() -> None:
    purity = clustering_purity(
        {
            "0": {"genuine_success": 4, "reward_hack": 1},
            "1": {"reward_hack": 3, "genuine_failure": 1},
        }
    )
    assert purity == pytest.approx(7 / 9)


def test_summarize_matched_pair_rows() -> None:
    summaries = summarize_matched_pair_rows(
        [
            {"match_level": "exact-task", "pair_group_id": "group-a"},
            {"match_level": "exact-task", "pair_group_id": "group-a"},
            {"match_level": "scenario", "pair_group_id": "group-b"},
            {"match_level": "scenario", "pair_group_id": "group-c"},
        ]
    )

    assert [
        (summary.match_level, summary.n_pairs, summary.n_pair_groups) for summary in summaries
    ] == [
        ("exact-task", 2, 1),
        ("scenario", 2, 2),
    ]


def test_build_probe_splits_respects_explicit_seed_holdout() -> None:
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    task_seeds = [10, 10, 11, 11, 12, 12, 13, 13]

    splits = build_probe_splits(
        y=y,
        task_seeds=task_seeds,
        test_size=0.25,
        split_strategy=ExperimentOneSplitStrategy.task_seed_holdout,
        split_random_seeds=[0, 1, 2],
        train_task_seeds=[10, 11, 12],
        eval_task_seeds=[13],
    )

    assert len(splits) == 1
    train_task_seed_slice = {task_seeds[index] for index in splits[0].train_indices}
    test_task_seed_slice = {task_seeds[index] for index in splits[0].test_indices}
    assert train_task_seed_slice == {10, 11, 12}
    assert test_task_seed_slice == {13}


def test_summarize_metric_distribution_computes_confidence_bounds() -> None:
    summary = summarize_metric_distribution(
        values=[0.6, 0.7, 0.8, 0.9],
        confidence_level=0.95,
        point_estimate=0.75,
    )

    assert summary["mean"] == pytest.approx(0.75)
    assert summary["std"] is not None
    assert summary["ci_low"] is not None
    assert summary["ci_high"] is not None
    assert summary["ci_low"] <= summary["mean"] <= summary["ci_high"]
