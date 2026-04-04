import pytest

from rewardhack_interp.experiments.experiment_one import (
    clustering_purity,
    parse_layer_index,
    summarize_matched_pair_rows,
)


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
