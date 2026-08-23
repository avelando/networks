import pytest

from link_prediction.metrics import (
    evaluate_ranking,
    rank_candidates,
)


def test_rank_candidates_uses_candidate_id_for_ties():
    ranked = rank_candidates(
        labels=[
            1,
            1,
            0,
        ],
        scores=[
            0.9,
            0.5,
            0.5,
        ],
        candidate_ids=[
            10,
            3,
            2,
        ],
    )

    assert ranked[
        "candidate_id"
    ].tolist() == [
        10,
        2,
        3,
    ]

    assert ranked[
        "label"
    ].tolist() == [
        1,
        0,
        1,
    ]


def test_evaluate_ranking_matches_rank_based_protocol():
    metrics = evaluate_ranking(
        labels=[
            1,
            0,
            1,
            0,
        ],
        scores=[
            0.9,
            0.8,
            0.8,
            0.1,
        ],
        candidate_ids=[
            0,
            1,
            2,
            3,
        ],
        cutoff=2,
    )

    assert metrics[
        "average_precision"
    ] == pytest.approx(
        5 / 6
    )

    assert metrics[
        "roc_auc"
    ] == pytest.approx(
        0.875
    )

    assert metrics[
        "precision"
    ] == pytest.approx(
        0.5
    )

    assert metrics[
        "recall"
    ] == pytest.approx(
        0.5
    )

    assert metrics[
        "f1"
    ] == pytest.approx(
        0.5
    )

    assert metrics[
        "ndcg"
    ] == pytest.approx(
        0.6131471927654584
    )


def test_cutoff_defaults_to_number_of_positives():
    metrics = evaluate_ranking(
        labels=[
            1,
            0,
            1,
            0,
        ],
        scores=[
            0.9,
            0.8,
            0.7,
            0.1,
        ],
        candidate_ids=[
            0,
            1,
            2,
            3,
        ],
    )

    assert metrics[
        "cutoff"
    ] == 2


def test_average_precision_is_invariant_to_tie_break_order():
    first = evaluate_ranking(
        labels=[
            1,
            0,
            0,
            1,
        ],
        scores=[
            1.0,
            0.5,
            0.5,
            0.5,
        ],
        candidate_ids=[
            0,
            1,
            2,
            3,
        ],
    )

    second = evaluate_ranking(
        labels=[
            1,
            0,
            0,
            1,
        ],
        scores=[
            1.0,
            0.5,
            0.5,
            0.5,
        ],
        candidate_ids=[
            0,
            3,
            2,
            1,
        ],
    )

    assert (
        first["average_precision"]
        == pytest.approx(
            second[
                "average_precision"
            ]
        )
    )

    assert (
        first["precision"]
        != second["precision"]
    )


def test_evaluate_ranking_reports_tie_diagnostics():
    metrics = evaluate_ranking(
        labels=[
            1,
            0,
            1,
            0,
        ],
        scores=[
            0.9,
            0.8,
            0.8,
            0.1,
        ],
        candidate_ids=[
            0,
            1,
            2,
            3,
        ],
        cutoff=2,
    )

    assert metrics[
        "distinct_score_count"
    ] == 3

    assert metrics[
        "tie_group_count"
    ] == 1

    assert metrics[
        "tied_candidate_count"
    ] == 2

    assert metrics[
        "tied_candidate_ratio"
    ] == pytest.approx(
        0.5
    )

    assert metrics[
        "largest_tie_group"
    ] == 2

    assert metrics[
        "cutoff_score"
    ] == pytest.approx(
        0.8
    )

    assert metrics[
        "cutoff_tie_size"
    ] == 2

    assert metrics[
        "cutoff_tie_positive_count"
    ] == 1

    assert metrics[
        "cutoff_tie_negative_count"
    ] == 1

    assert metrics[
        "cutoff_slots_in_tie"
    ] == 1

    assert metrics[
        "cutoff_tie_crosses_boundary"
    ] is True


def test_evaluate_ranking_reports_no_cutoff_boundary_tie():
    metrics = evaluate_ranking(
        labels=[
            1,
            0,
            1,
            0,
        ],
        scores=[
            0.9,
            0.8,
            0.7,
            0.1,
        ],
        candidate_ids=[
            0,
            1,
            2,
            3,
        ],
        cutoff=2,
    )

    assert metrics[
        "tie_group_count"
    ] == 0

    assert metrics[
        "tied_candidate_count"
    ] == 0

    assert metrics[
        "tied_candidate_ratio"
    ] == pytest.approx(
        0.0
    )

    assert metrics[
        "largest_tie_group"
    ] == 1

    assert metrics[
        "cutoff_tie_size"
    ] == 1

    assert metrics[
        "cutoff_slots_in_tie"
    ] == 1

    assert metrics[
        "cutoff_tie_crosses_boundary"
    ] is False