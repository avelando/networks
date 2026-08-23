from pathlib import Path

import pandas as pd

from link_prediction.evaluation import (
    evaluate_score_table,
    load_candidate_table,
    summarize_fold_metrics,
)


def test_load_candidate_table_preserves_node_ids_as_strings(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "candidates.csv"
    )

    pd.DataFrame(
        {
            "candidate_id": [
                0,
                1,
            ],
            "source": [
                1,
                10,
            ],
            "target": [
                2,
                20,
            ],
            "label": [
                1,
                0,
            ],
        }
    ).to_csv(
        path,
        index=False,
    )

    dataframe = (
        load_candidate_table(
            path
        )
    )

    assert dataframe[
        "source"
    ].tolist() == [
        "1",
        "10",
    ]

    assert dataframe[
        "target"
    ].tolist() == [
        "2",
        "20",
    ]


def test_evaluate_score_table_returns_one_row_per_method():
    candidates = pd.DataFrame(
        {
            "candidate_id": [
                0,
                1,
                2,
                3,
            ],
            "source": [
                "0",
                "0",
                "1",
                "1",
            ],
            "target": [
                "2",
                "3",
                "2",
                "3",
            ],
            "label": [
                1,
                0,
                1,
                0,
            ],
        }
    )

    scores = pd.DataFrame(
        {
            "cn": [
                0.9,
                0.8,
                0.7,
                0.1,
            ],
            "aa": [
                0.8,
                0.7,
                0.6,
                0.2,
            ],
        }
    )

    result = (
        evaluate_score_table(
            candidates=
                candidates,
            scores=
                scores,
            method_ids=[
                "cn",
                "aa",
            ],
        )
    )

    assert result[
        "method_id"
    ].tolist() == [
        "cn",
        "aa",
    ]

    assert (
        result[
            "cutoff"
        ] == 2
    ).all()


def test_summarize_fold_metrics_includes_tie_diagnostics():
    candidates = pd.DataFrame(
        {
            "candidate_id": [
                0,
                1,
                2,
                3,
            ],
            "source": [
                "0",
                "0",
                "1",
                "1",
            ],
            "target": [
                "2",
                "3",
                "2",
                "3",
            ],
            "label": [
                1,
                0,
                1,
                0,
            ],
        }
    )

    score_tables = [
        pd.DataFrame(
            {
                "cn": [
                    0.9,
                    0.8,
                    0.8,
                    0.1,
                ],
            }
        ),
        pd.DataFrame(
            {
                "cn": [
                    0.9,
                    0.8,
                    0.7,
                    0.1,
                ],
            }
        ),
    ]

    fold_rows = []

    for (
        fold_number,
        scores,
    ) in enumerate(
        score_tables,
        start=1,
    ):
        metrics = (
            evaluate_score_table(
                candidates=
                    candidates,
                scores=
                    scores,
                method_ids=[
                    "cn",
                ],
            )
        )

        metrics.insert(
            0,
            "method",
            "CN",
        )

        metrics.insert(
            0,
            "family",
            "local_similarity",
        )

        metrics.insert(
            0,
            "domain",
            "social",
        )

        metrics.insert(
            0,
            "network",
            "example",
        )

        metrics.insert(
            0,
            "network_id",
            "example",
        )

        metrics.insert(
            0,
            "fold",
            fold_number,
        )

        metrics.insert(
            0,
            "benchmark",
            "standard",
        )

        fold_rows.append(
            metrics
        )

    summary = (
        summarize_fold_metrics(
            pd.concat(
                fold_rows,
                ignore_index=True,
            )
        )
    )

    assert len(summary) == 1

    row = summary.iloc[0]

    assert row[
        "distinct_score_count_mean"
    ] == 3.5

    assert row[
        "tie_group_count_mean"
    ] == 0.5

    assert row[
        "tied_candidate_count_mean"
    ] == 1.0

    assert row[
        "tied_candidate_ratio_mean"
    ] == 0.25

    assert row[
        "largest_tie_group_mean"
    ] == 1.5

    assert row[
        "cutoff_tie_size_mean"
    ] == 1.5

    assert row[
        "cutoff_tie_crosses_boundary_mean"
    ] == 0.5