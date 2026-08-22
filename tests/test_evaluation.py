from pathlib import Path

import pandas as pd

from link_prediction.evaluation import (
    evaluate_score_table,
    load_candidate_table,
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