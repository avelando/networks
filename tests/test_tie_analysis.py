import pandas as pd
import pytest

from link_prediction.tie_analysis import (
    aggregate_method_tie_diagnostics,
    aggregate_network_tie_diagnostics,
    run_tie_analysis,
)


def build_tie_fold_metrics() -> pd.DataFrame:
    rows = []

    for network_id in (
        "first_network",
        "second_network",
    ):
        for fold, boundary_tie in (
            (1, True),
            (2, False),
        ):
            rows.append(
                {
                    "benchmark":
                        "standard",
                    "network_id":
                        network_id,
                    "network":
                        network_id,
                    "domain":
                        "social",
                    "fold":
                        fold,
                    "family":
                        "local_similarity",
                    "analysis_family":
                        "standard_local",
                    "method_id":
                        "cn",
                    "method":
                        "CN",
                    "distinct_score_count":
                        3 + fold,
                    "tie_group_count":
                        2 - fold,
                    "tied_candidate_count":
                        2 if boundary_tie else 0,
                    "tied_candidate_ratio":
                        (
                            0.5
                            if boundary_tie
                            else 0.0
                        ),
                    "largest_tie_group":
                        (
                            2
                            if boundary_tie
                            else 1
                        ),
                    "cutoff_tie_size":
                        (
                            2
                            if boundary_tie
                            else 1
                        ),
                    "cutoff_tie_crosses_boundary":
                        boundary_tie,
                }
            )

    return pd.DataFrame(
        rows
    )


def test_aggregate_network_tie_diagnostics():
    summary = (
        aggregate_network_tie_diagnostics(
            build_tie_fold_metrics()
        )
    )

    assert len(summary) == 2

    assert (
        summary[
            "fold_count"
        ] == 2
    ).all()

    assert summary[
        "mean_tied_candidate_ratio"
    ].tolist() == pytest.approx(
        [
            0.25,
            0.25,
        ]
    )

    assert (
        summary[
            "cutoff_boundary_tie_count"
        ] == 1
    ).all()

    assert summary[
        "cutoff_boundary_tie_rate"
    ].tolist() == pytest.approx(
        [
            0.5,
            0.5,
        ]
    )


def test_aggregate_method_tie_diagnostics():
    network_ties = (
        aggregate_network_tie_diagnostics(
            build_tie_fold_metrics()
        )
    )

    summary = (
        aggregate_method_tie_diagnostics(
            network_ties
        )
    )

    assert len(summary) == 1

    row = summary.iloc[0]

    assert row[
        "network_count"
    ] == 2

    assert row[
        "mean_tied_candidate_ratio"
    ] == pytest.approx(
        0.25
    )

    assert row[
        "mean_cutoff_boundary_tie_rate"
    ] == pytest.approx(
        0.5
    )


def test_run_tie_analysis_writes_outputs(
    tmp_path,
):
    outputs = run_tie_analysis(
        fold_metrics=
            build_tie_fold_metrics(),
        output_dir=
            tmp_path,
    )

    assert set(outputs) == {
        "network_ties",
        "method_ties",
    }

    assert (
        tmp_path
        / "standard_network_tie_diagnostics.csv"
    ).exists()

    assert (
        tmp_path
        / "standard_method_tie_diagnostics.csv"
    ).exists()

    assert not list(
        tmp_path.glob(
            "*.part"
        )
    )