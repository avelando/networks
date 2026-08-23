import pandas as pd
import pytest

from link_prediction.runtime_analysis import (
    aggregate_family_runtime,
    aggregate_network_runtime,
    extract_family_fold_runtime,
    run_runtime_analysis,
)


def build_runtime_metrics() -> pd.DataFrame:
    rows = []

    for network_id in (
        "first_network",
        "second_network",
    ):
        for fold in (
            1,
            2,
        ):
            for method_id in (
                "cn",
                "aa",
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
                            method_id,
                        "candidate_count":
                            100,
                        "family_scoring_seconds":
                            float(fold),
                    }
                )

    return pd.DataFrame(
        rows
    )


def test_extract_family_fold_runtime():
    runtime = (
        extract_family_fold_runtime(
            build_runtime_metrics()
        )
    )

    assert len(runtime) == 4

    assert (
        runtime[
            "method_count"
        ] == 2
    ).all()

    assert runtime[
        "candidates_per_second"
    ].tolist() == pytest.approx(
        [
            100.0,
            50.0,
            100.0,
            50.0,
        ]
    )


def test_extract_family_fold_runtime_rejects_inconsistency():
    fold_metrics = (
        build_runtime_metrics()
    )

    fold_metrics.loc[
        0,
        "family_scoring_seconds",
    ] = 3.0

    with pytest.raises(
        ValueError,
        match="inconsistent",
    ):
        extract_family_fold_runtime(
            fold_metrics
        )


def test_aggregate_runtime():
    family_folds = (
        extract_family_fold_runtime(
            build_runtime_metrics()
        )
    )

    network_runtime = (
        aggregate_network_runtime(
            family_folds
        )
    )

    assert len(network_runtime) == 2

    assert network_runtime[
        "mean_scoring_seconds"
    ].tolist() == pytest.approx(
        [
            1.5,
            1.5,
        ]
    )

    family_runtime = (
        aggregate_family_runtime(
            network_runtime
        )
    )

    assert len(family_runtime) == 1

    row = family_runtime.iloc[0]

    assert row[
        "network_count"
    ] == 2

    assert row[
        "method_count"
    ] == 2

    assert row[
        "mean_scoring_seconds"
    ] == pytest.approx(
        1.5
    )

    assert row[
        "mean_candidates_per_second"
    ] == pytest.approx(
        75.0
    )


def test_run_runtime_analysis_writes_outputs(
    tmp_path,
):
    outputs = run_runtime_analysis(
        fold_metrics=
            build_runtime_metrics(),
        output_dir=
            tmp_path,
    )

    assert set(outputs) == {
        "family_folds",
        "network_runtime",
        "family_runtime",
    }

    assert (
        tmp_path
        / "standard_family_fold_runtime.csv"
    ).exists()

    assert (
        tmp_path
        / "standard_family_runtime_by_network.csv"
    ).exists()

    assert (
        tmp_path
        / "standard_family_runtime_summary.csv"
    ).exists()

    assert not list(
        tmp_path.glob(
            "*.part"
        )
    )