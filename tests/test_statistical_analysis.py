import pandas as pd
import pytest

from link_prediction.statistical_analysis import (
    aggregate_network_metrics,
    build_method_metric_matrix,
    friedman_method_test,
    paired_rank_biserial,
    pairwise_wilcoxon_holm,
)


def build_fold_metrics() -> pd.DataFrame:
    rows = []

    method_values = {
        "first": [
            0.9,
            0.8,
            0.7,
            0.6,
        ],
        "second": [
            0.8,
            0.7,
            0.6,
            0.5,
        ],
        "third": [
            0.7,
            0.6,
            0.5,
            0.4,
        ],
    }

    for method_id, values in (
        method_values.items()
    ):
        for network_index, value in enumerate(
            values,
            start=1,
        ):
            for fold in (
                1,
                2,
            ):
                rows.append(
                    {
                        "benchmark":
                            "revision",
                        "network_id":
                            f"network_{network_index}",
                        "family":
                            "test_family",
                        "method_id":
                            method_id,
                        "fold":
                            fold,
                        "average_precision":
                            value,
                    }
                )

    return pd.DataFrame(
        rows
    )


def test_aggregate_network_metrics():
    network_metrics = (
        aggregate_network_metrics(
            build_fold_metrics(),
            "average_precision",
        )
    )

    assert len(
        network_metrics
    ) == 12

    assert set(
        network_metrics.columns
    ) == {
        "benchmark",
        "network_id",
        "family",
        "method_id",
        "average_precision",
    }


def test_friedman_method_test():
    network_metrics = (
        aggregate_network_metrics(
            build_fold_metrics(),
            "average_precision",
        )
    )

    matrix = (
        build_method_metric_matrix(
            network_metrics,
            "average_precision",
        )
    )

    result = friedman_method_test(
        matrix
    )

    assert result[
        "network_count"
    ] == 4

    assert result[
        "method_count"
    ] == 3

    assert result[
        "statistic"
    ] == pytest.approx(
        8.0
    )

    assert result[
        "p_value"
    ] < 0.05


def test_pairwise_wilcoxon_holm():
    network_metrics = (
        aggregate_network_metrics(
            build_fold_metrics(),
            "average_precision",
        )
    )

    matrix = (
        build_method_metric_matrix(
            network_metrics,
            "average_precision",
        )
    )

    comparisons = (
        pairwise_wilcoxon_holm(
            matrix
        )
    )

    assert len(
        comparisons
    ) == 3

    assert set(
        comparisons.columns
    ) == {
        "first_method",
        "second_method",
        "network_count",
        "statistic",
        "p_value",
        "median_difference",
        "rank_biserial",
        "first_wins",
        "ties",
        "second_wins",
        "adjusted_p_value",
        "reject_null",
    }

    assert set(
        comparisons[
            "network_count"
        ]
    ) == {
        4,
    }

    assert (
        comparisons[
            "adjusted_p_value"
        ]
        >= comparisons[
            "p_value"
        ]
    ).all()


def test_pairwise_wilcoxon_handles_ties():
    matrix = pd.DataFrame(
        {
            "first": [
                0.5,
                0.5,
                0.5,
            ],
            "second": [
                0.5,
                0.5,
                0.5,
            ],
        }
    )

    comparisons = (
        pairwise_wilcoxon_holm(
            matrix
        )
    )

    result = comparisons.iloc[0]

    assert result[
        "p_value"
    ] == 1.0

    assert result[
        "adjusted_p_value"
    ] == 1.0

    assert result[
        "rank_biserial"
    ] == 0.0

    assert result[
        "ties"
    ] == 3


def test_paired_rank_biserial():
    effect = paired_rank_biserial(
        pd.Series(
            [
                0.3,
                0.2,
                -0.1,
                0.0,
            ]
        ).to_numpy()
    )

    assert effect == pytest.approx(
        4.0 / 6.0
    )