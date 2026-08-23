import pandas as pd
import pytest

from link_prediction.statistical_analysis import (
    aggregate_network_metrics,
    analyze_confirmatory_metric,
    build_method_metric_matrix,
    friedman_method_test,
    load_benchmark_fold_metrics,
    mean_method_ranks,
    paired_rank_biserial,
    pairwise_wilcoxon_holm,
    run_confirmatory_analysis,
    select_confirmatory_methods,
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
                        "analysis_family":
                            "combined_family",
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


def build_methods_config() -> dict:
    return {
        "families": {
            "first_family": {
                "name":
                    "First family",
            },
            "second_family": {
                "name":
                    "Second family",
            },
        },
        "analysis_families": {
            "combined_family": {
                "name":
                    "Combined family",
                "execution_families": [
                    "first_family",
                    "second_family",
                ],
            },
        },
        "methods": {
            "first": {
                "family":
                    "first_family",
                "enabled":
                    True,
            },
            "second": {
                "family":
                    "second_family",
                "enabled":
                    True,
            },
            "third": {
                "family":
                    "second_family",
                "enabled":
                    True,
            },
        }
    }


def write_family_metrics(
    directory,
) -> None:
    fold_metrics = (
        build_fold_metrics()
    )

    family_mapping = {
        "first":
            "first_family",
        "second":
            "second_family",
        "third":
            "second_family",
    }

    fold_metrics[
        "family"
    ] = (
        fold_metrics[
            "method_id"
        ].map(
            family_mapping
        )
    )

    for family_id in (
        "first_family",
        "second_family",
    ):
        family_metrics = (
            fold_metrics[
                fold_metrics[
                    "family"
                ]
                == family_id
            ]
        )

        family_metrics.to_csv(
            directory
            / (
                "revision_"
                f"{family_id}_"
                "fold_metrics.csv"
            ),
            index=False,
        )


def test_load_benchmark_fold_metrics(
    tmp_path,
):
    write_family_metrics(
        tmp_path
    )

    fold_metrics = (
        load_benchmark_fold_metrics(
            summary_results_dir=
                tmp_path,
            methods_config=
                build_methods_config(),
        )
    )

    assert set(
        fold_metrics[
            "method_id"
        ]
    ) == {
        "first",
        "second",
        "third",
    }

    assert len(
        fold_metrics
    ) == 24


def test_load_benchmark_fold_metrics_rejects_missing_method(
    tmp_path,
):
    write_family_metrics(
        tmp_path
    )

    path = (
        tmp_path
        / (
            "revision_"
            "second_family_"
            "fold_metrics.csv"
        )
    )

    metrics = pd.read_csv(
        path
    )

    metrics = metrics[
        metrics[
            "method_id"
        ]
        != "third"
    ]

    metrics.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="enabled method registry",
    ):
        load_benchmark_fold_metrics(
            summary_results_dir=
                tmp_path,
            methods_config=
                build_methods_config(),
        )


def test_select_confirmatory_methods():
    fold_metrics = (
        build_fold_metrics()
    )

    selected = (
        select_confirmatory_methods(
            fold_metrics,
            [
                "third",
                "first",
                "second",
            ],
        )
    )

    assert set(
        selected[
            "method_id"
        ]
    ) == {
        "first",
        "second",
        "third",
    }

    assert (
        selected[
            "confirmatory_order"
        ]
        .drop_duplicates()
        .to_list()
    ) == [
        0,
        1,
        2,
    ]

    assert (
        selected[
            [
                "confirmatory_order",
                "method_id",
            ]
        ]
        .drop_duplicates()
        [
            "method_id"
        ]
        .to_list()
    ) == [
        "third",
        "first",
        "second",
    ]


def test_select_confirmatory_methods_rejects_missing_method():
    with pytest.raises(
        ValueError,
        match="missing from fold metrics",
    ):
        select_confirmatory_methods(
            build_fold_metrics(),
            [
                "first",
                "second",
                "missing",
            ],
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
        "analysis_family",
        "method_id",
        "average_precision",
    }


def test_mean_method_ranks():
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

    ranks = mean_method_ranks(
        matrix
    )

    assert ranks[
        "method_id"
    ].to_list() == [
        "first",
        "second",
        "third",
    ]

    assert ranks[
        "mean_rank"
    ].to_list() == pytest.approx(
        [
            1.0,
            2.0,
            3.0,
        ]
    )


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


def test_method_matrix_rejects_missing_values():
    network_metrics = (
        aggregate_network_metrics(
            build_fold_metrics(),
            "average_precision",
        )
    )

    incomplete = network_metrics[
        ~(
            (
                network_metrics[
                    "network_id"
                ]
                == "network_4"
            )
            & (
                network_metrics[
                    "method_id"
                ]
                == "third"
            )
        )
    ]

    with pytest.raises(
        ValueError,
        match="missing network-method values",
    ):
        build_method_metric_matrix(
            incomplete,
            "average_precision",
        )


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


def test_pairwise_wilcoxon_requires_two_networks():
    matrix = pd.DataFrame(
        {
            "first": [0.6],
            "second": [0.5],
        }
    )

    with pytest.raises(
        ValueError,
        match="at least two networks",
    ):
        pairwise_wilcoxon_holm(
            matrix
        )


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


def test_load_benchmark_fold_metrics_rejects_incomplete_blocks(
    tmp_path,
):
    write_family_metrics(
        tmp_path
    )

    path = (
        tmp_path
        / (
            "revision_"
            "second_family_"
            "fold_metrics.csv"
        )
    )

    metrics = pd.read_csv(
        path
    )

    metrics = metrics[
        ~(
            (
                metrics[
                    "method_id"
                ]
                == "third"
            )
            & (
                metrics[
                    "network_id"
                ]
                == "network_4"
            )
            & (
                metrics[
                    "fold"
                ]
                == 2
            )
        )
    ]

    metrics.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="incomplete method blocks",
    ):
        load_benchmark_fold_metrics(
            summary_results_dir=
                tmp_path,
            methods_config=
                build_methods_config(),
        )


def test_select_confirmatory_methods_requires_three_methods():
    with pytest.raises(
        ValueError,
        match="at least three methods",
    ):
        select_confirmatory_methods(
            build_fold_metrics(),
            [
                "first",
                "second",
            ],
        )

def test_select_confirmatory_methods_rejects_duplicates():
    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        select_confirmatory_methods(
            build_fold_metrics(),
            [
                "first",
                "second",
                "second",
            ],
        )


def test_analyze_confirmatory_metric():
    (
        network_metrics,
        friedman,
        mean_ranks,
        pairwise,
    ) = analyze_confirmatory_metric(
        build_fold_metrics(),
        [
            "first",
            "second",
            "third",
        ],
        "average_precision",
    )

    assert len(
        network_metrics
    ) == 12

    assert friedman.loc[
        0,
        "reject_null",
    ]

    assert friedman.loc[
        0,
        "p_value",
    ] < 0.05

    assert mean_ranks[
        "method_id"
    ].to_list() == [
        "first",
        "second",
        "third",
    ]

    assert len(
        pairwise
    ) == 3

    assert set(
        pairwise[
            "metric"
        ]
    ) == {
        "average_precision",
    }


def test_analyze_confirmatory_metric_skips_post_hoc():
    fold_metrics = (
        build_fold_metrics()
    )

    fold_metrics[
        "average_precision"
    ] = 0.5

    (
        _,
        friedman,
        mean_ranks,
        pairwise,
    ) = analyze_confirmatory_metric(
        fold_metrics,
        [
            "first",
            "second",
            "third",
        ],
        "average_precision",
    )

    assert friedman.loc[
        0,
        "statistic",
    ] == 0.0

    assert friedman.loc[
        0,
        "p_value",
    ] == 1.0

    assert not friedman.loc[
        0,
        "reject_null",
    ]

    assert set(
        mean_ranks[
            "mean_rank"
        ]
    ) == {
        2.0,
    }

    assert pairwise.empty


def test_run_confirmatory_analysis(
    tmp_path,
):
    outputs = (
        run_confirmatory_analysis(
            fold_metrics=
                build_fold_metrics(),
            method_ids=[
                "first",
                "second",
                "third",
            ],
            metrics=(
                "average_precision",
            ),
            benchmark_name=
                "revision",
            output_dir=
                tmp_path,
        )
    )

    assert set(
        outputs
    ) == {
        "network_metrics",
        "friedman",
        "mean_ranks",
        "pairwise",
    }

    assert set(
        outputs[
            "network_metrics"
        ][
            "metric"
        ]
    ) == {
        "average_precision",
    }

    assert set(
        outputs[
            "network_metrics"
        ][
            "value"
        ]
    ) == {
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
        0.4,
    }

    assert outputs[
        "friedman"
    ].loc[
        0,
        "reject_null",
    ]

    assert len(
        outputs[
            "pairwise"
        ]
    ) == 3

    for output_name in (
        outputs
    ):
        path = (
            tmp_path
            / (
                "revision_"
                "confirmatory_"
                f"{output_name}.csv"
            )
        )

        assert path.exists()

        persisted = pd.read_csv(
            path
        )

        assert len(
            persisted
        ) == len(
            outputs[
                output_name
            ]
        )