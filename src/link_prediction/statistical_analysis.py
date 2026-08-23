from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import (
    friedmanchisquare,
    rankdata,
    wilcoxon,
)
from statsmodels.stats.multitest import (
    multipletests,
)

from link_prediction.config import (
    SUMMARY_RESULTS_DIR,
    load_methods_config,
)

NETWORK_METRIC_COLUMNS = (
    "benchmark",
    "network_id",
    "family",
    "method_id",
)

FOLD_RESULT_COLUMNS = (
    "benchmark",
    "network_id",
    "fold",
    "family",
    "method_id",
)


def load_benchmark_fold_metrics(
    benchmark_name: str = "revision",
    summary_results_dir: Path = (
        SUMMARY_RESULTS_DIR
    ),
    methods_config: (
        dict[str, Any] | None
    ) = None,
) -> pd.DataFrame:
    if methods_config is None:
        methods_config = (
            load_methods_config()
        )

    enabled_methods = {
        method_id:
            method_config
        for method_id, method_config
        in methods_config[
            "methods"
        ].items()
        if method_config.get(
            "enabled",
            True,
        )
    }

    family_ids = sorted(
        {
            method_config[
                "family"
            ]
            for method_config
            in enabled_methods.values()
        }
    )

    frames = []

    for family_id in family_ids:
        path = (
            summary_results_dir
            / (
                f"{benchmark_name}_"
                f"{family_id}_"
                "fold_metrics.csv"
            )
        )

        if not path.exists():
            raise FileNotFoundError(
                "Family fold metrics not found: "
                f"{path}"
            )

        frame = pd.read_csv(
            path
        )

        missing_columns = (
            set(
                FOLD_RESULT_COLUMNS
            )
            - set(frame.columns)
        )

        if missing_columns:
            raise ValueError(
                "Family fold metrics are "
                "missing columns: "
                f"{sorted(missing_columns)}"
            )

        observed_benchmarks = set(
            frame[
                "benchmark"
            ]
        )

        if observed_benchmarks != {
            benchmark_name
        }:
            raise ValueError(
                "Unexpected benchmark values "
                f"in {path.name}: "
                f"{sorted(observed_benchmarks)}"
            )

        frames.append(
            frame
        )

    fold_metrics = pd.concat(
        frames,
        ignore_index=True,
    )

    expected_method_ids = set(
        enabled_methods
    )

    observed_method_ids = set(
        fold_metrics[
            "method_id"
        ]
    )

    missing_method_ids = (
        expected_method_ids
        - observed_method_ids
    )

    unexpected_method_ids = (
        observed_method_ids
        - expected_method_ids
    )

    if (
        missing_method_ids
        or unexpected_method_ids
    ):
        raise ValueError(
            "Fold metrics do not match "
            "the enabled method registry. "
            f"Missing: {sorted(missing_method_ids)}. "
            "Unexpected: "
            f"{sorted(unexpected_method_ids)}."
        )

    expected_families = {
        method_id:
            method_config[
                "family"
            ]
        for method_id, method_config
        in enabled_methods.items()
    }

    invalid_families = (
        fold_metrics[
            fold_metrics.apply(
                lambda row: (
                    expected_families[
                        row[
                            "method_id"
                        ]
                    ]
                    != row[
                        "family"
                    ]
                ),
                axis=1,
            )
        ]
    )

    if not invalid_families.empty:
        raise ValueError(
            "Fold metrics contain methods "
            "assigned to incorrect families."
        )

    duplicated = (
        fold_metrics.duplicated(
            subset=list(
                FOLD_RESULT_COLUMNS
            ),
            keep=False,
        )
    )

    if duplicated.any():
        raise ValueError(
            "Fold metrics contain duplicated "
            "method-network-fold rows."
        )

    reference_method = min(
        expected_method_ids
    )

    reference_blocks = set(
        fold_metrics[
            fold_metrics[
                "method_id"
            ]
            == reference_method
        ][
            [
                "benchmark",
                "network_id",
                "fold",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    for method_id in sorted(
        expected_method_ids
    ):
        method_blocks = set(
            fold_metrics[
                fold_metrics[
                    "method_id"
                ]
                == method_id
            ][
                [
                    "benchmark",
                    "network_id",
                    "fold",
                ]
            ].itertuples(
                index=False,
                name=None,
            )
        )

        if (
            method_blocks
            != reference_blocks
        ):
            raise ValueError(
                "Fold metrics contain "
                "incomplete method blocks: "
                f"{method_id}"
            )

    return (
        fold_metrics
        .sort_values(
            list(
                FOLD_RESULT_COLUMNS
            )
        )
        .reset_index(
            drop=True
        )
    )


def select_confirmatory_methods(
    fold_metrics: pd.DataFrame,
    method_ids: list[str],
) -> pd.DataFrame:
    if len(
        set(method_ids)
    ) != len(method_ids):
        raise ValueError(
            "Confirmatory method identifiers "
            "must be unique."
        )

    if len(method_ids) < 3:
        raise ValueError(
            "Confirmatory analysis requires "
            "at least three methods."
        )

    available_method_ids = set(
        fold_metrics[
            "method_id"
        ]
    )

    missing_method_ids = (
        set(method_ids)
        - available_method_ids
    )

    if missing_method_ids:
        raise ValueError(
            "Confirmatory methods are missing "
            "from fold metrics: "
            f"{sorted(missing_method_ids)}"
        )

    selected = (
        fold_metrics[
            fold_metrics[
                "method_id"
            ].isin(
                method_ids
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    selected[
        "confirmatory_order"
    ] = (
        selected[
            "method_id"
        ].map(
            {
                method_id:
                    order
                for order, method_id
                in enumerate(
                    method_ids
                )
            }
        )
    )

    return (
        selected
        .sort_values(
            [
                "confirmatory_order",
                "benchmark",
                "network_id",
                "fold",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def aggregate_network_metrics(
    fold_metrics: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    required_columns = {
        *NETWORK_METRIC_COLUMNS,
        metric,
    }

    missing_columns = (
        required_columns
        - set(fold_metrics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Fold metrics are missing columns: "
            f"{sorted(missing_columns)}"
        )

    return (
        fold_metrics
        .groupby(
            list(
                NETWORK_METRIC_COLUMNS
            ),
            as_index=False,
        )[
            metric
        ]
        .mean()
    )


def build_method_metric_matrix(
    network_metrics: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    required_columns = {
        "benchmark",
        "network_id",
        "method_id",
        metric,
    }

    missing_columns = (
        required_columns
        - set(network_metrics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Network metrics are missing columns: "
            f"{sorted(missing_columns)}"
        )

    matrix = (
        network_metrics
        .pivot(
            index=[
                "benchmark",
                "network_id",
            ],
            columns="method_id",
            values=metric,
        )
        .sort_index(
            axis=0,
        )
        .sort_index(
            axis=1,
        )
    )

    missing_value_count = int(
        matrix.isna().sum().sum()
    )

    if missing_value_count > 0:
        raise ValueError(
            "Method matrix contains "
            f"{missing_value_count} missing "
            "network-method values."
        )

    if matrix.shape[0] < 2:
        raise ValueError(
            "Statistical comparison requires "
            "at least two complete networks."
        )

    if matrix.shape[1] < 3:
        raise ValueError(
            "Friedman test requires "
            "at least three methods."
        )

    return matrix


def mean_method_ranks(
    method_matrix: pd.DataFrame,
) -> pd.DataFrame:
    ranks = (
        method_matrix.rank(
            axis=1,
            ascending=False,
            method="average",
        )
    )

    return (
        ranks
        .mean(
            axis=0
        )
        .rename(
            "mean_rank"
        )
        .reset_index()
        .rename(
            columns={
                "method_id":
                    "method_id",
                "index":
                    "method_id",
            }
        )
        .sort_values(
            [
                "mean_rank",
                "method_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def friedman_method_test(
    method_matrix: pd.DataFrame,
) -> dict[str, float | int]:
    values = method_matrix.to_numpy(
        dtype=float
    )

    if np.all(
        np.ptp(
            values,
            axis=1,
        )
        == 0.0
    ):
        return {
            "network_count":
                int(
                    method_matrix.shape[0]
                ),
            "method_count":
                int(
                    method_matrix.shape[1]
                ),
            "statistic":
                0.0,
            "p_value":
                1.0,
        }

    samples = [
        method_matrix[
            method_id
        ].to_numpy(
            dtype=float
        )
        for method_id
        in method_matrix.columns
    ]

    result = friedmanchisquare(
        *samples
    )

    return {
        "network_count":
            int(
                method_matrix.shape[0]
            ),
        "method_count":
            int(
                method_matrix.shape[1]
            ),
        "statistic":
            float(
                result.statistic
            ),
        "p_value":
            float(
                result.pvalue
            ),
    }


def paired_rank_biserial(
    differences: np.ndarray,
) -> float:
    nonzero = differences[
        differences != 0.0
    ]

    if len(nonzero) == 0:
        return 0.0

    ranks = rankdata(
        np.abs(nonzero)
    )

    positive_rank_sum = float(
        ranks[
            nonzero > 0.0
        ].sum()
    )

    negative_rank_sum = float(
        ranks[
            nonzero < 0.0
        ].sum()
    )

    total_rank_sum = (
        positive_rank_sum
        + negative_rank_sum
    )

    return (
        positive_rank_sum
        - negative_rank_sum
    ) / total_rank_sum


def pairwise_wilcoxon_holm(
    method_matrix: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    if method_matrix.shape[0] < 2:
        raise ValueError(
            "Pairwise comparison requires "
            "at least two networks."
        )

    if method_matrix.shape[1] < 2:
        raise ValueError(
            "Pairwise comparison requires "
            "at least two methods."
        )

    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "alpha must be between zero and one."
        )

    rows = []

    for (
        first_method,
        second_method,
    ) in combinations(
        method_matrix.columns,
        2,
    ):
        first_scores = (
            method_matrix[
                first_method
            ].to_numpy(
                dtype=float
            )
        )

        second_scores = (
            method_matrix[
                second_method
            ].to_numpy(
                dtype=float
            )
        )

        differences = (
            first_scores
            - second_scores
        )

        if np.all(
            differences == 0.0
        ):
            statistic = 0.0
            p_value = 1.0
        else:
            result = wilcoxon(
                first_scores,
                second_scores,
                zero_method="pratt",
                alternative="two-sided",
                method="auto",
            )

            statistic = float(
                result.statistic
            )

            p_value = float(
                result.pvalue
            )

        rows.append(
            {
                "first_method":
                    first_method,
                "second_method":
                    second_method,
                "network_count":
                    len(differences),
                "statistic":
                    statistic,
                "p_value":
                    p_value,
                "median_difference":
                    float(
                        np.median(
                            differences
                        )
                    ),
                "rank_biserial":
                    float(
                        paired_rank_biserial(
                            differences
                        )
                    ),
                "first_wins":
                    int(
                        np.sum(
                            differences > 0.0
                        )
                    ),
                "ties":
                    int(
                        np.sum(
                            differences == 0.0
                        )
                    ),
                "second_wins":
                    int(
                        np.sum(
                            differences < 0.0
                        )
                    ),
            }
        )

    comparisons = pd.DataFrame(
        rows
    )

    if comparisons.empty:
        raise ValueError(
            "Pairwise comparison requires "
            "at least two methods."
        )

    rejected, adjusted, _, _ = (
        multipletests(
            comparisons[
                "p_value"
            ].to_numpy(),
            alpha=alpha,
            method="holm",
        )
    )

    comparisons[
        "adjusted_p_value"
    ] = adjusted

    comparisons[
        "reject_null"
    ] = rejected

    return comparisons


def write_statistical_csv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.with_suffix(
            path.suffix
            + ".part"
        )
    )

    dataframe.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        path
    )


def analyze_confirmatory_metric(
    fold_metrics: pd.DataFrame,
    method_ids: list[str],
    metric: str,
    alpha: float = 0.05,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    selected_fold_metrics = (
        select_confirmatory_methods(
            fold_metrics,
            method_ids,
        )
    )

    network_metrics = (
        aggregate_network_metrics(
            selected_fold_metrics,
            metric,
        )
    )

    method_matrix = (
        build_method_metric_matrix(
            network_metrics,
            metric,
        )
    )

    friedman_result = (
        friedman_method_test(
            method_matrix
        )
    )

    friedman = pd.DataFrame(
        [
            {
                "metric":
                    metric,
                "alpha":
                    alpha,
                **friedman_result,
                "reject_null":
                    friedman_result[
                        "p_value"
                    ]
                    < alpha,
            }
        ]
    )

    mean_ranks = (
        mean_method_ranks(
            method_matrix
        )
    )

    mean_ranks.insert(
        0,
        "metric",
        metric,
    )

    if bool(
        friedman.loc[
            0,
            "reject_null",
        ]
    ):
        pairwise = (
            pairwise_wilcoxon_holm(
                method_matrix,
                alpha=alpha,
            )
        )

        pairwise.insert(
            0,
            "metric",
            metric,
        )
    else:
        pairwise = pd.DataFrame(
            columns=[
                "metric",
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
            ]
        )

    return (
        network_metrics,
        friedman,
        mean_ranks,
        pairwise,
    )


def run_confirmatory_analysis(
    fold_metrics: pd.DataFrame,
    method_ids: list[str],
    metrics: tuple[str, ...] = (
        "average_precision",
        "roc_auc",
    ),
    alpha: float = 0.05,
    benchmark_name: str = "revision",
    output_dir: Path = (
        SUMMARY_RESULTS_DIR
    ),
) -> dict[str, pd.DataFrame]:
    if not metrics:
        raise ValueError(
            "At least one statistical "
            "metric is required."
        )

    if len(
        set(metrics)
    ) != len(metrics):
        raise ValueError(
            "Statistical metrics "
            "must be unique."
        )

    observed_benchmarks = set(
        fold_metrics[
            "benchmark"
        ]
    )

    if observed_benchmarks != {
        benchmark_name
    }:
        raise ValueError(
            "Fold metrics do not match "
            "the requested benchmark: "
            f"{benchmark_name}"
        )

    network_metric_frames = []
    friedman_frames = []
    rank_frames = []
    pairwise_frames = []

    for metric in metrics:
        (
            network_metrics,
            friedman,
            mean_ranks,
            pairwise,
        ) = analyze_confirmatory_metric(
            fold_metrics=
                fold_metrics,
            method_ids=
                method_ids,
            metric=
                metric,
            alpha=
                alpha,
        )

        network_metrics = (
            network_metrics
            .rename(
                columns={
                    metric:
                        "value",
                }
            )
        )

        network_metrics.insert(
            4,
            "metric",
            metric,
        )

        network_metric_frames.append(
            network_metrics
        )

        friedman_frames.append(
            friedman
        )

        rank_frames.append(
            mean_ranks
        )

        pairwise_frames.append(
            pairwise
        )

    outputs = {
        "network_metrics":
            pd.concat(
                network_metric_frames,
                ignore_index=True,
            ),
        "friedman":
            pd.concat(
                friedman_frames,
                ignore_index=True,
            ),
        "mean_ranks":
            pd.concat(
                rank_frames,
                ignore_index=True,
            ),
        "pairwise":
            pd.concat(
                pairwise_frames,
                ignore_index=True,
            ),
    }

    for output_name, dataframe in (
        outputs.items()
    ):
        write_statistical_csv(
            dataframe,
            output_dir
            / (
                f"{benchmark_name}_"
                "confirmatory_"
                f"{output_name}.csv"
            ),
        )

    return outputs
