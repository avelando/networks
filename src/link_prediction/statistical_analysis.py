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


def friedman_method_test(
    method_matrix: pd.DataFrame,
) -> dict[str, float | int]:
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
