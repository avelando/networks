from itertools import combinations

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

NETWORK_METRIC_COLUMNS = (
    "benchmark",
    "network_id",
    "family",
    "method_id",
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
        .dropna(
            axis=0,
            how="any",
        )
        .sort_index(
            axis=0,
        )
        .sort_index(
            axis=1,
        )
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