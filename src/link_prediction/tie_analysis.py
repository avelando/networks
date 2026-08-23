from pathlib import Path

import pandas as pd

from link_prediction.config import (
    SUMMARY_RESULTS_DIR,
)
from link_prediction.statistical_analysis import (
    write_statistical_csv,
)

TIE_REQUIRED_COLUMNS = {
    "benchmark",
    "network_id",
    "network",
    "domain",
    "fold",
    "family",
    "analysis_family",
    "method_id",
    "method",
    "distinct_score_count",
    "tie_group_count",
    "tied_candidate_count",
    "tied_candidate_ratio",
    "largest_tie_group",
    "cutoff_tie_size",
    "cutoff_tie_crosses_boundary",
}


def aggregate_network_tie_diagnostics(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    missing_columns = (
        TIE_REQUIRED_COLUMNS
        - set(fold_metrics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Fold metrics are missing tie "
            "diagnostic columns: "
            f"{sorted(missing_columns)}"
        )

    if fold_metrics.empty:
        raise ValueError(
            "Fold metrics cannot be empty."
        )

    return (
        fold_metrics
        .groupby(
            [
                "benchmark",
                "network_id",
                "network",
                "domain",
                "family",
                "analysis_family",
                "method_id",
                "method",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            fold_count=(
                "fold",
                "nunique",
            ),
            mean_distinct_score_count=(
                "distinct_score_count",
                "mean",
            ),
            mean_tie_group_count=(
                "tie_group_count",
                "mean",
            ),
            mean_tied_candidate_count=(
                "tied_candidate_count",
                "mean",
            ),
            mean_tied_candidate_ratio=(
                "tied_candidate_ratio",
                "mean",
            ),
            mean_largest_tie_group=(
                "largest_tie_group",
                "mean",
            ),
            mean_cutoff_tie_size=(
                "cutoff_tie_size",
                "mean",
            ),
            cutoff_boundary_tie_count=(
                "cutoff_tie_crosses_boundary",
                "sum",
            ),
            cutoff_boundary_tie_rate=(
                "cutoff_tie_crosses_boundary",
                "mean",
            ),
        )
    )


def aggregate_method_tie_diagnostics(
    network_ties: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "benchmark",
        "network_id",
        "family",
        "analysis_family",
        "method_id",
        "method",
        "mean_distinct_score_count",
        "mean_tie_group_count",
        "mean_tied_candidate_count",
        "mean_tied_candidate_ratio",
        "mean_largest_tie_group",
        "mean_cutoff_tie_size",
        "cutoff_boundary_tie_rate",
    }

    missing_columns = (
        required_columns
        - set(network_ties.columns)
    )

    if missing_columns:
        raise ValueError(
            "Network tie diagnostics are "
            "missing columns: "
            f"{sorted(missing_columns)}"
        )

    if network_ties.empty:
        raise ValueError(
            "Network tie diagnostics "
            "cannot be empty."
        )

    return (
        network_ties
        .groupby(
            [
                "benchmark",
                "family",
                "analysis_family",
                "method_id",
                "method",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            network_count=(
                "network_id",
                "nunique",
            ),
            mean_distinct_score_count=(
                "mean_distinct_score_count",
                "mean",
            ),
            mean_tie_group_count=(
                "mean_tie_group_count",
                "mean",
            ),
            mean_tied_candidate_count=(
                "mean_tied_candidate_count",
                "mean",
            ),
            mean_tied_candidate_ratio=(
                "mean_tied_candidate_ratio",
                "mean",
            ),
            mean_largest_tie_group=(
                "mean_largest_tie_group",
                "mean",
            ),
            mean_cutoff_tie_size=(
                "mean_cutoff_tie_size",
                "mean",
            ),
            mean_cutoff_boundary_tie_rate=(
                "cutoff_boundary_tie_rate",
                "mean",
            ),
        )
        .sort_values(
            [
                "mean_cutoff_boundary_tie_rate",
                "mean_tied_candidate_ratio",
                "method_id",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def run_tie_analysis(
    fold_metrics: pd.DataFrame,
    benchmark_name: str = "standard",
    output_dir: Path = (
        SUMMARY_RESULTS_DIR
    ),
) -> dict[str, pd.DataFrame]:
    observed_benchmarks = set(
        fold_metrics[
            "benchmark"
        ]
    )

    if observed_benchmarks != {
        benchmark_name
    }:
        raise ValueError(
            "Unexpected benchmark values: "
            f"{sorted(observed_benchmarks)}"
        )

    network_ties = (
        aggregate_network_tie_diagnostics(
            fold_metrics
        )
    )

    method_ties = (
        aggregate_method_tie_diagnostics(
            network_ties
        )
    )

    outputs = {
        "network_ties":
            network_ties,
        "method_ties":
            method_ties,
    }

    filenames = {
        "network_ties":
            (
                f"{benchmark_name}_"
                "network_tie_diagnostics.csv"
            ),
        "method_ties":
            (
                f"{benchmark_name}_"
                "method_tie_diagnostics.csv"
            ),
    }

    for output_name, dataframe in (
        outputs.items()
    ):
        write_statistical_csv(
            dataframe,
            output_dir
            / filenames[
                output_name
            ],
        )

    return outputs