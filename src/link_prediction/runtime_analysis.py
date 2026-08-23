from pathlib import Path

import pandas as pd

from link_prediction.config import (
    SUMMARY_RESULTS_DIR,
)
from link_prediction.statistical_analysis import (
    write_statistical_csv,
)

RUNTIME_REQUIRED_COLUMNS = {
    "benchmark",
    "network_id",
    "network",
    "domain",
    "fold",
    "family",
    "analysis_family",
    "method_id",
    "candidate_count",
    "family_scoring_seconds",
}


def extract_family_fold_runtime(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    missing_columns = (
        RUNTIME_REQUIRED_COLUMNS
        - set(fold_metrics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Fold metrics are missing runtime "
            "columns: "
            f"{sorted(missing_columns)}"
        )

    if fold_metrics.empty:
        raise ValueError(
            "Fold metrics cannot be empty."
        )

    if (
        fold_metrics[
            "family_scoring_seconds"
        ] <= 0.0
    ).any():
        raise ValueError(
            "Family scoring time must "
            "be positive."
        )

    if (
        fold_metrics[
            "candidate_count"
        ] < 1
    ).any():
        raise ValueError(
            "Candidate count must "
            "be positive."
        )

    block_columns = [
        "benchmark",
        "network_id",
        "fold",
        "family",
    ]

    consistency = (
        fold_metrics
        .groupby(
            block_columns,
            sort=False,
        )[
            [
                "family_scoring_seconds",
                "candidate_count",
            ]
        ]
        .nunique(
            dropna=False
        )
    )

    if (
        consistency > 1
    ).any().any():
        raise ValueError(
            "Runtime values are inconsistent "
            "within a family-network-fold block."
        )

    method_counts = (
        fold_metrics
        .groupby(
            block_columns,
            sort=False,
        )[
            "method_id"
        ]
        .nunique()
        .rename(
            "method_count"
        )
        .reset_index()
    )

    family_folds = (
        fold_metrics[
            [
                "benchmark",
                "network_id",
                "network",
                "domain",
                "fold",
                "family",
                "analysis_family",
                "candidate_count",
                "family_scoring_seconds",
            ]
        ]
        .drop_duplicates(
            subset=block_columns
        )
        .merge(
            method_counts,
            on=block_columns,
            how="left",
            validate="one_to_one",
        )
        .reset_index(
            drop=True
        )
    )

    family_folds[
        "candidates_per_second"
    ] = (
        family_folds[
            "candidate_count"
        ]
        / family_folds[
            "family_scoring_seconds"
        ]
    )

    return family_folds


def aggregate_network_runtime(
    family_folds: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "benchmark",
        "network_id",
        "network",
        "domain",
        "fold",
        "family",
        "analysis_family",
        "method_count",
        "candidate_count",
        "family_scoring_seconds",
        "candidates_per_second",
    }

    missing_columns = (
        required_columns
        - set(family_folds.columns)
    )

    if missing_columns:
        raise ValueError(
            "Family fold runtime is missing "
            "columns: "
            f"{sorted(missing_columns)}"
        )

    return (
        family_folds
        .groupby(
            [
                "benchmark",
                "network_id",
                "network",
                "domain",
                "family",
                "analysis_family",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            fold_count=(
                "fold",
                "nunique",
            ),
            method_count=(
                "method_count",
                "max",
            ),
            mean_candidate_count=(
                "candidate_count",
                "mean",
            ),
            mean_scoring_seconds=(
                "family_scoring_seconds",
                "mean",
            ),
            std_scoring_seconds=(
                "family_scoring_seconds",
                "std",
            ),
            mean_candidates_per_second=(
                "candidates_per_second",
                "mean",
            ),
        )
    )


def aggregate_family_runtime(
    network_runtime: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "benchmark",
        "network_id",
        "family",
        "analysis_family",
        "method_count",
        "mean_candidate_count",
        "mean_scoring_seconds",
        "mean_candidates_per_second",
    }

    missing_columns = (
        required_columns
        - set(network_runtime.columns)
    )

    if missing_columns:
        raise ValueError(
            "Network runtime is missing "
            "columns: "
            f"{sorted(missing_columns)}"
        )

    return (
        network_runtime
        .groupby(
            [
                "benchmark",
                "family",
                "analysis_family",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            network_count=(
                "network_id",
                "nunique",
            ),
            method_count=(
                "method_count",
                "max",
            ),
            mean_candidate_count=(
                "mean_candidate_count",
                "mean",
            ),
            mean_scoring_seconds=(
                "mean_scoring_seconds",
                "mean",
            ),
            median_scoring_seconds=(
                "mean_scoring_seconds",
                "median",
            ),
            mean_candidates_per_second=(
                "mean_candidates_per_second",
                "mean",
            ),
        )
        .sort_values(
            [
                "mean_scoring_seconds",
                "family",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def run_runtime_analysis(
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

    family_folds = (
        extract_family_fold_runtime(
            fold_metrics
        )
    )

    network_runtime = (
        aggregate_network_runtime(
            family_folds
        )
    )

    family_runtime = (
        aggregate_family_runtime(
            network_runtime
        )
    )

    outputs = {
        "family_folds":
            family_folds,
        "network_runtime":
            network_runtime,
        "family_runtime":
            family_runtime,
    }

    filenames = {
        "family_folds":
            (
                f"{benchmark_name}_"
                "family_fold_runtime.csv"
            ),
        "network_runtime":
            (
                f"{benchmark_name}_"
                "family_runtime_by_network.csv"
            ),
        "family_runtime":
            (
                f"{benchmark_name}_"
                "family_runtime_summary.csv"
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