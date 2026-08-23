from time import perf_counter
from typing import Any

import pandas as pd
from threadpoolctl import (
    threadpool_limits,
)

from link_prediction.config import (
    SUMMARY_RESULTS_DIR,
    load_experiment_config,
    load_methods_config,
    load_networks_config,
)
from link_prediction.evaluation import load_fold_data
from link_prediction.execution import (
    run_process_tasks,
)
from link_prediction.methods.enhanced_local import (
    local_interacting_scores,
)
from link_prediction.methods.quasi_local_paths import (
    score_quasi_local_path_candidates,
)
from link_prediction.methods.quasi_local_walks import (
    score_quasi_local_walk_candidates,
)
from link_prediction.metrics import evaluate_ranking

METRIC_COLUMNS = (
    "average_precision",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "ndcg",
)


def sensitivity_values(
    method_config: dict[str, Any],
    parameter_name: str,
) -> list[int | float]:
    values = list(
        method_config
        .get("sensitivity", {})
        .get(parameter_name, [])
    )

    if not values:
        raise ValueError(
            "No sensitivity values configured "
            f"for parameter: {parameter_name}"
        )

    primary_value = (
        method_config["parameters"][
            parameter_name
        ]
    )

    if primary_value not in values:
        raise ValueError(
            f"Primary value {primary_value} "
            "is missing from sensitivity values "
            f"for parameter {parameter_name}."
        )

    return values


def sensitivity_method_name(
    method_id: str,
    parameter_name: str,
    parameter_value: float,
) -> str:
    if parameter_name == "beta":
        return (
            f"{method_id.upper()}-beta-"
            f"{float(parameter_value):g}"
        )

    if parameter_name == "steps":
        return (
            f"{method_id.upper()}-l"
            f"{int(parameter_value)}"
        )

    if parameter_name == "iterations":
        return (
            f"{method_id.upper()}-i"
            f"{int(parameter_value)}"
        )

    raise ValueError(
        "Unsupported sensitivity parameter: "
        f"{parameter_name}"
    )


def build_parameter_sensitivity_plan(
    methods_config: dict[str, Any],
) -> pd.DataFrame:
    methods = methods_config["methods"]

    rows = []

    lpi_config = methods["lpi"]

    for beta in sensitivity_values(
        lpi_config,
        "beta",
    ):
        rows.append(
            {
                "family":
                    lpi_config["family"],
                "method_id":
                    "lpi",
                "method":
                    sensitivity_method_name(
                        method_id="lpi",
                        parameter_name="beta",
                        parameter_value=beta,
                    ),
                "parameter":
                    "beta",
                "parameter_value":
                    beta,
                "is_primary":
                    beta
                    == lpi_config[
                        "parameters"
                    ][
                        "beta"
                    ],
            }
        )

        lit_config = methods["lit"]

    for iterations in sensitivity_values(
        lit_config,
        "iterations",
    ):
        rows.append(
            {
                "family":
                    lit_config["family"],
                "method_id":
                    "lit",
                "method":
                    sensitivity_method_name(
                        method_id="lit",
                        parameter_name=
                            "iterations",
                        parameter_value=
                            iterations,
                    ),
                "parameter":
                    "iterations",
                "parameter_value":
                    iterations,
                "is_primary":
                    iterations
                    == lit_config[
                        "parameters"
                    ][
                        "iterations"
                    ],
            }
        )

    for method_id in (
        "lrw",
        "srw",
        "pfp",
    ):
        method_config = methods[
            method_id
        ]

        for steps in sensitivity_values(
            method_config,
            "steps",
        ):
            rows.append(
                {
                    "family":
                        method_config[
                            "family"
                        ],
                    "method_id":
                        method_id,
                    "method":
                        sensitivity_method_name(
                            method_id=method_id,
                            parameter_name="steps",
                            parameter_value=steps,
                        ),
                    "parameter":
                        "steps",
                    "parameter_value":
                        steps,
                    "is_primary":
                        steps
                        == method_config[
                            "parameters"
                        ][
                            "steps"
                        ],
                }
            )

    return pd.DataFrame(rows)


def evaluate_parameter_scores(
    candidates: pd.DataFrame,
    scores,
) -> dict[str, float | int]:
    return evaluate_ranking(
        labels=candidates["label"],
        scores=scores,
        candidate_ids=
            candidates["candidate_id"],
        cutoff=int(
            candidates["label"].sum()
        ),
    )


def summarize_parameter_sensitivity(
    fold_metrics: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    group_columns = [
        "benchmark",
        "network_id",
        "network",
        "domain",
        "family",
        "method_id",
        "method",
        "parameter",
        "parameter_value",
        "is_primary",
    ]

    network_summary = (
        fold_metrics
        .groupby(
            group_columns,
            sort=False,
        )[
            list(METRIC_COLUMNS)
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    network_summary.columns = [
        "_".join(
            part
            for part in column
            if part
        )
        if isinstance(
            column,
            tuple,
        )
        else column
        for column
        in network_summary.columns
    ]

    overall_group_columns = [
        "benchmark",
        "family",
        "method_id",
        "method",
        "parameter",
        "parameter_value",
        "is_primary",
    ]

    overall_rows = []

    for values, group in (
        network_summary.groupby(
            overall_group_columns,
            sort=False,
        )
    ):
        row = dict(
            zip(
                overall_group_columns,
                values,
                strict=True,
            )
        )

        row["network_count"] = len(
            group
        )

        for metric in METRIC_COLUMNS:
            network_values = group[
                f"{metric}_mean"
            ]

            row[
                f"{metric}_mean"
            ] = float(
                network_values.mean()
            )

            row[
                f"{metric}_sd_across_networks"
            ] = float(
                network_values.std()
            )

        overall_rows.append(row)

    overall_summary = pd.DataFrame(
        overall_rows
    )

    return (
        network_summary,
        overall_summary,
    )


def evaluate_parameter_sensitivity_fold(
    benchmark_name: str,
    network_id: str,
    fold_number: int,
) -> pd.DataFrame:
    networks_config = load_networks_config()
    methods_config = load_methods_config()

    network_config = networks_config[
        "networks"
    ][
        network_id
    ]

    methods = methods_config[
        "methods"
    ]

    lpi_config = methods["lpi"]

    beta_values = sensitivity_values(
        lpi_config,
        "beta",
    )

    lpi_length = int(
        lpi_config[
            "parameters"
        ][
            "length"
        ]
    )

    lit_config = methods["lit"]

    lit_iteration_values = (
        sensitivity_values(
            lit_config,
            "iterations",
        )
    )

    walk_method_ids = (
        "lrw",
        "srw",
        "pfp",
    )

    walk_step_sets = {
        tuple(
            sensitivity_values(
                methods[method_id],
                "steps",
            )
        )
        for method_id
        in walk_method_ids
    }

    if len(walk_step_sets) != 1:
        raise ValueError(
            "LRW, SRW, and PFP must use "
            "the same sensitivity step values."
        )

    step_values = list(
        walk_step_sets.pop()
    )

    graph, candidates = load_fold_data(
        benchmark_name=benchmark_name,
        network_id=network_id,
        fold_number=fold_number,
    )

    rows = []

    with threadpool_limits(
        limits=1
    ):
        for beta in beta_values:
            start = perf_counter()

            score_table = (
                score_quasi_local_path_candidates(
                    graph=graph,
                    candidates=candidates,
                    beta=float(beta),
                    length=lpi_length,
                )
            )

            scoring_seconds = (
                perf_counter()
                - start
            )

            metrics = evaluate_parameter_scores(
                candidates,
                score_table["lpi"],
            )

            rows.append(
                {
                    "benchmark":
                        benchmark_name,
                    "network_id":
                        network_id,
                    "network":
                        network_config["name"],
                    "domain":
                        network_config["domain"],
                    "fold":
                        fold_number,
                    "family":
                        lpi_config["family"],
                    "method_id":
                        "lpi",
                    "method":
                        sensitivity_method_name(
                            method_id="lpi",
                            parameter_name="beta",
                            parameter_value=beta,
                        ),
                    "parameter":
                        "beta",
                    "parameter_value":
                        beta,
                    "is_primary":
                        beta
                        == lpi_config[
                            "parameters"
                        ][
                            "beta"
                        ],
                    "configuration_scoring_seconds":
                        scoring_seconds,
                    **metrics,
                }
            )
        for iterations in (
            lit_iteration_values
        ):
            start = perf_counter()

            scores = (
                local_interacting_scores(
                    graph=graph,
                    candidates=candidates,
                    iterations=int(
                        iterations
                    ),
                )
            )

            scoring_seconds = (
                perf_counter()
                - start
            )

            metrics = (
                evaluate_parameter_scores(
                    candidates,
                    scores,
                )
            )

            rows.append(
                {
                    "benchmark":
                        benchmark_name,
                    "network_id":
                        network_id,
                    "network":
                        network_config[
                            "name"
                        ],
                    "domain":
                        network_config[
                            "domain"
                        ],
                    "fold":
                        fold_number,
                    "family":
                        lit_config[
                            "family"
                        ],
                    "method_id":
                        "lit",
                    "method":
                        sensitivity_method_name(
                            method_id="lit",
                            parameter_name=
                                "iterations",
                            parameter_value=
                                iterations,
                        ),
                    "parameter":
                        "iterations",
                    "parameter_value":
                        iterations,
                    "is_primary":
                        iterations
                        == lit_config[
                            "parameters"
                        ][
                            "iterations"
                        ],
                    "configuration_scoring_seconds":
                        scoring_seconds,
                    **metrics,
                }
            )

        for steps in step_values:
            start = perf_counter()

            score_table = (
                score_quasi_local_walk_candidates(
                    graph=graph,
                    candidates=candidates,
                    steps=int(steps),
                )
            )

            scoring_seconds = (
                perf_counter()
                - start
            )

            for method_id in walk_method_ids:
                method_config = methods[
                    method_id
                ]

                metrics = (
                    evaluate_parameter_scores(
                        candidates,
                        score_table[
                            method_id
                        ],
                    )
                )

                rows.append(
                    {
                        "benchmark":
                            benchmark_name,
                        "network_id":
                            network_id,
                        "network":
                            network_config[
                                "name"
                            ],
                        "domain":
                            network_config[
                                "domain"
                            ],
                        "fold":
                            fold_number,
                        "family":
                            method_config[
                                "family"
                            ],
                        "method_id":
                            method_id,
                        "method":
                            sensitivity_method_name(
                                method_id=
                                    method_id,
                                parameter_name=
                                    "steps",
                                parameter_value=
                                    steps,
                            ),
                        "parameter":
                            "steps",
                        "parameter_value":
                            steps,
                        "is_primary":
                            steps
                            == method_config[
                                "parameters"
                            ][
                                "steps"
                            ],
                        "configuration_scoring_seconds":
                            scoring_seconds,
                        **metrics,
                    }
                )

    return pd.DataFrame(
        rows
    )


def run_parameter_sensitivity(
    benchmark_name: str = "revision",
    max_workers: int | str | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    experiment_config = (
        load_experiment_config()
    )

    networks_config = (
        load_networks_config()
    )

    n_folds = int(
        experiment_config[
            "experiment"
        ][
            "n_folds"
        ]
    )

    benchmark = networks_config[
        "benchmarks"
    ][
        benchmark_name
    ]

    network_definitions = (
        networks_config[
            "networks"
        ]
    )

    tasks = []

    for network_id in benchmark:
        if not network_definitions[
            network_id
        ].get(
            "enabled",
            True,
        ):
            continue

        for fold_number in range(
            1,
            n_folds + 1,
        ):
            tasks.append(
                (
                    benchmark_name,
                    network_id,
                    fold_number,
                )
            )

    rows = run_process_tasks(
        evaluate_parameter_sensitivity_fold,
        tasks,
        max_workers=max_workers,
        profile="sensitivity",
        label="parameter sensitivity",
    )

    fold_metrics = pd.concat(
        rows,
        ignore_index=True,
    )

    (
        network_summary,
        overall_summary,
    ) = summarize_parameter_sensitivity(
        fold_metrics
    )

    SUMMARY_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        SUMMARY_RESULTS_DIR
        / (
            f"{benchmark_name}_"
            "parameter_sensitivity_"
            "fold_metrics.csv"
        ),
        index=False,
    )

    network_summary.to_csv(
        SUMMARY_RESULTS_DIR
        / (
            f"{benchmark_name}_"
            "parameter_sensitivity_"
            "network_summary.csv"
        ),
        index=False,
    )

    overall_summary.to_csv(
        SUMMARY_RESULTS_DIR
        / (
            f"{benchmark_name}_"
            "parameter_sensitivity_"
            "overall_summary.csv"
        ),
        index=False,
    )

    return (
        fold_metrics,
        network_summary,
        overall_summary,
    )