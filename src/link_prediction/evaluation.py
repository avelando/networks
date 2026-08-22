from collections.abc import (
    Callable,
    Sequence,
)
from pathlib import Path
from time import perf_counter

import networkx as nx
import pandas as pd

from link_prediction.config import (
    FOLDS_DATA_DIR,
    RESULTS_DIR,
    SUMMARY_RESULTS_DIR,
    load_experiment_config,
    load_methods_config,
    load_networks_config,
)
from link_prediction.metrics import (
    evaluate_ranking,
)

Scorer = Callable[
    [
        nx.Graph,
        pd.DataFrame,
    ],
    pd.DataFrame,
]


def load_candidate_table(
    path: Path,
) -> pd.DataFrame:
    dataframe = pd.read_csv(
        path,
        dtype={
            "source": str,
            "target": str,
        },
    )

    required_columns = {
        "candidate_id",
        "source",
        "target",
        "label",
    }

    missing_columns = (
        required_columns
        - set(
            dataframe.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Candidate table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe[
        "candidate_id"
    ] = dataframe[
        "candidate_id"
    ].astype(int)

    dataframe[
        "label"
    ] = dataframe[
        "label"
    ].astype(int)

    return dataframe


def load_training_graph(
    path: Path,
) -> nx.Graph:
    graph = nx.read_edgelist(
        path,
        nodetype=str,
        create_using=nx.Graph(),
        data=False,
    )

    if (
        graph.number_of_nodes()
        == 0
    ):
        raise ValueError(
            f"Training graph is empty: {path}"
        )

    if not nx.is_connected(
        graph
    ):
        raise ValueError(
            "Training graph is disconnected: "
            f"{path}"
        )

    return graph


def load_fold_data(
    benchmark_name: str,
    network_id: str,
    fold_number: int,
) -> tuple[
    nx.Graph,
    pd.DataFrame,
]:
    fold_directory = (
        FOLDS_DATA_DIR
        / benchmark_name
        / network_id
        / f"fold_{fold_number:02d}"
    )

    graph = (
        load_training_graph(
            fold_directory
            / "train.edgelist"
        )
    )

    candidates = (
        load_candidate_table(
            fold_directory
            / "candidates.csv"
        )
    )

    candidate_nodes = (
        set(
            candidates["source"]
        )
        | set(
            candidates["target"]
        )
    )

    missing_nodes = (
        candidate_nodes
        - set(
            graph.nodes()
        )
    )

    if missing_nodes:
        raise ValueError(
            "Candidate table contains "
            f"{len(missing_nodes)} nodes "
            "absent from the training graph."
        )

    return (
        graph,
        candidates,
    )


def evaluate_score_table(
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
    method_ids: Sequence[str],
) -> pd.DataFrame:
    if len(candidates) != len(
        scores
    ):
        raise ValueError(
            "Candidate and score tables "
            "must have the same number of rows."
        )

    positive_count = int(
        candidates[
            "label"
        ].sum()
    )

    rows = []

    for method_id in method_ids:
        if (
            method_id
            not in scores.columns
        ):
            raise ValueError(
                "Missing score column "
                f"for method: {method_id}"
            )

        metrics = (
            evaluate_ranking(
                labels=
                    candidates[
                        "label"
                    ],
                scores=
                    scores[
                        method_id
                    ],
                candidate_ids=
                    candidates[
                        "candidate_id"
                    ],
                cutoff=
                    positive_count,
            )
        )

        rows.append(
            {
                "method_id":
                    method_id,
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def summarize_fold_metrics(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "ndcg",
    ]

    grouped = (
        fold_metrics.groupby(
            [
                "benchmark",
                "network_id",
                "network",
                "domain",
                "family",
                "method_id",
                "method",
            ],
            sort=False,
        )
    )

    summary = (
        grouped[
            metric_columns
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    summary.columns = [
        "_".join(
            part
            for part
            in column
            if part
        )
        if isinstance(
            column,
            tuple,
        )
        else column
        for column
        in summary.columns
    ]

    return summary


def run_method_family_benchmark(
    family_id: str,
    scorer: Scorer,
    benchmark_name: str = "revision",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    experiment_config = (
        load_experiment_config()
    )

    networks_config = (
        load_networks_config()
    )

    methods_config = (
        load_methods_config()
    )

    n_folds = int(
        experiment_config[
            "experiment"
        ][
            "n_folds"
        ]
    )

    benchmark = (
        networks_config[
            "benchmarks"
        ][
            benchmark_name
        ]
    )

    network_definitions = (
        networks_config[
            "networks"
        ]
    )

    method_ids = [
        method_id
        for (
            method_id,
            method_config,
        ) in methods_config[
            "methods"
        ].items()
        if (
            method_config[
                "family"
            ]
            == family_id
            and method_config.get(
                "enabled",
                True,
            )
        )
    ]

    if not method_ids:
        raise ValueError(
            "No enabled methods configured "
            f"for family: {family_id}"
        )

    evaluation_directory = (
        RESULTS_DIR
        / "evaluations"
        / benchmark_name
        / family_id
    )

    evaluation_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for network_id in benchmark:
        network_config = (
            network_definitions[
                network_id
            ]
        )

        if not network_config.get(
            "enabled",
            True,
        ):
            continue

        for fold_number in range(
            1,
            n_folds + 1,
        ):
            (
                graph,
                candidates,
            ) = load_fold_data(
                benchmark_name=
                    benchmark_name,
                network_id=
                    network_id,
                fold_number=
                    fold_number,
            )

            start = perf_counter()

            scores = scorer(
                graph,
                candidates,
            )

            family_scoring_seconds = (
                perf_counter()
                - start
            )

            metrics = (
                evaluate_score_table(
                    candidates=
                        candidates,
                    scores=
                        scores,
                    method_ids=
                        method_ids,
                )
            )

            metrics.insert(
                0,
                "fold",
                fold_number,
            )

            metrics.insert(
                0,
                "domain",
                network_config[
                    "domain"
                ],
            )

            metrics.insert(
                0,
                "network",
                network_config[
                    "name"
                ],
            )

            metrics.insert(
                0,
                "network_id",
                network_id,
            )

            metrics.insert(
                0,
                "benchmark",
                benchmark_name,
            )

            metrics.insert(
                5,
                "family",
                family_id,
            )

            metrics[
                "method"
            ] = metrics[
                "method_id"
            ].map(
                {
                    method_id:
                        methods_config[
                            "methods"
                        ][
                            method_id
                        ][
                            "name"
                        ]
                    for method_id
                    in method_ids
                }
            )

            metrics[
                "family_scoring_seconds"
            ] = (
                family_scoring_seconds
            )

            rows.append(
                metrics
            )

            scored_candidates = (
                candidates.copy()
            )

            for method_id in method_ids:
                scored_candidates[
                    method_id
                ] = scores[
                    method_id
                ].to_numpy()

            network_directory = (
                evaluation_directory
                / network_id
            )

            network_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            scored_candidates.to_csv(
                network_directory
                / (
                    f"fold_"
                    f"{fold_number:02d}"
                    "_scores.csv"
                ),
                index=False,
            )

    fold_metrics = pd.concat(
        rows,
        ignore_index=True,
    )

    network_summary = (
        summarize_fold_metrics(
            fold_metrics
        )
    )

    SUMMARY_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.to_csv(
        SUMMARY_RESULTS_DIR
        / (
            f"{benchmark_name}_"
            f"{family_id}_"
            "fold_metrics.csv"
        ),
        index=False,
    )

    network_summary.to_csv(
        SUMMARY_RESULTS_DIR
        / (
            f"{benchmark_name}_"
            f"{family_id}_"
            "network_summary.csv"
        ),
        index=False,
    )

    return (
        fold_metrics,
        network_summary,
    )