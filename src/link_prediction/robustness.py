import random
from collections.abc import Iterable
from typing import Any

import networkx as nx
import pandas as pd

from link_prediction.config import (
    RESULTS_DIR,
    SUMMARY_RESULTS_DIR,
    load_experiment_config,
    load_methods_config,
    load_networks_config,
)
from link_prediction.evaluation import (
    evaluate_score_table,
    load_fold_data,
)
from link_prediction.methods.degree_based import (
    score_degree_based_candidates,
)
from link_prediction.methods.enhanced_local import (
    score_enhanced_local_candidates,
)
from link_prediction.methods.local_bayesian import (
    score_local_bayesian_candidates,
)
from link_prediction.methods.local_similarity import (
    score_local_similarity_candidates,
)
from link_prediction.methods.quasi_local_paths import (
    score_quasi_local_path_candidates,
)
from link_prediction.methods.quasi_local_walks import (
    score_quasi_local_walk_candidates,
)
from link_prediction.preprocessing import (
    load_processed_graph,
)
from link_prediction.sampling import (
    canonical_edge,
    derive_seed,
    sample_non_edges,
)

METRIC_COLUMNS = (
    "average_precision",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "ndcg",
)


def normalize_negative_ratios(
    ratios: Iterable[int],
    primary_ratio: int,
) -> list[int]:
    normalized = sorted(
        {
            int(ratio)
            for ratio in ratios
        }
    )

    if not normalized:
        raise ValueError(
            "At least one negative sampling ratio is required."
        )

    if any(
        ratio < 1
        for ratio in normalized
    ):
        raise ValueError(
            "Negative sampling ratios must be at least 1."
        )

    if primary_ratio not in normalized:
        raise ValueError(
            "Primary ratio must be included in robustness ratios."
        )

    return normalized


def candidate_edges(
    candidates: pd.DataFrame,
    label: int,
) -> list[tuple[object, object]]:
    if label not in {
        0,
        1,
    }:
        raise ValueError(
            "label must be 0 or 1."
        )

    selected = candidates[
        candidates["label"] == label
    ]

    return [
        canonical_edge(
            source,
            target,
        )
        for source, target
        in selected[
            [
                "source",
                "target",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    ]


def build_nested_robustness_candidates(
    full_graph: nx.Graph,
    primary_candidates: pd.DataFrame,
    ratios: list[int],
    primary_ratio: int,
    random_seed: int,
    network_id: str,
    fold_number: int,
) -> dict[int, pd.DataFrame]:
    ratios = normalize_negative_ratios(
        ratios=ratios,
        primary_ratio=primary_ratio,
    )

    positive_edges = candidate_edges(
        primary_candidates,
        label=1,
    )

    base_negatives = candidate_edges(
        primary_candidates,
        label=0,
    )

    if not positive_edges:
        raise ValueError(
            "Primary candidates must contain positive edges."
        )

    if (
        primary_candidates[
            "candidate_id"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Primary candidate_id values must be unique."
        )

    expected_base_negatives = (
        len(positive_edges)
        * primary_ratio
    )

    if (
        len(base_negatives)
        != expected_base_negatives
    ):
        raise ValueError(
            "Primary candidates do not match "
            "the configured negative ratio."
        )

    if (
        len(
            set(
                positive_edges
            )
        )
        != len(
            positive_edges
        )
    ):
        raise ValueError(
            "Positive candidate pairs must be unique."
        )

    if (
        len(
            set(
                base_negatives
            )
        )
        != len(
            base_negatives
        )
    ):
        raise ValueError(
            "Negative candidate pairs must be unique."
        )

    maximum_ratio = max(
        ratios
    )

    maximum_negative_count = (
        len(positive_edges)
        * maximum_ratio
    )

    additional_count = (
        maximum_negative_count
        - len(base_negatives)
    )

    additional_negatives = (
        sample_non_edges(
            graph=full_graph,
            n_samples=
                additional_count,
            seed=derive_seed(
                random_seed,
                network_id,
                "robustness_negative_sample",
                fold_number,
                maximum_ratio,
            ),
            exclude=
                base_negatives,
        )
    )

    nested_rng = random.Random(
        derive_seed(
            random_seed,
            network_id,
            "robustness_negative_order",
            fold_number,
            maximum_ratio,
        )
    )

    nested_rng.shuffle(
        additional_negatives
    )

    base_negative_ranks = {
        edge: rank
        for rank, edge
        in enumerate(
            base_negatives
        )
    }

    ordered_primary = (
        primary_candidates
        .sort_values(
            "candidate_id",
            kind="mergesort",
        )
    )

    records = []

    for (
        candidate_id,
        source,
        target,
        label,
    ) in ordered_primary[
        [
            "candidate_id",
            "source",
            "target",
            "label",
        ]
    ].itertuples(
        index=False,
        name=None,
    ):
        edge = canonical_edge(
            source,
            target,
        )

        records.append(
            {
                "source":
                    edge[0],
                "target":
                    edge[1],
                "label":
                    int(label),
                "_negative_rank":
                    (
                        base_negative_ranks[
                            edge
                        ]
                        if int(label) == 0
                        else -1
                    ),
                "_order_key":
                    float(
                        candidate_id
                    ),
                "_order_tie":
                    0,
            }
        )

    order_rng = random.Random(
        derive_seed(
            random_seed,
            network_id,
            "robustness_candidate_order",
            fold_number,
            maximum_ratio,
        )
    )

    if (
        len(
            ordered_primary
        )
        == 1
    ):
        lower_bound = -0.5
        upper_bound = 0.5

    else:
        lower_bound = (
            float(
                ordered_primary[
                    "candidate_id"
                ].min()
            )
            - 0.5
        )

        upper_bound = (
            float(
                ordered_primary[
                    "candidate_id"
                ].max()
            )
            + 0.5
        )

    for (
        negative_rank,
        (
            source,
            target,
        ),
    ) in enumerate(
        additional_negatives,
        start=len(
            base_negatives
        ),
    ):
        records.append(
            {
                "source":
                    source,
                "target":
                    target,
                "label":
                    0,
                "_negative_rank":
                    negative_rank,
                "_order_key":
                    order_rng.uniform(
                        lower_bound,
                        upper_bound,
                    ),
                "_order_tie":
                    negative_rank + 1,
            }
        )

    maximum_table = (
        pd.DataFrame(
            records
        )
        .sort_values(
            [
                "_order_key",
                "_order_tie",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    maximum_table.insert(
        0,
        "candidate_id",
        range(
            len(
                maximum_table
            )
        ),
    )

    tables: dict[
        int,
        pd.DataFrame,
    ] = {}

    for ratio in ratios:
        desired_negative_count = (
            len(
                positive_edges
            )
            * ratio
        )

        mask = (
            maximum_table[
                "label"
            ].eq(
                1
            )
            |
            maximum_table[
                "_negative_rank"
            ].lt(
                desired_negative_count
            )
        )

        table = (
            maximum_table.loc[
                mask,
                [
                    "candidate_id",
                    "source",
                    "target",
                    "label",
                ],
            ]
            .copy()
        )

        if (
            int(
                table[
                    "label"
                ].sum()
            )
            != len(
                positive_edges
            )
        ):
            raise RuntimeError(
                "Positive candidate count "
                "changed across ratios."
            )

        if (
            int(
                table[
                    "label"
                ]
                .eq(
                    0
                )
                .sum()
            )
            != desired_negative_count
        ):
            raise RuntimeError(
                "Negative candidate count "
                "does not match ratio."
            )

        tables[
            ratio
        ] = table

    return tables


def score_all_primary_methods(
    graph: nx.Graph,
    candidates: pd.DataFrame,
    methods_config: dict[str, Any],
) -> pd.DataFrame:
    methods = (
        methods_config[
            "methods"
        ]
    )

    lpi_parameters = (
        methods[
            "lpi"
        ][
            "parameters"
        ]
    )

    walk_steps = {
        int(
            methods[
                method_id
            ][
                "parameters"
            ][
                "steps"
            ]
        )
        for method_id
        in (
            "lrw",
            "srw",
            "pfp",
        )
    }

    if len(
        walk_steps
    ) != 1:
        raise ValueError(
            "LRW, SRW, and PFP must use "
            "the same primary step value."
        )

    steps = (
        walk_steps.pop()
    )

    scores = pd.concat(
        [
            score_local_similarity_candidates(
                graph,
                candidates,
            ),

            score_degree_based_candidates(
                graph,
                candidates,
            ),

            score_local_bayesian_candidates(
                graph,
                candidates,
            ),

            score_enhanced_local_candidates(
                graph,
                candidates,
            ),

            score_quasi_local_path_candidates(
                graph=graph,
                candidates=candidates,
                beta=float(
                    lpi_parameters[
                        "beta"
                    ]
                ),
                length=int(
                    lpi_parameters[
                        "length"
                    ]
                ),
            ),

            score_quasi_local_walk_candidates(
                graph=graph,
                candidates=candidates,
                steps=steps,
            ),
        ],
        axis=1,
    )

    method_ids = [
        method_id
        for (
            method_id,
            method_config,
        ) in methods.items()
        if method_config.get(
            "enabled",
            True,
        )
    ]

    if len(
        method_ids
    ) != 19:
        raise ValueError(
            "Expected 19 enabled primary methods, "
            f"found {len(method_ids)}."
        )

    if (
        set(
            scores.columns
        )
        != set(
            method_ids
        )
    ):
        missing = sorted(
            set(
                method_ids
            )
            - set(
                scores.columns
            )
        )

        unexpected = sorted(
            set(
                scores.columns
            )
            - set(
                method_ids
            )
        )

        raise ValueError(
            "Primary scorer columns do not match "
            "method registry. "
            f"Missing={missing}, "
            f"unexpected={unexpected}."
        )

    return scores[
        method_ids
    ]


def summarize_negative_sampling_robustness(
    fold_metrics: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    network_group_columns = [
        "benchmark",
        "network_id",
        "network",
        "domain",
        "negative_ratio",
        "is_primary_ratio",
        "family",
        "method_id",
        "method",
    ]

    network_summary = (
        fold_metrics
        .groupby(
            network_group_columns,
            sort=False,
        )[
            list(
                METRIC_COLUMNS
            )
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
        in network_summary.columns
    ]

    overall_group_columns = [
        "benchmark",
        "negative_ratio",
        "is_primary_ratio",
        "family",
        "method_id",
        "method",
    ]

    rows: list[
        dict[str, Any]
    ] = []

    for (
        values,
        group,
    ) in network_summary.groupby(
        overall_group_columns,
        sort=False,
    ):
        row = dict(
            zip(
                overall_group_columns,
                values,
                strict=True,
            )
        )

        row[
            "network_count"
        ] = len(
            group
        )

        for metric in (
            METRIC_COLUMNS
        ):
            network_values = (
                group[
                    f"{metric}_mean"
                ]
            )

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

        rows.append(
            row
        )

    overall_summary = (
        pd.DataFrame(
            rows
        )
    )

    overall_summary[
        "average_precision_rank"
    ] = (
        overall_summary
        .groupby(
            "negative_ratio"
        )[
            "average_precision_mean"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(
            int
        )
    )

    return (
        network_summary,
        overall_summary,
    )


def run_negative_sampling_robustness(
    benchmark_name: str = "revision",
    resume: bool = True,
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

    methods_config = (
        load_methods_config()
    )

    random_seed = int(
        experiment_config[
            "experiment"
        ][
            "random_seed"
        ]
    )

    n_folds = int(
        experiment_config[
            "experiment"
        ][
            "n_folds"
        ]
    )

    negative_config = (
        experiment_config[
            "negative_sampling"
        ]
    )

    primary_ratio = int(
        negative_config[
            "primary_ratio"
        ]
    )

    ratios = normalize_negative_ratios(
        ratios=
            negative_config[
                "robustness_ratios"
            ],
        primary_ratio=
            primary_ratio,
    )

    maximum_ratio = max(
        ratios
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

    methods = (
        methods_config[
            "methods"
        ]
    )

    method_ids = [
        method_id
        for (
            method_id,
            method_config,
        ) in methods.items()
        if method_config.get(
            "enabled",
            True,
        )
    ]

    expected_rows_per_fold = (
        len(
            method_ids
        )
        * len(
            ratios
        )
    )

    checkpoint_directory = (
        RESULTS_DIR
        / "robustness"
        / benchmark_name
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_directory
        / "negative_sampling_checkpoint.csv"
    )

    completed_rows = (
        pd.DataFrame()
    )

    completed_keys: set[
        tuple[
            str,
            int,
        ]
    ] = set()

    if (
        resume
        and checkpoint_path.exists()
    ):
        checkpoint = pd.read_csv(
            checkpoint_path
        )

        required_checkpoint_columns = {
            "network_id",
            "fold",
            "negative_ratio",
            "method_id",
        }

        if not (
            required_checkpoint_columns
            <= set(
                checkpoint.columns
            )
        ):
            raise ValueError(
                "Existing robustness checkpoint "
                "has an incompatible schema. "
                "Delete it or run with resume=False."
            )

        group_sizes = (
            checkpoint.groupby(
                [
                    "network_id",
                    "fold",
                ]
            )
            .size()
        )

        completed_keys = {
            (
                str(
                    network_id
                ),
                int(
                    fold_number
                ),
            )
            for (
                network_id,
                fold_number,
            ), size
            in group_sizes.items()
            if int(
                size
            )
            == expected_rows_per_fold
        }

        if completed_keys:
            key_index = (
                pd.MultiIndex
                .from_frame(
                    checkpoint[
                        [
                            "network_id",
                            "fold",
                        ]
                    ]
                    .assign(
                        network_id=
                            lambda frame:
                                frame[
                                    "network_id"
                                ].astype(
                                    str
                                ),
                        fold=
                            lambda frame:
                                frame[
                                    "fold"
                                ].astype(
                                    int
                                ),
                    )
                )
            )

            completed_rows = (
                checkpoint.loc[
                    key_index.isin(
                        completed_keys
                    )
                ]
                .copy()
            )

    fold_frames = []

    if not (
        completed_rows.empty
    ):
        fold_frames.append(
            completed_rows
        )

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

        full_graph = (
            load_processed_graph(
                network_config[
                    "name"
                ]
            )
        )

        for fold_number in range(
            1,
            n_folds + 1,
        ):
            fold_key = (
                str(
                    network_id
                ),
                int(
                    fold_number
                ),
            )

            if (
                fold_key
                in completed_keys
            ):
                continue

            (
                training_graph,
                primary_candidates,
            ) = load_fold_data(
                benchmark_name=
                    benchmark_name,
                network_id=
                    network_id,
                fold_number=
                    fold_number,
            )

            candidate_tables = (
                build_nested_robustness_candidates(
                    full_graph=
                        full_graph,
                    primary_candidates=
                        primary_candidates,
                    ratios=
                        ratios,
                    primary_ratio=
                        primary_ratio,
                    random_seed=
                        random_seed,
                    network_id=
                        network_id,
                    fold_number=
                        fold_number,
                )
            )

            maximum_candidates = (
                candidate_tables[
                    maximum_ratio
                ]
            )

            maximum_scores = (
                score_all_primary_methods(
                    graph=
                        training_graph,
                    candidates=
                        maximum_candidates,
                    methods_config=
                        methods_config,
                )
            )

            current_fold_frames = []

            for ratio in ratios:
                candidates = (
                    candidate_tables[
                        ratio
                    ]
                )

                scores = (
                    maximum_scores.loc[
                        candidates.index
                    ]
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

                metrics[
                    "negative_ratio"
                ] = ratio

                metrics[
                    "is_primary_ratio"
                ] = (
                    ratio
                    == primary_ratio
                )

                metrics[
                    "family"
                ] = (
                    metrics[
                        "method_id"
                    ]
                    .map(
                        {
                            method_id:
                                methods[
                                    method_id
                                ][
                                    "family"
                                ]
                            for method_id
                            in method_ids
                        }
                    )
                )

                metrics[
                    "method"
                ] = (
                    metrics[
                        "method_id"
                    ]
                    .map(
                        {
                            method_id:
                                methods[
                                    method_id
                                ][
                                    "name"
                                ]
                            for method_id
                            in method_ids
                        }
                    )
                )

                current_fold_frames.append(
                    metrics
                )

            current_fold = (
                pd.concat(
                    current_fold_frames,
                    ignore_index=True,
                )
            )

            if (
                len(
                    current_fold
                )
                != expected_rows_per_fold
            ):
                raise RuntimeError(
                    "Unexpected number of robustness rows "
                    f"for {network_id} "
                    f"fold {fold_number}: "
                    f"{len(current_fold)}."
                )

            fold_frames.append(
                current_fold
            )

            checkpoint_frame = (
                pd.concat(
                    fold_frames,
                    ignore_index=True,
                )
            )

            checkpoint_frame.to_csv(
                checkpoint_path,
                index=False,
            )

    if not fold_frames:
        raise RuntimeError(
            "No robustness results were generated or loaded."
        )

    fold_metrics = (
        pd.concat(
            fold_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "network_id",
                "fold",
                "negative_ratio",
                "method_id",
            ]
        )
        .drop_duplicates(
            subset=[
                "network_id",
                "fold",
                "negative_ratio",
                "method_id",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    expected_total_rows = (
        len(
            benchmark
        )
        * n_folds
        * len(
            ratios
        )
        * len(
            method_ids
        )
    )

    if (
        len(
            fold_metrics
        )
        != expected_total_rows
    ):
        raise RuntimeError(
            f"Expected {expected_total_rows} "
            "robustness rows, "
            f"found {len(fold_metrics)}."
        )

    (
        network_summary,
        overall_summary,
    ) = (
        summarize_negative_sampling_robustness(
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
            "negative_sampling_robustness_"
            "fold_metrics.csv"
        ),
        index=False,
    )

    network_summary.to_csv(
        SUMMARY_RESULTS_DIR
        / (
            f"{benchmark_name}_"
            "negative_sampling_robustness_"
            "network_summary.csv"
        ),
        index=False,
    )

    overall_summary.to_csv(
        SUMMARY_RESULTS_DIR
        / (
            f"{benchmark_name}_"
            "negative_sampling_robustness_"
            "overall_summary.csv"
        ),
        index=False,
    )

    fold_metrics.to_csv(
        checkpoint_path,
        index=False,
    )

    return (
        fold_metrics,
        network_summary,
        overall_summary,
    )