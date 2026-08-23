import json
import random
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from networkx.utils import UnionFind

from link_prediction.config import (
    FOLDS_DATA_DIR,
    SUMMARY_RESULTS_DIR,
    load_experiment_config,
    load_networks_config,
)
from link_prediction.execution import (
    run_process_tasks,
)
from link_prediction.preprocessing import (
    load_processed_graph,
)
from link_prediction.sampling import (
    Edge,
    canonical_edge,
    derive_seed,
    edge_sort_key,
    graph_edge_set,
    sample_non_edges,
)


def build_fixed_spanning_tree(
    graph: nx.Graph,
    seed: int,
) -> list[Edge]:
    if graph.is_directed():
        raise ValueError(
            "Spanning-tree construction requires "
            "an undirected graph."
        )

    if graph.number_of_nodes() == 0:
        raise ValueError(
            "Cannot build a spanning tree for an empty graph."
        )

    if not nx.is_connected(graph):
        raise ValueError(
            "The graph must be connected before "
            "fold construction."
        )

    edges = sorted(
        graph_edge_set(graph),
        key=edge_sort_key,
    )

    rng = random.Random(seed)
    rng.shuffle(edges)

    union_find = UnionFind(
        graph.nodes()
    )

    tree_edges: list[Edge] = []

    for source, target in edges:
        if (
            union_find[source]
            == union_find[target]
        ):
            continue

        union_find.union(
            source,
            target,
        )

        tree_edges.append(
            canonical_edge(
                source,
                target,
            )
        )

        if (
            len(tree_edges)
            == graph.number_of_nodes() - 1
        ):
            break

    if (
        len(tree_edges)
        != graph.number_of_nodes() - 1
    ):
        raise RuntimeError(
            "Failed to construct a complete spanning tree."
        )

    return sorted(
        tree_edges,
        key=edge_sort_key,
    )


def split_removable_edges(
    graph: nx.Graph,
    spanning_tree_edges: list[Edge],
    n_folds: int,
    seed: int,
) -> list[list[Edge]]:
    if n_folds < 2:
        raise ValueError(
            "n_folds must be at least 2."
        )

    tree_edge_set = set(
        spanning_tree_edges
    )

    removable_edges = sorted(
        graph_edge_set(graph)
        - tree_edge_set,
        key=edge_sort_key,
    )

    if len(removable_edges) < n_folds:
        raise ValueError(
            "The number of removable edges must "
            "be at least the number of folds."
        )

    rng = random.Random(seed)
    rng.shuffle(removable_edges)

    quotient, remainder = divmod(
        len(removable_edges),
        n_folds,
    )

    folds: list[list[Edge]] = []

    start = 0

    for fold_index in range(n_folds):
        fold_size = (
            quotient
            + (
                1
                if fold_index < remainder
                else 0
            )
        )

        stop = start + fold_size

        folds.append(
            removable_edges[start:stop]
        )

        start = stop

    return folds


def build_training_graph(
    graph: nx.Graph,
    positive_edges: list[Edge],
) -> nx.Graph:
    training_graph = graph.copy()

    training_graph.remove_edges_from(
        positive_edges
    )

    if not nx.is_connected(
        training_graph
    ):
        raise RuntimeError(
            "Training graph became disconnected."
        )

    return training_graph


def build_candidate_table(
    positive_edges: list[Edge],
    negative_edges: list[Edge],
    seed: int,
) -> pd.DataFrame:
    records = [
        {
            "source": source,
            "target": target,
            "label": 1,
        }
        for source, target
        in positive_edges
    ]

    records.extend(
        {
            "source": source,
            "target": target,
            "label": 0,
        }
        for source, target
        in negative_edges
    )

    rng = random.Random(seed)
    rng.shuffle(records)

    dataframe = pd.DataFrame(
        records
    )

    dataframe.insert(
        0,
        "candidate_id",
        range(len(dataframe)),
    )

    return dataframe


def write_sorted_edgelist(
    graph: nx.Graph,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    edges = sorted(
        graph_edge_set(graph),
        key=edge_sort_key,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for source, target in edges:
            file.write(
                f"{source} {target}\n"
            )


def write_edge_table(
    edges: list[Edge],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        sorted(
            edges,
            key=edge_sort_key,
        ),
        columns=[
            "source",
            "target",
        ],
    )

    dataframe.to_csv(
        path,
        index=False,
    )


def build_network_folds(
    graph: nx.Graph,
    network_id: str,
    network_name: str,
    benchmark_name: str,
    n_folds: int,
    negative_ratio: int,
    random_seed: int,
) -> pd.DataFrame:
    if negative_ratio < 1:
        raise ValueError(
            "negative_ratio must be at least 1."
        )

    tree_seed = derive_seed(
        random_seed,
        network_id,
        "spanning_tree",
    )

    split_seed = derive_seed(
        random_seed,
        network_id,
        "positive_folds",
    )

    spanning_tree_edges = (
        build_fixed_spanning_tree(
            graph=graph,
            seed=tree_seed,
        )
    )

    positive_folds = (
        split_removable_edges(
            graph=graph,
            spanning_tree_edges=spanning_tree_edges,
            n_folds=n_folds,
            seed=split_seed,
        )
    )

    network_directory = (
        FOLDS_DATA_DIR
        / benchmark_name
        / network_id
    )

    network_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_edge_table(
        spanning_tree_edges,
        network_directory
        / "spanning_tree.csv",
    )

    removable_edges = sorted(
        graph_edge_set(graph)
        - set(spanning_tree_edges),
        key=edge_sort_key,
    )

    write_edge_table(
        removable_edges,
        network_directory
        / "removable_edges.csv",
    )

    summary_rows: list[
        dict[str, Any]
    ] = []

    for (
        fold_number,
        positive_edges,
    ) in enumerate(
        positive_folds,
        start=1,
    ):
        negative_seed = derive_seed(
            random_seed,
            network_id,
            "negatives",
            fold_number,
        )

        candidate_seed = derive_seed(
            random_seed,
            network_id,
            "candidates",
            fold_number,
        )

        negative_edges = (
            sample_non_edges(
                graph=graph,
                n_samples=(
                    len(positive_edges)
                    * negative_ratio
                ),
                seed=negative_seed,
            )
        )

        training_graph = (
            build_training_graph(
                graph=graph,
                positive_edges=positive_edges,
            )
        )

        candidate_table = (
            build_candidate_table(
                positive_edges=positive_edges,
                negative_edges=negative_edges,
                seed=candidate_seed,
            )
        )

        fold_directory = (
            network_directory
            / f"fold_{fold_number:02d}"
        )

        fold_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_sorted_edgelist(
            training_graph,
            fold_directory
            / "train.edgelist",
        )

        write_edge_table(
            positive_edges,
            fold_directory
            / "positives.csv",
        )

        candidate_table.to_csv(
            fold_directory
            / "candidates.csv",
            index=False,
        )

        metadata = {
            "benchmark": benchmark_name,
            "network_id": network_id,
            "network": network_name,
            "fold": fold_number,
            "random_seed": random_seed,
            "spanning_tree_seed": tree_seed,
            "positive_split_seed": split_seed,
            "negative_seed": negative_seed,
            "candidate_seed": candidate_seed,
            "negative_ratio": negative_ratio,
            "nodes": (
                graph.number_of_nodes()
            ),
            "original_edges": (
                graph.number_of_edges()
            ),
            "spanning_tree_edges": (
                len(spanning_tree_edges)
            ),
            "removable_edges": (
                len(removable_edges)
            ),
            "positive_edges": (
                len(positive_edges)
            ),
            "negative_edges": (
                len(negative_edges)
            ),
            "training_edges": (
                training_graph.number_of_edges()
            ),
            "candidate_pairs": (
                len(candidate_table)
            ),
            "training_connected": (
                nx.is_connected(
                    training_graph
                )
            ),
        }

        metadata_path = (
            fold_directory
            / "metadata.json"
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
                sort_keys=True,
            )

        summary_rows.append(
            metadata
        )

    return pd.DataFrame(
        summary_rows
    )


def build_network_folds_task(
    benchmark_name: str,
    network_id: str,
    network_name: str,
    n_folds: int,
    negative_ratio: int,
    random_seed: int,
) -> pd.DataFrame:
    graph = load_processed_graph(
        network_name
    )

    return build_network_folds(
        graph=graph,
        network_id=network_id,
        network_name=network_name,
        benchmark_name=benchmark_name,
        n_folds=n_folds,
        negative_ratio=negative_ratio,
        random_seed=random_seed,
    )


def prepare_benchmark(
    benchmark_name: str = "revision",
    overwrite: bool = False,
    max_workers: int | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiment_config = (
        load_experiment_config()
    )

    networks_config = (
        load_networks_config()
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

    negative_ratio = int(
        experiment_config[
            "negative_sampling"
        ][
            "primary_ratio"
        ]
    )

    connectivity_config = (
        experiment_config[
            "connectivity"
        ]
    )

    if not connectivity_config.get(
        "preserve_spanning_tree",
        True,
    ):
        raise ValueError(
            "The current experimental protocol "
            "requires spanning-tree preservation."
        )

    strategy = (
        connectivity_config.get(
            "spanning_tree_strategy",
            "seeded_randomized_kruskal",
        )
    )

    if (
        strategy
        != "seeded_randomized_kruskal"
    ):
        raise ValueError(
            "Unsupported spanning-tree "
            f"strategy: {strategy}"
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

        tasks.append(
            (
                benchmark_name,
                network_id,
                network_config[
                    "name"
                ],
                n_folds,
                negative_ratio,
                random_seed,
            )
        )

    summaries = run_process_tasks(
        build_network_folds_task,
        tasks,
        max_workers=max_workers,
        profile="fold_building",
        label="fold building",
    )

    fold_summary = pd.concat(
        summaries,
        ignore_index=True,
    )

    SUMMARY_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        SUMMARY_RESULTS_DIR
        / f"{benchmark_name}_fold_summary.csv"
    )

    fold_summary.to_csv(
        output_path,
        index=False,
    )

    return fold_summary