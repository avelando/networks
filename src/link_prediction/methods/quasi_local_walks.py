from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from link_prediction.sampling import stable_node_key

QUASI_LOCAL_WALK_METHODS = (
    "lrw",
    "srw",
    "pfp",
)


def validate_quasi_local_walk_inputs(
    graph: nx.Graph,
    candidates: pd.DataFrame,
    steps: int,
) -> None:
    if graph.is_directed():
        raise ValueError(
            "Quasi-local walk methods require an undirected graph."
        )

    if steps < 1:
        raise ValueError(
            "steps must be at least 1."
        )

    required_columns = {
        "source",
        "target",
    }

    missing_columns = (
        required_columns
        - set(candidates.columns)
    )

    if missing_columns:
        raise ValueError(
            "Candidate table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    candidate_nodes = (
        set(candidates["source"])
        | set(candidates["target"])
    )

    missing_nodes = (
        candidate_nodes
        - set(graph.nodes())
    )

    if missing_nodes:
        raise ValueError(
            "Candidate table contains "
            f"{len(missing_nodes)} nodes "
            "absent from the training graph."
        )


def candidate_entries(
    matrix,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        matrix[
            source_indices,
            target_indices,
        ],
        dtype=np.float64,
    ).ravel()


def score_local_and_superposed_random_walks(
    graph: nx.Graph,
    candidates: pd.DataFrame,
    steps: int,
) -> pd.DataFrame:
    validate_quasi_local_walk_inputs(
        graph=graph,
        candidates=candidates,
        steps=steps,
    )

    nodes = list(
        graph.nodes()
    )

    node_to_index = {
        node: index
        for index, node
        in enumerate(nodes)
    }

    degrees = np.asarray(
        [
            graph.degree(node)
            for node in nodes
        ],
        dtype=np.float64,
    )

    if np.any(
        degrees <= 0.0
    ):
        raise ValueError(
            "Random-walk methods require "
            "a graph without isolated nodes."
        )

    adjacency = (
        nx.to_scipy_sparse_array(
            graph,
            nodelist=nodes,
            dtype=np.float64,
            format="csr",
        )
    )

    transition = (
        sparse.diags(
            1.0 / degrees
        )
        @ adjacency
    ).tocsr()

    current = transition.copy()
    accumulated = transition.copy()

    for _ in range(
        2,
        steps + 1,
    ):
        current = (
            current
            @ transition
        ).tocsr()

        accumulated = (
            accumulated
            + current
        ).tocsr()

    source_indices = np.fromiter(
        (
            node_to_index[node]
            for node
            in candidates["source"]
        ),
        dtype=np.int64,
        count=len(candidates),
    )

    target_indices = np.fromiter(
        (
            node_to_index[node]
            for node
            in candidates["target"]
        ),
        dtype=np.int64,
        count=len(candidates),
    )

    two_m = (
        2.0
        * graph.number_of_edges()
    )

    source_weights = (
        degrees[source_indices]
        / two_m
    )

    target_weights = (
        degrees[target_indices]
        / two_m
    )

    current_source_to_target = (
        candidate_entries(
            current,
            source_indices,
            target_indices,
        )
    )

    current_target_to_source = (
        candidate_entries(
            current,
            target_indices,
            source_indices,
        )
    )

    accumulated_source_to_target = (
        candidate_entries(
            accumulated,
            source_indices,
            target_indices,
        )
    )

    accumulated_target_to_source = (
        candidate_entries(
            accumulated,
            target_indices,
            source_indices,
        )
    )

    lrw_scores = (
        source_weights
        * current_source_to_target
        + target_weights
        * current_target_to_source
    )

    srw_scores = (
        source_weights
        * accumulated_source_to_target
        + target_weights
        * accumulated_target_to_source
    )

    return pd.DataFrame(
        {
            "lrw": lrw_scores,
            "srw": srw_scores,
        }
    )


def propflow_from_source(
    graph: nx.Graph,
    source: object,
    steps: int,
) -> dict[object, float]:
    if steps < 1:
        raise ValueError(
            "steps must be at least 1."
        )

    if source not in graph:
        raise ValueError(
            "Source node is absent "
            f"from the graph: {source}"
        )

    scores = defaultdict(
        float
    )

    scores[source] = 1.0

    found_nodes = {
        source
    }

    frontier = [
        source
    ]

    for _ in range(
        steps
    ):
        next_frontier = set()

        for node in sorted(
            frontier,
            key=stable_node_key,
        ):
            degree = graph.degree(
                node
            )

            if degree == 0:
                continue

            contribution = (
                scores[node]
                / degree
            )

            for neighbor in sorted(
                graph.neighbors(node),
                key=stable_node_key,
            ):
                scores[
                    neighbor
                ] += contribution

                if (
                    neighbor
                    not in found_nodes
                ):
                    found_nodes.add(
                        neighbor
                    )

                    next_frontier.add(
                        neighbor
                    )

        frontier = sorted(
            next_frontier,
            key=stable_node_key,
        )

        if not frontier:
            break

    return dict(
        scores
    )


def score_propflow_candidates(
    graph: nx.Graph,
    candidates: pd.DataFrame,
    steps: int,
) -> np.ndarray:
    sources = (
        candidates[
            "source"
        ].unique()
    )

    cache = {
        source:
            propflow_from_source(
                graph=graph,
                source=source,
                steps=steps,
            )
        for source in sources
    }

    return np.fromiter(
        (
            cache[
                source
            ].get(
                target,
                0.0,
            )
            for source, target
            in candidates[
                [
                    "source",
                    "target",
                ]
            ].itertuples(
                index=False,
                name=None,
            )
        ),
        dtype=np.float64,
        count=len(candidates),
    )


def score_quasi_local_walk_candidates(
    graph: nx.Graph,
    candidates: pd.DataFrame,
    steps: int = 3,
) -> pd.DataFrame:
    validate_quasi_local_walk_inputs(
        graph=graph,
        candidates=candidates,
        steps=steps,
    )

    random_walk_scores = (
        score_local_and_superposed_random_walks(
            graph=graph,
            candidates=candidates,
            steps=steps,
        )
    )

    propflow_scores = (
        score_propflow_candidates(
            graph=graph,
            candidates=candidates,
            steps=steps,
        )
    )

    return pd.DataFrame(
        {
            "lrw":
                random_walk_scores[
                    "lrw"
                ].to_numpy(),
            "srw":
                random_walk_scores[
                    "srw"
                ].to_numpy(),
            "pfp":
                propflow_scores,
        },
        columns=QUASI_LOCAL_WALK_METHODS,
    )