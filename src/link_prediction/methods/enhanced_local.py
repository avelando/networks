import networkx as nx
import pandas as pd

from link_prediction.sampling import canonical_edge

ENHANCED_LOCAL_METHODS = (
    "ra_cni",
    "ia1",
    "ia2",
    "car_ra",
    "fsw",
    "lit",
)


def validate_enhanced_local_inputs(
    graph: nx.Graph,
    candidates: pd.DataFrame,
) -> None:
    if graph.is_directed():
        raise ValueError(
            "Enhanced local methods require an undirected graph."
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


def count_internal_edges(
    nodes: set[object],
    neighbors: dict[object, set[object]],
) -> int:
    if len(nodes) < 2:
        return 0

    return (
        sum(
            len(
                neighbors[node]
                & nodes
            )
            for node in nodes
        )
        // 2
    )


def ra_cni_interaction_score(
    source_neighbors: set[object],
    target_neighbors: set[object],
    neighbors: dict[object, set[object]],
    degrees: dict[object, int],
) -> float:
    interaction_edges = set()

    for node_i in source_neighbors:
        for node_j in (
            neighbors[node_i]
            & target_neighbors
        ):
            if node_i == node_j:
                continue

            interaction_edges.add(
                canonical_edge(
                    node_i,
                    node_j,
                )
            )

    score = 0.0

    for node_i, node_j in interaction_edges:
        degree_i = degrees[node_i]
        degree_j = degrees[node_j]

        if degree_i == degree_j:
            continue

        score += abs(
            1.0 / degree_i
            - 1.0 / degree_j
        )

    return score


def functional_similarity_weight(
    source_neighbors: set[object],
    target_neighbors: set[object],
    average_degree: float,
) -> float:
    common_neighbors = (
        source_neighbors
        & target_neighbors
    )

    common_count = len(
        common_neighbors
    )

    if common_count == 0:
        return 0.0

    source_only_count = len(
        source_neighbors
        - target_neighbors
    )

    target_only_count = len(
        target_neighbors
        - source_neighbors
    )

    source_lambda = max(
        0.0,
        average_degree
        - (
            source_only_count
            + common_count
        ),
    )

    target_lambda = max(
        0.0,
        average_degree
        - (
            target_only_count
            + common_count
        ),
    )

    numerator = (
        2.0
        * common_count
    )

    source_term = (
        numerator
        / (
            source_only_count
            + 2.0 * common_count
            + source_lambda
        )
    )

    target_term = (
        numerator
        / (
            target_only_count
            + 2.0 * common_count
            + target_lambda
        )
    )

    return float(
        source_term
        * target_term
    )


def local_interacting_scores(
    graph: nx.Graph,
    candidates: pd.DataFrame,
    iterations: int,
) -> list[float]:
    if (
        isinstance(
            iterations,
            bool,
        )
        or not isinstance(
            iterations,
            int,
        )
        or iterations < 1
    ):
        raise ValueError(
            "LIT iterations must be "
            "a positive integer."
        )

    neighbors = {
        node: set(
            graph.neighbors(node)
        )
        for node in graph.nodes()
    }

    edge_weights = {
        canonical_edge(
            source,
            target,
        ): 1.0
        for source, target
        in graph.edges()
    }

    candidate_pairs = list(
        candidates[
            [
                "source",
                "target",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    candidate_scores = [
        0.0
        for _ in candidate_pairs
    ]

    for _ in range(
        iterations
    ):
        strengths = {
            node: sum(
                edge_weights[
                    canonical_edge(
                        node,
                        neighbor,
                    )
                ]
                for neighbor
                in node_neighbors
            )
            for node, node_neighbors
            in neighbors.items()
        }

        average_strength = (
            sum(
                strengths.values()
            )
            / len(strengths)
        )

        penalties = {
            node: max(
                0.0,
                average_strength
                - strength,
            )
            for node, strength
            in strengths.items()
        }

        def pair_score(
            source: object,
            target: object,
        ) -> float:
            common_neighbors = (
                neighbors[source]
                & neighbors[target]
            )

            numerator = sum(
                edge_weights[
                    canonical_edge(
                        common_neighbor,
                        source,
                    )
                ]
                + edge_weights[
                    canonical_edge(
                        common_neighbor,
                        target,
                    )
                ]
                for common_neighbor
                in common_neighbors
            )

            denominator = (
                strengths[source]
                + strengths[target]
                + penalties[source]
                + penalties[target]
            )

            if denominator == 0.0:
                return 0.0

            return float(
                numerator
                / denominator
            )

        candidate_scores = [
            pair_score(
                source,
                target,
            )
            for source, target
            in candidate_pairs
        ]

        edge_weights = {
            canonical_edge(
                source,
                target,
            ): pair_score(
                source,
                target,
            )
            for source, target
            in graph.edges()
        }

    return candidate_scores


def score_enhanced_local_candidates(
    graph: nx.Graph,
    candidates: pd.DataFrame,
    lit_iterations: int = 2,
) -> pd.DataFrame:
    validate_enhanced_local_inputs(
        graph=graph,
        candidates=candidates,
    )

    neighbors = {
        node: set(
            graph.neighbors(node)
        )
        for node in graph.nodes()
    }

    degrees = {
        node: len(node_neighbors)
        for node, node_neighbors
        in neighbors.items()
    }

    average_degree = (
        2.0
        * graph.number_of_edges()
        / graph.number_of_nodes()
    )

    lit_scores = (
        local_interacting_scores(
            graph=graph,
            candidates=candidates,
            iterations=lit_iterations,
        )
    )

    rows = []

    for candidate_index, (
        source,
        target,
    ) in enumerate(
        candidates[
            [
                "source",
                "target",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    ):
        source_neighbors = (
            neighbors[source]
        )

        target_neighbors = (
            neighbors[target]
        )

        common_neighbors = (
            source_neighbors
            & target_neighbors
        )

        common_count = len(
            common_neighbors
        )

        resource_allocation = sum(
            1.0 / degrees[node]
            for node
            in common_neighbors
        )

        interaction = (
            ra_cni_interaction_score(
                source_neighbors=
                    source_neighbors,
                target_neighbors=
                    target_neighbors,
                neighbors=neighbors,
                degrees=degrees,
            )
        )

        internal_edges = (
            count_internal_edges(
                nodes=common_neighbors,
                neighbors=neighbors,
            )
        )

        ia1 = sum(
            (
                len(
                    neighbors[node]
                    & common_neighbors
                )
                + 2.0
            )
            / degrees[node]
            for node
            in common_neighbors
        )

        if common_count > 0:
            ia2 = sum(
                (
                    internal_edges
                    + 2.0
                )
                / (
                    degrees[node]
                    * common_count
                )
                for node
                in common_neighbors
            )
        else:
            ia2 = 0.0

        car_ra = sum(
            len(
                neighbors[node]
                & common_neighbors
            )
            / degrees[node]
            for node
            in common_neighbors
        )

        fsw = (
            functional_similarity_weight(
                source_neighbors=
                    source_neighbors,
                target_neighbors=
                    target_neighbors,
                average_degree=
                    average_degree,
            )
        )

        rows.append(
            {
                "ra_cni":
                    float(
                        resource_allocation
                        + interaction
                    ),
                "ia1":
                    float(ia1),
                "ia2":
                    float(ia2),
                "car_ra":
                    float(car_ra),
                "fsw":
                    float(fsw),
                "lit":
                    float(
                        lit_scores[
                            candidate_index
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=ENHANCED_LOCAL_METHODS,
    )