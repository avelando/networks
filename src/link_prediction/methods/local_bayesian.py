import math

import networkx as nx
import pandas as pd

LOCAL_BAYESIAN_METHODS = (
    "lnb_ra",
)


def compute_lnb_roles(
    graph: nx.Graph,
) -> dict[object, float]:
    degrees = dict(
        graph.degree()
    )

    triangles = nx.triangles(
        graph
    )

    roles: dict[
        object,
        float,
    ] = {}

    for node, degree in degrees.items():
        possible_neighbor_pairs = (
            degree
            * (degree - 1)
            // 2
        )

        connected_neighbor_pairs = int(
            triangles[node]
        )

        disconnected_neighbor_pairs = (
            possible_neighbor_pairs
            - connected_neighbor_pairs
        )

        roles[node] = (
            connected_neighbor_pairs
            + 1.0
        ) / (
            disconnected_neighbor_pairs
            + 1.0
        )

    return roles


def score_local_bayesian_candidates(
    graph: nx.Graph,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    if graph.is_directed():
        raise ValueError(
            "Local Bayesian methods "
            "require an undirected graph."
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

    node_count = (
        graph.number_of_nodes()
    )

    edge_count = (
        graph.number_of_edges()
    )

    possible_edges = (
        node_count
        * (node_count - 1)
        / 2.0
    )

    if (
        node_count < 2
        or edge_count == 0
    ):
        raise ValueError(
            "LNB-RA requires a graph "
            "with at least two nodes "
            "and one edge."
        )

    prior_odds = (
        possible_edges
        / edge_count
        - 1.0
    )

    if prior_odds <= 0.0:
        raise ValueError(
            "LNB-RA requires a "
            "non-complete training graph."
        )

    neighbors = {
        node: set(
            graph.neighbors(node)
        )
        for node in graph.nodes()
    }

    degrees = dict(
        graph.degree()
    )

    roles = compute_lnb_roles(
        graph
    )

    log_prior_odds = math.log(
        prior_odds
    )

    scores: list[float] = []

    for (
        source,
        target,
    ) in candidates[
        [
            "source",
            "target",
        ]
    ].itertuples(
        index=False,
        name=None,
    ):
        if (
            source not in neighbors
            or target not in neighbors
        ):
            raise ValueError(
                "Candidate pair contains "
                "node absent from training graph: "
                f"({source}, {target})"
            )

        common_neighbors = (
            neighbors[source]
            & neighbors[target]
        )

        score = sum(
            (
                log_prior_odds
                + math.log(
                    roles[node]
                )
            )
            / degrees[node]
            for node
            in common_neighbors
        )

        scores.append(
            float(score)
        )

    return pd.DataFrame(
        {
            "lnb_ra": scores,
        }
    )