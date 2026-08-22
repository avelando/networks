import math

import networkx as nx
import pandas as pd

LOCAL_SIMILARITY_METHODS = (
    "cn",
    "aa",
    "ra",
    "ja",
    "sa",
    "so",
    "hpi",
    "hdi",
    "llhn",
)


def score_local_similarity_candidates(
    graph: nx.Graph,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    if graph.is_directed():
        raise ValueError(
            "Local similarity methods "
            "require an undirected graph."
        )

    required_columns = {
        "source",
        "target",
    }

    missing_columns = (
        required_columns
        - set(
            candidates.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Candidate table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    neighbors = {
        node: set(
            graph.neighbors(
                node
            )
        )
        for node in graph.nodes()
    }

    degrees = {
        node: len(
            node_neighbors
        )
        for (
            node,
            node_neighbors,
        ) in neighbors.items()
    }

    rows = []

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

        source_degree = (
            degrees[source]
        )

        target_degree = (
            degrees[target]
        )

        common_count = len(
            common_neighbors
        )

        union_size = (
            source_degree
            + target_degree
            - common_count
        )

        degree_product = (
            source_degree
            * target_degree
        )

        degree_sum = (
            source_degree
            + target_degree
        )

        minimum_degree = min(
            source_degree,
            target_degree,
        )

        maximum_degree = max(
            source_degree,
            target_degree,
        )

        aa = sum(
            1.0
            / math.log(
                degrees[node]
            )
            for node
            in common_neighbors
            if degrees[node] > 1
        )

        ra = sum(
            1.0
            / degrees[node]
            for node
            in common_neighbors
            if degrees[node] > 0
        )

        rows.append(
            {
                "cn":
                    float(
                        common_count
                    ),
                "aa":
                    float(aa),
                "ra":
                    float(ra),
                "ja":
                    (
                        float(
                            common_count
                            / union_size
                        )
                        if union_size
                        else 0.0
                    ),
                "sa":
                    (
                        float(
                            common_count
                            / math.sqrt(
                                degree_product
                            )
                        )
                        if degree_product
                        else 0.0
                    ),
                "so":
                    (
                        float(
                            2.0
                            * common_count
                            / degree_sum
                        )
                        if degree_sum
                        else 0.0
                    ),
                "hpi":
                    (
                        float(
                            common_count
                            / minimum_degree
                        )
                        if minimum_degree
                        else 0.0
                    ),
                "hdi":
                    (
                        float(
                            common_count
                            / maximum_degree
                        )
                        if maximum_degree
                        else 0.0
                    ),
                "llhn":
                    (
                        float(
                            common_count
                            / degree_product
                        )
                        if degree_product
                        else 0.0
                    ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=LOCAL_SIMILARITY_METHODS,
    )