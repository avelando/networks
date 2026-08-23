import networkx as nx
import numpy as np
import pandas as pd

QUASI_LOCAL_PATH_METHODS = (
    "lpi",
    "fl",
)


def score_quasi_local_path_candidates(
    graph: nx.Graph,
    candidates: pd.DataFrame,
    beta: float = 0.001,
    length: int = 3,
    friendlink_length: int = 3,
) -> pd.DataFrame:
    if graph.is_directed():
        raise ValueError(
            "Quasi-local path methods require "
            "an undirected graph."
        )

    if length != 3:
        raise ValueError(
            "This LPI implementation requires length=3."
        )

    if friendlink_length != 3:
        raise ValueError(
            "This FriendLink implementation "
            "requires length=3."
        )

    if beta < 0:
        raise ValueError(
            "beta must be non-negative."
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

    nodes = list(
        graph.nodes()
    )

    if (
        len(nodes)
        <= friendlink_length
    ):
        raise ValueError(
            "FriendLink requires more graph "
            "nodes than its maximum length."
        )

    node_to_index = {
        node: index
        for index, node
        in enumerate(nodes)
    }

    source_nodes = (
        candidates["source"].to_numpy()
    )

    target_nodes = (
        candidates["target"].to_numpy()
    )

    missing_nodes = (
        set(source_nodes)
        | set(target_nodes)
    ) - set(node_to_index)

    if missing_nodes:
        raise ValueError(
            "Candidate table contains "
            f"{len(missing_nodes)} nodes "
            "absent from the training graph."
        )

    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=nodes,
        dtype=np.float64,
        format="csr",
    )

    length_two = (
        adjacency
        @ adjacency
    ).tocsr()

    length_three = (
        length_two
        @ adjacency
    ).tocsr()

    source_indices = np.fromiter(
        (
            node_to_index[node]
            for node in source_nodes
        ),
        dtype=np.int64,
        count=len(candidates),
    )

    target_indices = np.fromiter(
        (
            node_to_index[node]
            for node in target_nodes
        ),
        dtype=np.int64,
        count=len(candidates),
    )

    length_two_scores = np.asarray(
        length_two[
            source_indices,
            target_indices,
        ],
        dtype=np.float64,
    ).ravel()

    length_three_scores = np.asarray(
        length_three[
            source_indices,
            target_indices,
        ],
        dtype=np.float64,
    ).ravel()

    number_of_nodes = float(
        len(nodes)
    )

    friendlink_scores = (
        length_two_scores
        / (
            number_of_nodes
            - 2.0
        )
        + length_three_scores
        / (
            2.0
            * (
                number_of_nodes
                - 2.0
            )
            * (
                number_of_nodes
                - 3.0
            )
        )
    )

    return pd.DataFrame(
        {
            "lpi": (
                length_two_scores
                + beta
                * length_three_scores
            ),
            "fl":
                friendlink_scores,
        },
        columns=
            QUASI_LOCAL_PATH_METHODS,
    )