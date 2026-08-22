import networkx as nx
import numpy as np
import pandas as pd
import pytest

from link_prediction.methods.quasi_local_paths import (
    QUASI_LOCAL_PATH_METHODS,
    score_quasi_local_path_candidates,
)


def test_lpi_matches_adjacency_matrix_definition():
    graph = nx.Graph()

    graph.add_edges_from(
        [
            ("0", "1"),
            ("1", "2"),
            ("2", "3"),
            ("0", "2"),
        ]
    )

    candidates = pd.DataFrame(
        {
            "source": [
                "0",
                "1",
                "0",
            ],
            "target": [
                "3",
                "3",
                "1",
            ],
        }
    )

    beta = 0.1

    scores = (
        score_quasi_local_path_candidates(
            graph=graph,
            candidates=candidates,
            beta=beta,
            length=3,
        )
    )

    nodes = list(
        graph.nodes()
    )

    adjacency = nx.to_numpy_array(
        graph,
        nodelist=nodes,
        dtype=float,
    )

    expected_matrix = (
        adjacency
        @ adjacency
        + beta
        * adjacency
        @ adjacency
        @ adjacency
    )

    node_to_index = {
        node: index
        for index, node
        in enumerate(nodes)
    }

    expected = [
        expected_matrix[
            node_to_index[source],
            node_to_index[target],
        ]
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
    ]

    assert (
        tuple(scores.columns)
        == QUASI_LOCAL_PATH_METHODS
    )

    assert (
        scores["lpi"].to_numpy()
        == pytest.approx(
            np.asarray(expected)
        )
    )


def test_lpi_is_symmetric_for_undirected_graphs():
    graph = nx.cycle_graph(
        [
            "0",
            "1",
            "2",
            "3",
            "4",
        ]
    )

    candidates = pd.DataFrame(
        {
            "source": [
                "0",
                "3",
            ],
            "target": [
                "3",
                "0",
            ],
        }
    )

    scores = (
        score_quasi_local_path_candidates(
            graph=graph,
            candidates=candidates,
        )
    )

    assert (
        scores.loc[0, "lpi"]
        == pytest.approx(
            scores.loc[1, "lpi"]
        )
    )


def test_lpi_rejects_unsupported_length():
    graph = nx.path_graph(
        [
            "0",
            "1",
            "2",
        ]
    )

    candidates = pd.DataFrame(
        {
            "source": ["0"],
            "target": ["2"],
        }
    )

    with pytest.raises(
        ValueError,
        match="length=3",
    ):
        score_quasi_local_path_candidates(
            graph=graph,
            candidates=candidates,
            length=4,
        )