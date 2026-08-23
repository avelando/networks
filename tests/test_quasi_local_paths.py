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

    length_two = (
        adjacency
        @ adjacency
    )

    length_three = (
        length_two
        @ adjacency
    )

    number_of_nodes = float(
        graph.number_of_nodes()
    )

    expected_friendlink = (
        length_two
        / (
            number_of_nodes
            - 2.0
        )
        + length_three
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

    expected_fl = [
        expected_friendlink[
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
        scores["fl"].to_numpy()
        == pytest.approx(
            np.asarray(
                expected_fl
            )
        )
    )


def test_friendlink_matches_length_three_formula():
    graph = nx.Graph()

    graph.add_edges_from(
        [
            ("x", "a"),
            ("x", "b"),
            ("y", "a"),
            ("y", "b"),
            ("a", "b"),
        ]
    )

    candidates = pd.DataFrame(
        {
            "source": ["x"],
            "target": ["y"],
        }
    )

    scores = (
        score_quasi_local_path_candidates(
            graph=graph,
            candidates=candidates,
            friendlink_length=3,
        )
    )

    expected = (
        2.0 / 2.0
        + 2.0
        / (
            2.0
            * 2.0
            * 1.0
        )
    )

    assert scores.loc[
        0,
        "fl",
    ] == pytest.approx(
        expected
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

    assert (
        scores.loc[0, "fl"]
        == pytest.approx(
            scores.loc[1, "fl"]
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


def test_friendlink_rejects_unsupported_length():
    graph = nx.path_graph(
        [
            "0",
            "1",
            "2",
            "3",
        ]
    )

    candidates = pd.DataFrame(
        {
            "source": ["0"],
            "target": ["3"],
        }
    )

    with pytest.raises(
        ValueError,
        match="FriendLink implementation",
    ):
        score_quasi_local_path_candidates(
            graph=graph,
            candidates=candidates,
            friendlink_length=4,
        )


def test_friendlink_rejects_small_graph():
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
        match="more graph nodes",
    ):
        score_quasi_local_path_candidates(
            graph=graph,
            candidates=candidates,
        )