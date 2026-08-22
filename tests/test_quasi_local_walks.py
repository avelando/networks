import networkx as nx
import numpy as np
import pandas as pd
import pytest

from link_prediction.methods.quasi_local_walks import (
    QUASI_LOCAL_WALK_METHODS,
    propflow_from_source,
    score_quasi_local_walk_candidates,
)


def build_walk_test_graph():
    graph = nx.Graph()

    graph.add_edges_from(
        [
            ("0", "1"),
            ("1", "2"),
            ("2", "3"),
            ("0", "2"),
        ]
    )

    return graph


def test_lrw_and_srw_match_matrix_definition():
    graph = build_walk_test_graph()

    candidates = pd.DataFrame(
        {
            "source": [
                "0",
                "1",
            ],
            "target": [
                "3",
                "3",
            ],
        }
    )

    steps = 3

    scores = (
        score_quasi_local_walk_candidates(
            graph=graph,
            candidates=candidates,
            steps=steps,
        )
    )

    nodes = list(
        graph.nodes()
    )

    node_to_index = {
        node: index
        for index, node
        in enumerate(nodes)
    }

    adjacency = (
        nx.to_numpy_array(
            graph,
            nodelist=nodes,
            dtype=float,
        )
    )

    degrees = (
        adjacency.sum(
            axis=1
        )
    )

    transition = (
        adjacency
        / degrees[:, None]
    )

    current = (
        np.linalg.matrix_power(
            transition,
            steps,
        )
    )

    accumulated = sum(
        np.linalg.matrix_power(
            transition,
            step,
        )
        for step in range(
            1,
            steps + 1,
        )
    )

    two_m = (
        2.0
        * graph.number_of_edges()
    )

    expected_lrw = []
    expected_srw = []

    for (
        source,
        target,
    ) in candidates.itertuples(
        index=False,
        name=None,
    ):
        source_index = (
            node_to_index[source]
        )

        target_index = (
            node_to_index[target]
        )

        source_weight = (
            degrees[source_index]
            / two_m
        )

        target_weight = (
            degrees[target_index]
            / two_m
        )

        expected_lrw.append(
            source_weight
            * current[
                source_index,
                target_index,
            ]
            + target_weight
            * current[
                target_index,
                source_index,
            ]
        )

        expected_srw.append(
            source_weight
            * accumulated[
                source_index,
                target_index,
            ]
            + target_weight
            * accumulated[
                target_index,
                source_index,
            ]
        )

    assert (
        tuple(scores.columns)
        == QUASI_LOCAL_WALK_METHODS
    )

    assert (
        scores[
            "lrw"
        ].to_numpy()
        == pytest.approx(
            expected_lrw
        )
    )

    assert (
        scores[
            "srw"
        ].to_numpy()
        == pytest.approx(
            expected_srw
        )
    )


def test_lrw_and_srw_are_symmetric():
    graph = build_walk_test_graph()

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
        score_quasi_local_walk_candidates(
            graph=graph,
            candidates=candidates,
            steps=3,
        )
    )

    assert (
        scores.loc[
            0,
            "lrw",
        ]
        == pytest.approx(
            scores.loc[
                1,
                "lrw",
            ]
        )
    )

    assert (
        scores.loc[
            0,
            "srw",
        ]
        == pytest.approx(
            scores.loc[
                1,
                "srw",
            ]
        )
    )


def test_propflow_matches_restricted_walk_definition():
    graph = nx.path_graph(
        [
            "0",
            "1",
            "2",
        ]
    )

    scores = (
        propflow_from_source(
            graph=graph,
            source="0",
            steps=2,
        )
    )

    assert scores[
        "0"
    ] == pytest.approx(
        1.5
    )

    assert scores[
        "1"
    ] == pytest.approx(
        1.0
    )

    assert scores[
        "2"
    ] == pytest.approx(
        0.5
    )


def test_quasi_local_walk_rejects_missing_node():
    graph = nx.path_graph(
        [
            "0",
            "1",
            "2",
        ]
    )

    candidates = pd.DataFrame(
        {
            "source": [
                "0"
            ],
            "target": [
                "99"
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="absent from the training graph",
    ):
        score_quasi_local_walk_candidates(
            graph=graph,
            candidates=candidates,
            steps=3,
        )


def test_quasi_local_walk_rejects_invalid_steps():
    graph = nx.path_graph(
        [
            "0",
            "1",
            "2",
        ]
    )

    candidates = pd.DataFrame(
        {
            "source": [
                "0"
            ],
            "target": [
                "2"
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="steps must be at least 1",
    ):
        score_quasi_local_walk_candidates(
            graph=graph,
            candidates=candidates,
            steps=0,
        )