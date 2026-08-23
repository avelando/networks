import networkx as nx
import pandas as pd
import pytest

from link_prediction.methods.enhanced_local import (
    ENHANCED_LOCAL_METHODS,
    functional_similarity_weight,
    score_enhanced_local_candidates,
)


def build_enhanced_test_graph() -> nx.Graph:
    graph = nx.Graph()

    graph.add_edges_from(
        [
            ("x", "a"),
            ("x", "b"),
            ("x", "p"),
            ("y", "a"),
            ("y", "b"),
            ("y", "q"),
            ("a", "b"),
            ("p", "q"),
            ("q", "s"),
        ]
    )

    return graph


def test_enhanced_local_scores():
    graph = build_enhanced_test_graph()

    candidates = pd.DataFrame(
        {
            "source": ["x"],
            "target": ["y"],
        }
    )

    scores = (
        score_enhanced_local_candidates(
            graph=graph,
            candidates=candidates,
        )
        .iloc[0]
    )

    assert (
        tuple(scores.index)
        == ENHANCED_LOCAL_METHODS
    )

    assert scores[
        "ra_cni"
    ] == pytest.approx(
        5 / 6
    )

    assert scores[
        "ia1"
    ] == pytest.approx(
        2.0
    )

    assert scores[
        "ia2"
    ] == pytest.approx(
        1.0
    )

    assert scores[
        "car_ra"
    ] == pytest.approx(
        2 / 3
    )

    assert scores[
        "fsw"
    ] == pytest.approx(
        16 / 25
    )


def test_ia1_uses_per_common_neighbor_internal_links():
    graph = nx.Graph()

    graph.add_edges_from(
        [
            ("x", "a"),
            ("x", "b"),
            ("x", "c"),
            ("y", "a"),
            ("y", "b"),
            ("y", "c"),
            ("a", "b"),
            ("b", "c"),
            ("a", "p"),
            ("c", "q"),
        ]
    )

    candidates = pd.DataFrame(
        {
            "source": ["x"],
            "target": ["y"],
        }
    )

    scores = (
        score_enhanced_local_candidates(
            graph=graph,
            candidates=candidates,
        )
        .iloc[0]
    )

    expected_ia1 = (
        3.0 / 4.0
        + 4.0 / 4.0
        + 3.0 / 4.0
    )

    expected_ia2 = (
        4.0 / 12.0
        + 4.0 / 12.0
        + 4.0 / 12.0
    )

    assert scores[
        "ia1"
    ] == pytest.approx(
        expected_ia1
    )

    assert scores[
        "ia2"
    ] == pytest.approx(
        expected_ia2
    )


def test_enhanced_local_scores_are_symmetric():
    graph = build_enhanced_test_graph()

    candidates = pd.DataFrame(
        {
            "source": [
                "x",
                "y",
            ],
            "target": [
                "y",
                "x",
            ],
        }
    )

    scores = (
        score_enhanced_local_candidates(
            graph=graph,
            candidates=candidates,
        )
    )

    for method_id in (
        ENHANCED_LOCAL_METHODS
    ):
        assert scores.loc[
            0,
            method_id,
        ] == pytest.approx(
            scores.loc[
                1,
                method_id,
            ]
        )


def test_fsw_uses_bidirectional_chua_product():
    source_neighbors = {
        "a",
        "b",
    }

    target_neighbors = {
        "a",
        "b",
        "c",
        "d",
    }

    score = (
        functional_similarity_weight(
            source_neighbors=
                source_neighbors,
            target_neighbors=
                target_neighbors,
            average_degree=3.0,
        )
    )

    expected = (
        (4.0 / 5.0)
        * (4.0 / 6.0)
    )

    assert score == pytest.approx(
        expected
    )


def test_enhanced_local_zero_common_neighbors():
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

    scores = (
        score_enhanced_local_candidates(
            graph=graph,
            candidates=candidates,
        )
        .iloc[0]
    )

    assert scores["ia1"] == 0.0
    assert scores["ia2"] == 0.0
    assert scores["car_ra"] == 0.0
    assert scores["fsw"] == 0.0


def test_enhanced_local_rejects_missing_node():
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
            "target": ["99"],
        }
    )

    with pytest.raises(
        ValueError,
        match="absent from the training graph",
    ):
        score_enhanced_local_candidates(
            graph=graph,
            candidates=candidates,
        )