import networkx as nx
import pandas as pd
import pytest

from link_prediction.methods.degree_based import (
    DEGREE_BASED_METHODS,
    score_degree_based_candidates,
)


def test_preferential_attachment_scores():
    graph = nx.Graph()

    graph.add_edges_from(
        [
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("1", "2"),
            ("2", "3"),
            ("3", "4"),
        ]
    )

    candidates = pd.DataFrame(
        {
            "source": [
                "1",
                "1",
            ],
            "target": [
                "3",
                "4",
            ],
        }
    )

    scores = (
        score_degree_based_candidates(
            graph,
            candidates,
        )
    )

    assert tuple(
        scores.columns
    ) == DEGREE_BASED_METHODS

    assert scores[
        "pa"
    ].tolist() == pytest.approx(
        [
            6.0,
            2.0,
        ]
    )


def test_degree_based_rejects_missing_node():
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
                "0",
            ],
            "target": [
                "99",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="node absent",
    ):
        score_degree_based_candidates(
            graph,
            candidates,
        )