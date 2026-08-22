import math

import networkx as nx
import pandas as pd
import pytest

from link_prediction.methods.local_similarity import (
    LOCAL_SIMILARITY_METHODS,
    score_local_similarity_candidates,
)


def test_local_similarity_scores():
    graph = nx.Graph()

    graph.add_edges_from(
        [
            ("0", "2"),
            ("1", "2"),
            ("0", "3"),
            ("1", "3"),
            ("2", "4"),
        ]
    )

    candidates = pd.DataFrame(
        {
            "source": [
                "0"
            ],
            "target": [
                "1"
            ],
        }
    )

    scores = (
        score_local_similarity_candidates(
            graph,
            candidates,
        ).iloc[0]
    )

    assert tuple(
        scores.index
    ) == (
        LOCAL_SIMILARITY_METHODS
    )

    assert scores[
        "cn"
    ] == pytest.approx(
        2.0
    )

    assert scores[
        "aa"
    ] == pytest.approx(
        1 / math.log(3)
        + 1 / math.log(2)
    )

    assert scores[
        "ra"
    ] == pytest.approx(
        1 / 3
        + 1 / 2
    )

    assert scores[
        "ja"
    ] == pytest.approx(1.0)

    assert scores[
        "sa"
    ] == pytest.approx(1.0)

    assert scores[
        "so"
    ] == pytest.approx(1.0)

    assert scores[
        "hpi"
    ] == pytest.approx(1.0)

    assert scores[
        "hdi"
    ] == pytest.approx(1.0)

    assert scores[
        "llhn"
    ] == pytest.approx(0.5)


def test_local_similarity_zero_common_neighbors():
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
            "source": [
                "0"
            ],
            "target": [
                "3"
            ],
        }
    )

    scores = (
        score_local_similarity_candidates(
            graph,
            candidates,
        ).iloc[0]
    )

    assert (
        scores == 0.0
    ).all()