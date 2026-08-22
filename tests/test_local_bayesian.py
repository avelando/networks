import math

import networkx as nx
import pandas as pd
import pytest

from link_prediction.methods.local_bayesian import (
    LOCAL_BAYESIAN_METHODS,
    compute_lnb_roles,
    score_local_bayesian_candidates,
)


def build_lnb_test_graph() -> nx.Graph:
    graph = nx.Graph()

    graph.add_edges_from(
        [
            ("0", "2"),
            ("1", "2"),
            ("2", "3"),
            ("3", "4"),
            ("4", "5"),
        ]
    )

    return graph


def test_lnb_roles():
    graph = build_lnb_test_graph()

    roles = compute_lnb_roles(
        graph
    )

    assert roles["2"] == pytest.approx(
        0.25
    )

    assert roles["0"] == pytest.approx(
        1.0
    )

    assert roles["3"] == pytest.approx(
        0.5
    )


def test_lnb_ra_score():
    graph = build_lnb_test_graph()

    candidates = pd.DataFrame(
        {
            "source": [
                "0",
                "0",
            ],
            "target": [
                "1",
                "5",
            ],
        }
    )

    scores = (
        score_local_bayesian_candidates(
            graph,
            candidates,
        )
    )

    assert tuple(
        scores.columns
    ) == LOCAL_BAYESIAN_METHODS

    possible_edges = (
        6 * 5 / 2
    )

    prior_odds = (
        possible_edges
        / 5
        - 1
    )

    expected = (
        math.log(
            prior_odds
            * 0.25
        )
        / 3
    )

    assert scores.loc[
        0,
        "lnb_ra",
    ] == pytest.approx(
        expected
    )

    assert scores.loc[
        1,
        "lnb_ra",
    ] == pytest.approx(
        0.0
    )


def test_lnb_ra_rejects_missing_node():
    graph = build_lnb_test_graph()

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
        score_local_bayesian_candidates(
            graph,
            candidates,
        )