import networkx as nx

from link_prediction.analysis import graph_profile


def test_graph_profile_preserves_raw_counts():
    raw_graph = nx.DiGraph()

    raw_graph.add_edges_from(
        [
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 3),
            (10, 11),
        ]
    )

    processed_graph = nx.Graph()
    processed_graph.add_edges_from(
        [
            (1, 2),
            (2, 3),
        ]
    )

    profile = graph_profile(
        network_name="test-network",
        domain="test",
        raw_graph=raw_graph,
        processed_graph=processed_graph,
    )

    assert profile["raw_directed"] is True
    assert profile["raw_nodes"] == 5
    assert profile["raw_edges"] == 5

    assert profile["raw_undirected_nodes"] == 5
    assert profile["raw_undirected_edges"] == 3

    assert profile["processed_nodes"] == 3
    assert profile["processed_edges"] == 2
    assert profile["processed_components"] == 1