import networkx as nx

from link_prediction.preprocessing import standardize_graph


def test_standardize_graph():
    graph = nx.DiGraph()

    graph.add_edges_from(
        [
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 3),
            (10, 11),
        ]
    )

    processed = standardize_graph(
        graph,
        make_undirected=True,
        remove_self_loops=True,
        use_largest_connected_component=True,
    )

    assert isinstance(processed, nx.Graph)
    assert not processed.is_directed()
    assert nx.number_of_selfloops(processed) == 0
    assert nx.is_connected(processed)

    assert set(processed.nodes()) == {1, 2, 3}

    edges = {
        frozenset(edge)
        for edge in processed.edges()
    }

    expected_edges = {
        frozenset((1, 2)),
        frozenset((2, 3)),
    }

    assert edges == expected_edges