import networkx as nx

from link_prediction.folds import (
    build_candidate_table,
    build_fixed_spanning_tree,
    build_training_graph,
    split_removable_edges,
)
from link_prediction.sampling import (
    graph_edge_set,
    sample_non_edges,
)


def build_test_graph() -> nx.Graph:
    graph = nx.cycle_graph(10)

    graph.add_edges_from(
        [
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 6),
            (5, 7),
            (6, 8),
            (7, 9),
            (8, 0),
            (9, 1),
        ]
    )

    return graph


def test_fixed_spanning_tree_is_deterministic():
    graph = build_test_graph()

    first = build_fixed_spanning_tree(
        graph,
        seed=42,
    )

    second = build_fixed_spanning_tree(
        graph,
        seed=42,
    )

    assert first == second
    assert len(first) == graph.number_of_nodes() - 1
    assert set(first) <= graph_edge_set(graph)

    tree = nx.Graph()
    tree.add_nodes_from(graph.nodes())
    tree.add_edges_from(first)

    assert nx.is_tree(tree)


def test_positive_folds_partition_all_removable_edges():
    graph = build_test_graph()

    spanning_tree = build_fixed_spanning_tree(
        graph,
        seed=42,
    )

    folds = split_removable_edges(
        graph=graph,
        spanning_tree_edges=spanning_tree,
        n_folds=5,
        seed=123,
    )

    flattened = [
        edge
        for fold in folds
        for edge in fold
    ]

    removable = (
        graph_edge_set(graph)
        - set(spanning_tree)
    )

    assert len(folds) == 5
    assert set(flattened) == removable
    assert len(flattened) == len(set(flattened))
    assert max(map(len, folds)) - min(map(len, folds)) <= 1


def test_training_graph_keeps_connectivity():
    graph = build_test_graph()

    spanning_tree = build_fixed_spanning_tree(
        graph,
        seed=42,
    )

    folds = split_removable_edges(
        graph=graph,
        spanning_tree_edges=spanning_tree,
        n_folds=5,
        seed=123,
    )

    for positive_edges in folds:
        training_graph = build_training_graph(
            graph,
            positive_edges,
        )

        assert nx.is_connected(training_graph)

        assert (
            training_graph.number_of_edges()
            == graph.number_of_edges()
            - len(positive_edges)
        )

        assert (
            set(spanning_tree)
            <= graph_edge_set(training_graph)
        )


def test_candidate_table_is_balanced_and_deterministic():
    graph = build_test_graph()

    spanning_tree = build_fixed_spanning_tree(
        graph,
        seed=42,
    )

    positive_edges = split_removable_edges(
        graph=graph,
        spanning_tree_edges=spanning_tree,
        n_folds=5,
        seed=123,
    )[0]

    negative_edges = sample_non_edges(
        graph=graph,
        n_samples=len(positive_edges),
        seed=999,
    )

    first = build_candidate_table(
        positive_edges=positive_edges,
        negative_edges=negative_edges,
        seed=777,
    )

    second = build_candidate_table(
        positive_edges=positive_edges,
        negative_edges=negative_edges,
        seed=777,
    )

    assert first.equals(second)

    assert (
        len(first)
        == 2 * len(positive_edges)
    )

    assert (
        first["label"].value_counts().to_dict()
        == {
            1: len(positive_edges),
            0: len(positive_edges),
        }
    )

    assert (
        first["candidate_id"].tolist()
        == list(range(len(first)))
    )