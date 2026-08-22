import networkx as nx

from link_prediction.sampling import (
    sample_non_edges,
)


def test_sample_non_edges_is_deterministic_and_valid():
    graph = nx.cycle_graph(8)

    first = sample_non_edges(
        graph=graph,
        n_samples=6,
        seed=42,
    )

    second = sample_non_edges(
        graph=graph,
        n_samples=6,
        seed=42,
    )

    assert first == second
    assert len(first) == 6
    assert len(set(first)) == 6

    for source, target in first:
        assert source != target
        assert not graph.has_edge(
            source,
            target,
        )


def test_sample_non_edges_respects_exclusion():
    graph = nx.path_graph(6)

    excluded = sample_non_edges(
        graph=graph,
        n_samples=3,
        seed=10,
    )

    sampled = sample_non_edges(
        graph=graph,
        n_samples=3,
        seed=11,
        exclude=excluded,
    )

    assert set(sampled).isdisjoint(
        excluded
    )