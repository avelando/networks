import hashlib
import random
from collections.abc import Hashable, Iterable

import networkx as nx

Node = Hashable
Edge = tuple[Node, Node]


def stable_node_key(node: Node) -> tuple[str, str]:
    return type(node).__name__, repr(node)


def canonical_edge(source: Node, target: Node) -> Edge:
    if stable_node_key(source) <= stable_node_key(target):
        return source, target

    return target, source


def edge_sort_key(
    edge: Edge,
) -> tuple[tuple[str, str], tuple[str, str]]:
    source, target = canonical_edge(*edge)

    return stable_node_key(source), stable_node_key(target)


def derive_seed(
    base_seed: int,
    *parts: object,
) -> int:
    payload = "::".join(
        [
            str(base_seed),
            *(str(part) for part in parts),
        ]
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
    ) % (2**32)


def graph_edge_set(
    graph: nx.Graph,
) -> set[Edge]:
    return {
        canonical_edge(source, target)
        for source, target in graph.edges()
        if source != target
    }


def sample_non_edges(
    graph: nx.Graph,
    n_samples: int,
    seed: int,
    exclude: Iterable[Edge] | None = None,
) -> list[Edge]:
    if graph.is_directed():
        raise ValueError(
            "Negative sampling requires an undirected graph."
        )

    if n_samples < 0:
        raise ValueError(
            "n_samples must be non-negative."
        )

    if n_samples == 0:
        return []

    nodes = sorted(
        graph.nodes(),
        key=stable_node_key,
    )

    if len(nodes) < 2:
        raise ValueError(
            "At least two nodes are required for negative sampling."
        )

    existing_edges = graph_edge_set(graph)

    excluded_edges = {
        canonical_edge(source, target)
        for source, target in (exclude or [])
    }

    excluded_non_edges = (
        excluded_edges - existing_edges
    )

    total_pairs = (
        len(nodes) * (len(nodes) - 1) // 2
    )

    available_non_edges = (
        total_pairs
        - len(existing_edges)
        - len(excluded_non_edges)
    )

    if n_samples > available_non_edges:
        raise ValueError(
            f"Requested {n_samples} negative pairs, "
            f"but only {available_non_edges} are available."
        )

    rng = random.Random(seed)

    sampled: set[Edge] = set()

    max_attempts = max(
        10_000,
        n_samples * 50,
    )

    attempts = 0

    while (
        len(sampled) < n_samples
        and attempts < max_attempts
    ):
        source_index = rng.randrange(
            len(nodes)
        )

        target_index = rng.randrange(
            len(nodes) - 1
        )

        if target_index >= source_index:
            target_index += 1

        edge = canonical_edge(
            nodes[source_index],
            nodes[target_index],
        )

        attempts += 1

        if edge in existing_edges:
            continue

        if edge in excluded_edges:
            continue

        sampled.add(edge)

    if len(sampled) != n_samples:
        raise RuntimeError(
            f"Could not sample {n_samples} unique "
            f"non-edges after {attempts} attempts."
        )

    return sorted(
        sampled,
        key=edge_sort_key,
    )