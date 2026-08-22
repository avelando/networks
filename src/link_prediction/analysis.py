import warnings
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def safe_degree_assortativity(graph: nx.Graph) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = nx.degree_assortativity_coefficient(graph)

    if pd.isna(value):
        return np.nan

    return float(value)


def graph_profile(
    network_name: str,
    domain: str,
    raw_graph: nx.Graph,
    processed_graph: nx.Graph,
) -> dict[str, Any]:
    raw_undirected = nx.Graph(raw_graph)
    raw_undirected.remove_edges_from(
        nx.selfloop_edges(raw_undirected)
    )

    raw_nodes = raw_undirected.number_of_nodes()
    raw_edges = raw_undirected.number_of_edges()

    processed_nodes = processed_graph.number_of_nodes()
    processed_edges = processed_graph.number_of_edges()

    average_degree = (
        2.0 * processed_edges / processed_nodes
        if processed_nodes > 0
        else np.nan
    )

    return {
        "network": network_name,
        "domain": domain,
        "raw_nodes": raw_nodes,
        "raw_edges": raw_edges,
        "processed_nodes": processed_nodes,
        "processed_edges": processed_edges,
        "removed_nodes": raw_nodes - processed_nodes,
        "removed_edges": raw_edges - processed_edges,
        "density": nx.density(processed_graph),
        "average_degree": average_degree,
        "average_clustering": nx.average_clustering(processed_graph),
        "transitivity": nx.transitivity(processed_graph),
        "degree_assortativity": safe_degree_assortativity(
            processed_graph
        ),
        "connected_components": nx.number_connected_components(
            processed_graph
        ),
    }


def build_network_profile_table(
    profiles: list[dict[str, Any]],
) -> pd.DataFrame:
    dataframe = pd.DataFrame(profiles)

    if dataframe.empty:
        return dataframe

    return dataframe.sort_values(
        ["domain", "network"]
    ).reset_index(drop=True)