from pathlib import Path

import networkx as nx

from link_prediction.config import PROCESSED_DATA_DIR


def standardize_graph(
    graph: nx.Graph,
    make_undirected: bool = True,
    remove_self_loops: bool = True,
    use_largest_connected_component: bool = True,
) -> nx.Graph:
    if make_undirected:
        standardized = nx.Graph(graph)
    else:
        standardized = graph.copy()

    if remove_self_loops:
        standardized.remove_edges_from(nx.selfloop_edges(standardized))

    isolates = list(nx.isolates(standardized))

    if isolates:
        standardized.remove_nodes_from(isolates)

    if (
        use_largest_connected_component
        and standardized.number_of_nodes() > 0
    ):
        if standardized.is_directed():
            components = nx.weakly_connected_components(standardized)
        else:
            components = nx.connected_components(standardized)

        largest_component = max(components, key=len)

        standardized = standardized.subgraph(
            largest_component
        ).copy()

    return standardized


def save_processed_graph(
    graph: nx.Graph,
    network_name: str,
    output_directory: Path = PROCESSED_DATA_DIR,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / f"{network_name}.edgelist"

    nx.write_edgelist(
        graph,
        output_path,
        data=False,
        encoding="utf-8",
    )

    return output_path


def load_processed_graph(
    network_name: str,
    input_directory: Path = PROCESSED_DATA_DIR,
) -> nx.Graph:
    path = input_directory / f"{network_name}.edgelist"

    if not path.exists():
        raise FileNotFoundError(f"Processed graph not found: {path}")

    return nx.read_edgelist(
        path,
        nodetype=str,
        create_using=nx.Graph(),
        data=False,
    )