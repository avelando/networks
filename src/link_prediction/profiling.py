from typing import Any

import networkx as nx
import pandas as pd

from link_prediction.analysis import (
    build_network_profile_table,
    graph_profile,
)
from link_prediction.config import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SUMMARY_RESULTS_DIR,
    ensure_project_directories,
    load_experiment_config,
    load_networks_config,
)
from link_prediction.datasets import (
    compute_sha256,
    load_graph,
    prepare_dataset,
)
from link_prediction.execution import (
    run_process_tasks,
)
from link_prediction.preprocessing import (
    save_processed_graph,
    standardize_graph,
)


def project_relative_path(path) -> str:
    return path.resolve().relative_to(
        PROJECT_ROOT.resolve()
    ).as_posix()


def prepare_network(
    network_id: str,
    network_config: dict[str, Any],
    graph_config: dict[str, Any],
    overwrite: bool = False,
) -> tuple[nx.Graph, dict[str, Any], dict[str, Any]]:
    graph_path = prepare_dataset(
        network_name=network_id,
        network_config=network_config,
        overwrite=overwrite,
    )

    raw_graph = load_graph(
        path=graph_path,
        parser_config=network_config["parser"],
    )

    processed_graph = standardize_graph(
        graph=raw_graph,
        make_undirected=not graph_config.get(
            "directed",
            False,
        ),
        remove_self_loops=graph_config.get(
            "remove_self_loops",
            True,
        ),
        use_largest_connected_component=graph_config.get(
            "use_largest_connected_component",
            True,
        ),
    )

    processed_path = save_processed_graph(
        graph=processed_graph,
        network_name=network_config["name"],
    )

    profile = graph_profile(
        network_name=network_config["name"],
        domain=network_config["domain"],
        raw_graph=raw_graph,
        processed_graph=processed_graph,
    )

    profile.update(
        {
            "network_id": network_id,
            "role": network_config["role"],
            "repository": network_config["repository"],
        }
    )

    source = network_config["source"]
    downloaded_path = RAW_DATA_DIR / source["filename"]

    manifest = {
        "network_id": network_id,
        "network": network_config["name"],
        "domain": network_config["domain"],
        "role": network_config["role"],
        "repository": network_config["repository"],
        "source_url": source["url"],
        "source_file": source["filename"],
        "source_sha256": compute_sha256(downloaded_path),
        "graph_file": project_relative_path(graph_path),
        "processed_file": project_relative_path(processed_path),
    }

    return processed_graph, profile, manifest


def prepare_network_summary(
    network_id: str,
    network_config: dict[str, Any],
    graph_config: dict[str, Any],
    overwrite: bool,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    _, profile, manifest = (
        prepare_network(
            network_id=network_id,
            network_config=network_config,
            graph_config=graph_config,
            overwrite=overwrite,
        )
    )

    return (
        profile,
        manifest,
    )


def prepare_benchmark(
    benchmark_name: str = "revision",
    overwrite: bool = False,
    max_workers: int | str | None = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_project_directories()

    networks_config = load_networks_config()
    experiment_config = load_experiment_config()

    benchmark = networks_config["benchmarks"][benchmark_name]
    network_definitions = networks_config["networks"]
    graph_config = experiment_config["graph"]

    tasks = []

    for network_id in benchmark:
        network_config = (
            network_definitions[
                network_id
            ]
        )

        if not network_config.get(
            "enabled",
            True,
        ):
            continue

        tasks.append(
            (
                network_id,
                network_config,
                graph_config,
                overwrite,
            )
        )

    prepared = run_process_tasks(
        prepare_network_summary,
        tasks,
        max_workers=max_workers,
        profile="profiling",
        label="network profiling",
    )

    profiles = [
        profile
        for profile, _
        in prepared
    ]

    manifests = [
        manifest
        for _, manifest
        in prepared
    ]

    profile_table = build_network_profile_table(profiles)

    manifest_table = pd.DataFrame(manifests).sort_values(
        ["domain", "network"]
    ).reset_index(drop=True)

    SUMMARY_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile_path = (
        SUMMARY_RESULTS_DIR
        / f"{benchmark_name}_network_profiles.csv"
    )

    manifest_path = (
        SUMMARY_RESULTS_DIR
        / f"{benchmark_name}_dataset_manifest.csv"
    )

    profile_table.to_csv(
        profile_path,
        index=False,
    )

    manifest_table.to_csv(
        manifest_path,
        index=False,
    )

    return profile_table, manifest_table