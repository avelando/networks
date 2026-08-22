import gzip
import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import requests
from scipy.io import mmread
from scipy.sparse import csr_array, issparse

from link_prediction.config import RAW_DATA_DIR


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    overwrite: bool = False,
    expected_sha256: str | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        if expected_sha256 is not None:
            actual_sha256 = compute_sha256(destination)

            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Checksum mismatch for existing file: {destination}"
                )

        return destination

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    temporary_path = destination.with_suffix(destination.suffix + ".part")

    with temporary_path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)

    temporary_path.replace(destination)

    if expected_sha256 is not None:
        actual_sha256 = compute_sha256(destination)

        if actual_sha256 != expected_sha256:
            destination.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch: {destination}")

    return destination


def extract_gzip(
    source: Path,
    destination: Path,
    overwrite: bool = False,
) -> Path:
    if destination.exists() and not overwrite:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(source, "rb") as input_file, destination.open("wb") as output_file:
        shutil.copyfileobj(input_file, output_file)

    return destination


def extract_zip(
    source: Path,
    destination: Path,
    overwrite: bool = False,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)

    if any(destination.iterdir()) and not overwrite:
        return destination

    with zipfile.ZipFile(source, "r") as archive:
        archive.extractall(destination)

    return destination


def resolve_extracted_file(
    directory: Path,
    member: str | None = None,
    member_patterns: list[str] | None = None,
) -> Path:
    if member is not None:
        path = directory / member

        if not path.exists():
            raise FileNotFoundError(
                f"Configured archive member not found: {path}"
            )

        return path

    for pattern in member_patterns or []:
        matches = sorted(
            path
            for path in directory.rglob(pattern)
            if path.is_file()
        )

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            edge_candidates = [
                path
                for path in matches
                if any(
                    keyword in path.name.lower()
                    for keyword in ("edge", "edges", "link", "links")
                )
            ]

            if len(edge_candidates) == 1:
                return edge_candidates[0]

            raise ValueError(
                f"Multiple files match pattern '{pattern}' "
                f"in {directory}: {matches}"
            )

    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
    )

    if len(files) == 1:
        return files[0]

    raise FileNotFoundError(
        f"Could not determine graph file in: {directory}"
    )


def prepare_dataset(
    network_name: str,
    network_config: dict[str, Any],
    overwrite: bool = False,
) -> Path:
    source = network_config.get("source", {})

    url = source.get("url")
    filename = source.get("filename")
    archive_type = source.get("archive")
    extracted_filename = source.get("extracted_filename")
    member = source.get("member")
    member_patterns = source.get("member_patterns", [])
    expected_sha256 = source.get("sha256")

    if not url:
        raise ValueError(f"Missing download URL for network: {network_name}")

    if not filename:
        raise ValueError(f"Missing source filename for network: {network_name}")

    downloaded_path = RAW_DATA_DIR / filename

    download_file(
        url=url,
        destination=downloaded_path,
        overwrite=overwrite,
        expected_sha256=expected_sha256,
    )

    if archive_type is None:
        return downloaded_path

    if archive_type == "gzip":
        if not extracted_filename:
            raise ValueError(
                f"Missing extracted_filename for gzip network: {network_name}"
            )

        extracted_path = RAW_DATA_DIR / extracted_filename

        return extract_gzip(
            source=downloaded_path,
            destination=extracted_path,
            overwrite=overwrite,
        )

    if archive_type == "zip":
        extraction_directory = RAW_DATA_DIR / network_name

        extract_zip(
            source=downloaded_path,
            destination=extraction_directory,
            overwrite=overwrite,
        )

        return resolve_extracted_file(
            directory=extraction_directory,
            member=member,
            member_patterns=member_patterns,
        )

    raise ValueError(
        f"Unsupported archive type for {network_name}: {archive_type}"
    )


def resolve_nodetype(value: str | None):
    if value == "int":
        return int

    if value == "float":
        return float

    return str


def normalize_column_name(column_name: str) -> str:
    return (
        str(column_name)
        .strip()
        .lower()
        .replace("#", "")
        .strip()
    )


def find_edge_columns(
    dataframe: pd.DataFrame,
) -> tuple[str, str]:
    normalized_columns = [
        normalize_column_name(column)
        for column in dataframe.columns
    ]

    possible_column_pairs = [
        ("source", "target"),
        ("src", "dst"),
        ("from", "to"),
        ("node1", "node2"),
        ("vertex1", "vertex2"),
        ("v1", "v2"),
        ("u", "v"),
        ("i", "j"),
    ]

    for source_candidate, target_candidate in possible_column_pairs:
        if (
            source_candidate in normalized_columns
            and target_candidate in normalized_columns
        ):
            source_index = normalized_columns.index(
                source_candidate
            )
            target_index = normalized_columns.index(
                target_candidate
            )

            return (
                dataframe.columns[source_index],
                dataframe.columns[target_index],
            )

    if dataframe.shape[1] < 2:
        raise ValueError(
            "The edge table must contain at least two columns."
        )

    return dataframe.columns[0], dataframe.columns[1]


def load_matrix_market_graph(
    path: Path,
    directed: bool = False,
) -> nx.Graph:
    matrix = mmread(path)

    if issparse(matrix):
        matrix = matrix.tocsr()
    else:
        matrix = csr_array(matrix)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Matrix Market graph must be square: "
            f"{path} has shape {matrix.shape}"
        )

    graph_class = nx.DiGraph if directed else nx.Graph

    weighted_graph = nx.from_scipy_sparse_array(
        matrix,
        create_using=graph_class(),
    )

    graph = graph_class()

    graph.add_nodes_from(
        int(node) + 1
        for node in weighted_graph.nodes()
    )

    graph.add_edges_from(
        (int(source) + 1, int(target) + 1)
        for source, target in weighted_graph.edges()
    )

    return graph


def load_graph(
    path: Path,
    parser_config: dict[str, Any],
) -> nx.Graph:
    parser_type = parser_config["type"]
    directed = parser_config.get("directed", False)

    graph_class = nx.DiGraph if directed else nx.Graph

    if parser_type == "edgelist":
        return nx.read_edgelist(
            path,
            comments=parser_config.get("comments", "#"),
            delimiter=parser_config.get("delimiter"),
            nodetype=resolve_nodetype(parser_config.get("nodetype")),
            create_using=graph_class(),
            data=False,
        )

    if parser_type == "matrix_market":
        return load_matrix_market_graph(
            path=path,
            directed=directed,
        )

    if parser_type == "csv":
        separator = parser_config.get("separator", "auto")

        if separator == "auto" or separator is None:
            dataframe = pd.read_csv(
                path,
                sep=None,
                engine="python",
            )
        else:
            dataframe = pd.read_csv(
                path,
                sep=separator,
            )

        source_column = parser_config.get("source_column")
        target_column = parser_config.get("target_column")

        if source_column is None or target_column is None:
            source_column, target_column = find_edge_columns(
                dataframe
            )

        return nx.from_pandas_edgelist(
            dataframe,
            source=source_column,
            target=target_column,
            create_using=graph_class(),
        )
    
    if parser_type == "pajek":
        return nx.read_pajek(path)

    if parser_type == "gml":
        return nx.read_gml(path)

    if parser_type == "graphml":
        return nx.read_graphml(path)

    raise ValueError(f"Unsupported parser type: {parser_type}")