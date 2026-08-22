import gzip
import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import requests

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

        if member:
            member_path = extraction_directory / member

            if not member_path.exists():
                raise FileNotFoundError(
                    f"Configured archive member not found: {member_path}"
                )

            return member_path

        return extraction_directory

    raise ValueError(
        f"Unsupported archive type for {network_name}: {archive_type}"
    )


def resolve_nodetype(value: str | None):
    if value == "int":
        return int

    if value == "float":
        return float

    return str


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

    if parser_type == "csv":
        separator = parser_config.get("separator", ",")
        source_column = parser_config.get("source_column")
        target_column = parser_config.get("target_column")

        dataframe = pd.read_csv(path, sep=separator)

        if source_column is None or target_column is None:
            if dataframe.shape[1] < 2:
                raise ValueError(
                    f"CSV requires at least two columns: {path}"
                )

            source_column = dataframe.columns[0]
            target_column = dataframe.columns[1]

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