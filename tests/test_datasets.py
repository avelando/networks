from pathlib import Path

import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import coo_matrix

from link_prediction.datasets import (
    find_edge_columns,
    load_graph,
    resolve_extracted_file,
)


def test_find_edge_columns():
    dataframe = pd.DataFrame(
        {
            "# source": [1, 2],
            "target": [2, 3],
            "weight": [1.0, 1.0],
        }
    )

    source, target = find_edge_columns(dataframe)

    assert source == "# source"
    assert target == "target"


def test_resolve_extracted_file(tmp_path: Path):
    edge_file = tmp_path / "network.edges"
    metadata_file = tmp_path / "README.txt"

    edge_file.write_text(
        "1 2\n2 3\n",
        encoding="utf-8",
    )

    metadata_file.write_text(
        "metadata",
        encoding="utf-8",
    )

    resolved = resolve_extracted_file(
        directory=tmp_path,
        member_patterns=["*.edges"],
    )

    assert resolved == edge_file


def test_load_matrix_market_graph(tmp_path: Path):
    path = tmp_path / "network.mtx"

    matrix = coo_matrix(
        (
            [1, 1, 1, 1],
            (
                [0, 1, 1, 2],
                [1, 0, 2, 1],
            ),
        ),
        shape=(3, 3),
    )

    mmwrite(path, matrix)

    graph = load_graph(
        path=path,
        parser_config={
            "type": "matrix_market",
            "directed": False,
        },
    )

    assert set(graph.nodes()) == {1, 2, 3}

    edges = {
        frozenset(edge)
        for edge in graph.edges()
    }

    assert edges == {
        frozenset((1, 2)),
        frozenset((2, 3)),
    }


def test_resolve_matrix_market_file(tmp_path: Path):
    matrix_file = tmp_path / "network.mtx"
    readme_file = tmp_path / "README.txt"

    matrix_file.write_text(
        "%%MatrixMarket matrix coordinate pattern symmetric\n"
        "3 3 2\n"
        "1 2\n"
        "2 3\n",
        encoding="utf-8",
    )

    readme_file.write_text(
        "metadata",
        encoding="utf-8",
    )

    resolved = resolve_extracted_file(
        directory=tmp_path,
        member_patterns=[
            "*.mtx",
            "*.edges",
        ],
    )

    assert resolved == matrix_file