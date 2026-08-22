from pathlib import Path

import pandas as pd

from link_prediction.datasets import (
    find_edge_columns,
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