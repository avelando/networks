from link_prediction.config import PROJECT_ROOT
from link_prediction.profiling import project_relative_path


def test_project_relative_path():
    path = PROJECT_ROOT / "data" / "raw" / "network.txt"

    assert project_relative_path(path) == "data/raw/network.txt"