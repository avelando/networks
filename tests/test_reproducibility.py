from link_prediction.reproducibility import collect_environment_metadata


def test_collect_environment_metadata():
    metadata = collect_environment_metadata()

    assert "python_version" in metadata
    assert "operating_system" in metadata
    assert "logical_cpus" in metadata
    assert "git_commit" in metadata
    assert "packages" in metadata
    assert "git_dirty" in metadata

    assert metadata["python_version"]
    assert metadata["logical_cpus"] is not None