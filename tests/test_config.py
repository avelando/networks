from link_prediction.config import (
    PROJECT_ROOT,
    load_experiment_config,
)


def test_project_root_exists():
    assert PROJECT_ROOT.exists()


def test_experiment_configuration():
    config = load_experiment_config()

    assert config["experiment"]["random_seed"] == 42
    assert config["experiment"]["n_folds"] == 5
    assert config["negative_sampling"]["primary_ratio"] == 1