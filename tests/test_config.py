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
    assert (
        config["evaluation"]["primary_metric"]
        == "average_precision"
    )

    assert (
        config["evaluation"]["cutoff"]["strategy"]
        == "positives_per_fold"
    )

    assert set(
        config["evaluation"]["cutoff_metrics"]
    ) == {
        "precision",
        "recall",
        "f1",
        "ndcg",
    }

    assert (
        config["evaluation"]["ranking"]["tie_breaker"]
        == "candidate_id"
    )

    assert (
        config["evaluation"][
            "average_precision"
        ][
            "implementation"
        ]
        == "sklearn"
    )

    assert (
        config["evaluation"][
            "average_precision"
        ][
            "definition"
        ]
        == "non_interpolated_precision_recall"
    )

    assert config[
        "statistics"
    ][
        "alpha"
    ] == 0.05

    assert config[
        "statistics"
    ][
        "metrics"
    ] == [
        "average_precision",
        "roc_auc",
    ]

    assert config[
        "statistics"
    ][
        "confirmatory_methods"
    ] == [
        "srw",
        "lpi",
        "pfp",
    ]