from link_prediction.config import (
    load_methods_config,
    resolve_analysis_family_map,
    resolve_method_complexity_map,
)


def test_method_registry():
    config = load_methods_config()

    families = config["families"]
    methods = config["methods"]

    assert len(families) == 6
    assert len(methods) == 27

    for method in methods.values():
        assert method["family"] in families
        assert method["enabled"] is True


def test_analysis_family_registry():
    config = load_methods_config()

    mapping = (
        resolve_analysis_family_map(
            config
        )
    )

    assert len(
        config[
            "analysis_families"
        ]
    ) == 5

    assert mapping == {
        "local_similarity":
            "standard_local",
        "degree_based":
            "standard_local",
        "local_bayesian":
            "local_bayesian",
        "enhanced_local":
            "enhanced_local",
        "quasi_local_path":
            "quasi_local_path",
        "quasi_local_walk":
            "quasi_local_walk",
    }

    methods = config["methods"]

    analysis_family_sizes = {
        analysis_family: sum(
            mapping[
                method[
                    "family"
                ]
            ]
            == analysis_family
            for method
            in methods.values()
            if method.get(
                "enabled",
                True,
            )
        )
        for analysis_family
        in config[
            "analysis_families"
        ]
    }

    assert min(
        analysis_family_sizes.values()
    ) >= 2


def test_quasi_local_primary_parameters():
    config = load_methods_config()

    methods = config["methods"]

    assert methods["lpi"]["parameters"] == {
        "beta": 0.001,
        "length": 3,
    }

    assert methods[
        "ora_cni"
    ][
        "parameters"
    ] == {
        "beta": 0.001,
    }

    assert methods["fl"]["parameters"] == {
        "length": 3,
    }

    assert methods["lit"]["parameters"] == {
        "iterations": 2,
    }

    assert methods["lrw"]["parameters"]["steps"] == 3
    assert methods["srw"]["parameters"]["steps"] == 3
    assert methods["pfp"]["parameters"]["steps"] == 3


def test_method_complexity_registry():
    config = load_methods_config()

    mapping = (
        resolve_method_complexity_map(
            config
        )
    )

    enabled_method_ids = {
        method_id
        for method_id, method
        in config[
            "methods"
        ].items()
        if method.get(
            "enabled",
            True,
        )
    }

    assert set(mapping) == (
        enabled_method_ids
    )

    assert len(mapping) == 27

    assert mapping["pa"] == (
        "degree_pair"
    )

    assert mapping["cn"] == (
        "local_neighborhood"
    )

    assert mapping["ora_cni"] == (
        "third_order_path"
    )

    assert mapping["pfp"] == (
        "propflow"
    )