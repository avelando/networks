from link_prediction.config import load_methods_config


def test_method_registry():
    config = load_methods_config()

    families = config["families"]
    methods = config["methods"]

    assert len(families) == 6
    assert len(methods) == 20

    assert methods["ia1"] == {
        "name": "IA1",
        "family": "enhanced_local",
        "enabled": True,
        "parameters": {},
    }

    for method in methods.values():
        assert method["family"] in families
        assert method["enabled"] is True


def test_quasi_local_primary_parameters():
    config = load_methods_config()

    methods = config["methods"]

    assert methods["lpi"]["parameters"] == {
        "beta": 0.001,
        "length": 3,
    }

    assert methods["lrw"]["parameters"]["steps"] == 3
    assert methods["srw"]["parameters"]["steps"] == 3
    assert methods["pfp"]["parameters"]["steps"] == 3