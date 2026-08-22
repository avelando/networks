from collections import Counter

from link_prediction.config import load_networks_config


BASELINE_NETWORKS = {
    "ego_facebook",
    "ca_grqc",
    "ca_hepth",
    "email_eu_core",
    "power_grid",
}

ADDITIONAL_NETWORKS = {
    "socfb_middlebury45",
    "email_univ",
    "power_1138_bus",
}


def test_revision_benchmark_is_balanced():
    config = load_networks_config()

    revision = config["benchmarks"]["revision"]
    networks = config["networks"]

    assert len(revision) == 8
    assert len(set(revision)) == 8

    domains = Counter(
        networks[network_id]["domain"]
        for network_id in revision
    )

    assert domains == {
        "social": 2,
        "scientific_collaboration": 2,
        "communication": 2,
        "infrastructure": 2,
    }


def test_revision_preserves_published_networks():
    config = load_networks_config()

    revision = set(config["benchmarks"]["revision"])

    assert BASELINE_NETWORKS <= revision

    for network_id in BASELINE_NETWORKS:
        assert config["networks"][network_id]["role"] == "original"


def test_revision_adds_only_selected_networks():
    config = load_networks_config()

    revision = set(config["benchmarks"]["revision"])

    assert revision - BASELINE_NETWORKS == ADDITIONAL_NETWORKS

    for network_id in ADDITIONAL_NETWORKS:
        assert config["networks"][network_id]["role"] == "additional"