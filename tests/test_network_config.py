from collections import Counter

from link_prediction.config import load_networks_config


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


def test_original_networks_are_preserved():
    config = load_networks_config()

    original = set(config["benchmarks"]["original"])
    revision = set(config["benchmarks"]["revision"])

    assert original <= revision
    assert len(original) == 5


def test_only_three_networks_are_added():
    config = load_networks_config()

    original = set(config["benchmarks"]["original"])
    revision = set(config["benchmarks"]["revision"])

    added = revision - original

    assert added == {
        "socfb_middlebury45",
        "email_univ",
        "power_1138_bus",
    }