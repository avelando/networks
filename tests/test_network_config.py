from collections import Counter

from link_prediction.config import (
    load_networks_config,
)

EXPECTED_NETWORKS = {
    "ego_facebook",
    "socfb_middlebury45",
    "ca_grqc",
    "ca_hepth",
    "email_eu_core",
    "email_univ",
    "power_grid",
    "power_1138_bus",
}


def test_standard_benchmark_is_balanced():
    config = load_networks_config()

    benchmark = config[
        "benchmarks"
    ][
        "standard"
    ]

    networks = config[
        "networks"
    ]

    assert len(benchmark) == 8
    assert len(set(benchmark)) == 8

    domains = Counter(
        networks[
            network_id
        ][
            "domain"
        ]
        for network_id
        in benchmark
    )

    assert domains == {
        "social": 2,
        "scientific_collaboration": 2,
        "communication": 2,
        "infrastructure": 2,
    }


def test_standard_benchmark_contains_expected_networks():
    config = load_networks_config()

    benchmark = set(
        config[
            "benchmarks"
        ][
            "standard"
        ]
    )

    assert benchmark == (
        EXPECTED_NETWORKS
    )


def test_standard_networks_are_enabled():
    config = load_networks_config()

    benchmark = config[
        "benchmarks"
    ][
        "standard"
    ]

    networks = config[
        "networks"
    ]

    assert all(
        networks[
            network_id
        ].get(
            "enabled",
            True,
        )
        for network_id
        in benchmark
    )