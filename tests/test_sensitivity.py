from link_prediction.config import (
    load_methods_config,
)
from link_prediction.sensitivity import (
    build_parameter_sensitivity_plan,
    sensitivity_method_name,
)


def test_parameter_sensitivity_plan():
    plan = (
        build_parameter_sensitivity_plan(
            load_methods_config()
        )
    )

    assert len(plan) == 12

    assert set(
        plan["method_id"]
    ) == {
        "lpi",
        "lrw",
        "srw",
        "pfp",
    }

    lpi = plan[
        plan["method_id"]
        == "lpi"
    ]

    assert set(
        lpi["parameter_value"]
    ) == {
        0.1,
        0.01,
        0.001,
    }

    walks = plan[
        plan["method_id"]
        != "lpi"
    ]

    assert set(
        walks["parameter_value"]
    ) == {
        3,
        5,
        7,
    }

    assert int(
        plan[
            "is_primary"
        ].sum()
    ) == 4


def test_parameter_sensitivity_names():
    assert sensitivity_method_name(
        "lpi",
        "beta",
        0.001,
    ) == "LPI-beta-0.001"

    assert sensitivity_method_name(
        "lpi",
        "beta",
        0.01,
    ) == "LPI-beta-0.01"

    assert sensitivity_method_name(
        "lpi",
        "beta",
        0.1,
    ) == "LPI-beta-0.1"

    assert sensitivity_method_name(
        "lrw",
        "steps",
        5,
    ) == "LRW-l5"

    assert sensitivity_method_name(
        "srw",
        "steps",
        7,
    ) == "SRW-l7"

    assert sensitivity_method_name(
        "pfp",
        "steps",
        3,
    ) == "PFP-l3"


def test_parameter_sensitivity_plan_uses_configuration_specific_names():
    plan = build_parameter_sensitivity_plan(
        load_methods_config()
    )

    observed = set(
        zip(
            plan["method_id"],
            plan["parameter_value"],
            plan["method"],
            strict=True,
        )
    )

    expected = {
        ("lpi", 0.001, "LPI-beta-0.001"),
        ("lpi", 0.01, "LPI-beta-0.01"),
        ("lpi", 0.1, "LPI-beta-0.1"),
        ("lrw", 3, "LRW-l3"),
        ("lrw", 5, "LRW-l5"),
        ("lrw", 7, "LRW-l7"),
        ("srw", 3, "SRW-l3"),
        ("srw", 5, "SRW-l5"),
        ("srw", 7, "SRW-l7"),
        ("pfp", 3, "PFP-l3"),
        ("pfp", 5, "PFP-l5"),
        ("pfp", 7, "PFP-l7"),
    }

    assert observed == expected