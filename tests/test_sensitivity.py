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