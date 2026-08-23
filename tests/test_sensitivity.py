from link_prediction.config import (
    load_methods_config,
)
from link_prediction.sensitivity import (
    build_parameter_sensitivity_plan,
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