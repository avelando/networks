import networkx as nx
import pandas as pd
import pytest

from link_prediction.config import (
    load_methods_config,
)
from link_prediction.methods.enhanced_local import (
    local_interacting_scores,
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

    assert len(plan) == 18

    assert set(
        plan["method_id"]
    ) == {
        "lit",
        "lpi",
        "ora_cni",
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

    ora_cni = plan[
        plan["method_id"]
        == "ora_cni"
    ]

    assert set(
        ora_cni[
            "parameter_value"
        ]
    ) == {
        0.1,
        0.01,
        0.001,
    }

    lit = plan[
        plan["method_id"]
        == "lit"
    ]

    assert set(
        lit["parameter_value"]
    ) == {
        2,
        4,
        6,
    }

    walks = plan[
        plan["method_id"].isin(
            [
                "lrw",
                "srw",
                "pfp",
            ]
        )
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
    ) == 6


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
        "ora_cni",
        "beta",
        0.001,
    ) == "ORA-CNI-beta-0.001"

    assert sensitivity_method_name(
        "ora_cni",
        "beta",
        0.01,
    ) == "ORA-CNI-beta-0.01"

    assert sensitivity_method_name(
        "ora_cni",
        "beta",
        0.1,
    ) == "ORA-CNI-beta-0.1"

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

    assert sensitivity_method_name(
        "lit",
        "iterations",
        2,
    ) == "LIT-i2"

    assert sensitivity_method_name(
        "lit",
        "iterations",
        4,
    ) == "LIT-i4"

    assert sensitivity_method_name(
        "lit",
        "iterations",
        6,
    ) == "LIT-i6"


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
        ("lit", 2, "LIT-i2"),
        ("lit", 4, "LIT-i4"),
        ("lit", 6, "LIT-i6"),
        ("lpi", 0.001, "LPI-beta-0.001"),
        ("lpi", 0.01, "LPI-beta-0.01"),
        ("lpi", 0.1, "LPI-beta-0.1"),
        (
            "ora_cni",
            0.001,
            "ORA-CNI-beta-0.001",
        ),
        (
            "ora_cni",
            0.01,
            "ORA-CNI-beta-0.01",
        ),
        (
            "ora_cni",
            0.1,
            "ORA-CNI-beta-0.1",
        ),
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


def test_lit_sensitivity_scores_are_distinct():
    graph = nx.Graph()

    graph.add_edges_from(
        [
            ("x", "a"),
            ("x", "b"),
            ("y", "a"),
            ("y", "b"),
            ("a", "b"),
        ]
    )

    candidates = pd.DataFrame(
        {
            "candidate_id": [0],
            "source": ["x"],
            "target": ["y"],
            "label": [1],
        }
    )

    observed = {
        iterations:
            local_interacting_scores(
                graph=graph,
                candidates=candidates,
                iterations=iterations,
            )[0]
        for iterations
        in (
            2,
            4,
            6,
        )
    }

    assert observed[2] == pytest.approx(
        24.0 / 35.0
    )

    assert len(
        {
            round(
                score,
                12,
            )
            for score
            in observed.values()
        }
    ) == 3