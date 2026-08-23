import networkx as nx
import pandas as pd

from link_prediction.config import (
    load_methods_config,
)
from link_prediction.robustness import (
    build_nested_robustness_candidates,
    normalize_negative_ratios,
    score_all_primary_methods,
)
from link_prediction.sampling import (
    canonical_edge,
)


def pair_to_candidate_id(
    table: pd.DataFrame,
) -> dict[
    tuple[
        object,
        object,
    ],
    int,
]:
    return {
        canonical_edge(
            source,
            target,
        ): int(
            candidate_id
        )
        for (
            candidate_id,
            source,
            target,
        )
        in table[
            [
                "candidate_id",
                "source",
                "target",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    }


def test_negative_ratios():
    assert (
        normalize_negative_ratios(
            [
                10,
                1,
                5,
                5,
            ],
            primary_ratio=1,
        )
        == [
            1,
            5,
            10,
        ]
    )


def test_robustness_candidates_are_nested_and_valid():
    graph = (
        nx.cycle_graph(
            [
                str(
                    index
                )
                for index
                in range(
                    8
                )
            ]
        )
    )

    primary = pd.DataFrame(
        {
            "candidate_id":
                [
                    0,
                    1,
                ],
            "source":
                [
                    "0",
                    "0",
                ],
            "target":
                [
                    "1",
                    "2",
                ],
            "label":
                [
                    1,
                    0,
                ],
        }
    )

    tables = (
        build_nested_robustness_candidates(
            full_graph=
                graph,
            primary_candidates=
                primary,
            ratios=[
                1,
                5,
                10,
            ],
            primary_ratio=
                1,
            random_seed=
                42,
            network_id=
                "toy",
            fold_number=
                1,
        )
    )

    negative_sets = {}
    positive_sets = {}
    candidate_ids = {}

    for ratio in (
        1,
        5,
        10,
    ):
        table = tables[
            ratio
        ]

        assert int(
            table[
                "label"
            ]
            .eq(
                1
            )
            .sum()
        ) == 1

        assert int(
            table[
                "label"
            ]
            .eq(
                0
            )
            .sum()
        ) == ratio

        positives = table[
            table[
                "label"
            ]
            == 1
        ]

        negatives = table[
            table[
                "label"
            ]
            == 0
        ]

        positive_sets[
            ratio
        ] = {
            canonical_edge(
                source,
                target,
            )
            for source, target
            in positives[
                [
                    "source",
                    "target",
                ]
            ].itertuples(
                index=False,
                name=None,
            )
        }

        negative_sets[
            ratio
        ] = {
            canonical_edge(
                source,
                target,
            )
            for source, target
            in negatives[
                [
                    "source",
                    "target",
                ]
            ].itertuples(
                index=False,
                name=None,
            )
        }

        candidate_ids[
            ratio
        ] = (
            pair_to_candidate_id(
                table
            )
        )

        for edge in (
            negative_sets[
                ratio
            ]
        ):
            assert not (
                graph.has_edge(
                    *edge
                )
            )

    assert (
        positive_sets[1]
        == positive_sets[5]
        == positive_sets[10]
    )

    assert (
        negative_sets[1]
        <= negative_sets[5]
        <= negative_sets[10]
    )

    assert (
        positive_sets[1]
        == {
            canonical_edge(
                "0",
                "1",
            )
        }
    )

    assert (
        negative_sets[1]
        == {
            canonical_edge(
                "0",
                "2",
            )
        }
    )

    shared_pairs = (
        positive_sets[1]
        | negative_sets[1]
    )

    for pair in (
        shared_pairs
    ):
        assert (
            candidate_ids[
                1
            ][
                pair
            ]
            == candidate_ids[
                5
            ][
                pair
            ]
        )

        assert (
            candidate_ids[
                1
            ][
                pair
            ]
            == candidate_ids[
                10
            ][
                pair
            ]
        )


def test_primary_ratio_preserves_original_tie_order():
    graph = (
        nx.cycle_graph(
            [
                str(
                    index
                )
                for index
                in range(
                    8
                )
            ]
        )
    )

    primary = pd.DataFrame(
        {
            "candidate_id":
                [
                    0,
                    1,
                    2,
                    3,
                ],
            "source":
                [
                    "0",
                    "0",
                    "1",
                    "1",
                ],
            "target":
                [
                    "1",
                    "2",
                    "2",
                    "3",
                ],
            "label":
                [
                    1,
                    0,
                    1,
                    0,
                ],
        }
    )

    tables = (
        build_nested_robustness_candidates(
            full_graph=
                graph,
            primary_candidates=
                primary,
            ratios=[
                1,
                5,
                10,
            ],
            primary_ratio=
                1,
            random_seed=
                42,
            network_id=
                "toy",
            fold_number=
                1,
        )
    )

    observed = (
        tables[
            1
        ]
        .sort_values(
            "candidate_id"
        )[
            [
                "source",
                "target",
                "label",
            ]
        ]
        .reset_index(
            drop=True
        )
    )

    expected = (
        primary
        .sort_values(
            "candidate_id"
        )[
            [
                "source",
                "target",
                "label",
            ]
        ]
        .reset_index(
            drop=True
        )
    )

    assert observed.equals(
        expected
    )


def test_score_all_primary_methods_returns_registry_columns():
    graph = (
        nx.cycle_graph(
            [
                str(
                    index
                )
                for index
                in range(
                    6
                )
            ]
        )
    )

    candidates = (
        pd.DataFrame(
            {
                "candidate_id":
                    [
                        0,
                        1,
                        2,
                    ],
                "source":
                    [
                        "0",
                        "1",
                        "2",
                    ],
                "target":
                    [
                        "2",
                        "3",
                        "4",
                    ],
                "label":
                    [
                        1,
                        0,
                        0,
                    ],
            }
        )
    )

    methods_config = (
        load_methods_config()
    )

    scores = (
        score_all_primary_methods(
            graph=
                graph,
            candidates=
                candidates,
            methods_config=
                methods_config,
        )
    )

    expected = [
        method_id
        for (
            method_id,
            method_config,
        )
        in methods_config[
            "methods"
        ].items()
        if method_config.get(
            "enabled",
            True,
        )
    ]

    assert list(
        scores.columns
    ) == expected

    assert len(
        scores
    ) == len(
        candidates
    )

    assert (
        scores
        .notna()
        .all()
        .all()
    )