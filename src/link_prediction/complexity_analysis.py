from pathlib import Path
from typing import Any

import pandas as pd

from link_prediction.config import (
    SUMMARY_RESULTS_DIR,
    load_methods_config,
    resolve_analysis_family_map,
    resolve_method_complexity_map,
)
from link_prediction.statistical_analysis import (
    write_statistical_csv,
)


def build_method_complexity_table(
    methods_config: (
        dict[str, Any] | None
    ) = None,
) -> pd.DataFrame:
    if methods_config is None:
        methods_config = (
            load_methods_config()
        )

    analysis_family_map = (
        resolve_analysis_family_map(
            methods_config
        )
    )

    complexity_map = (
        resolve_method_complexity_map(
            methods_config
        )
    )

    complexity_classes = (
        methods_config[
            "complexity_classes"
        ]
    )

    rows = []

    for (
        method_id,
        method_config,
    ) in methods_config[
        "methods"
    ].items():
        if not method_config.get(
            "enabled",
            True,
        ):
            continue

        family = method_config[
            "family"
        ]

        complexity_id = (
            complexity_map[
                method_id
            ]
        )

        complexity_config = (
            complexity_classes[
                complexity_id
            ]
        )

        rows.append(
            {
                "method_id":
                    method_id,
                "method":
                    method_config[
                        "name"
                    ],
                "family":
                    family,
                "analysis_family":
                    analysis_family_map[
                        family
                    ],
                "complexity_id":
                    complexity_id,
                "complexity":
                    complexity_config[
                        "expression"
                    ],
                "complexity_latex":
                    complexity_config[
                        "latex"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


def run_complexity_analysis(
    output_dir: Path = (
        SUMMARY_RESULTS_DIR
    ),
    methods_config: (
        dict[str, Any] | None
    ) = None,
) -> pd.DataFrame:
    complexity_table = (
        build_method_complexity_table(
            methods_config
        )
    )

    write_statistical_csv(
        complexity_table,
        output_dir
        / "revision_method_complexity.csv",
    )

    return complexity_table