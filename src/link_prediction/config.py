from functools import cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FOLDS_DATA_DIR = DATA_DIR / "folds"

RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_RESULTS_DIR = RESULTS_DIR / "summaries"
METADATA_RESULTS_DIR = RESULTS_DIR / "metadata"


@cache
def load_yaml_config(filename: str) -> dict[str, Any]:
    path = CONFIG_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data or {}


def load_experiment_config() -> dict[str, Any]:
    return load_yaml_config("experiment.yaml")


def load_networks_config() -> dict[str, Any]:
    return load_yaml_config("networks.yaml")


def load_methods_config() -> dict[str, Any]:
    return load_yaml_config("methods.yaml")


def resolve_analysis_family_map(
    methods_config: dict[str, Any],
) -> dict[str, str]:
    execution_families = set(
        methods_config[
            "families"
        ]
    )

    analysis_families = (
        methods_config[
            "analysis_families"
        ]
    )

    mapping = {}

    for (
        analysis_family,
        analysis_config,
    ) in analysis_families.items():
        grouped_families = list(
            analysis_config.get(
                "execution_families",
                [],
            )
        )

        if not grouped_families:
            raise ValueError(
                "Analysis family has no "
                "execution families: "
                f"{analysis_family}"
            )

        for execution_family in (
            grouped_families
        ):
            if (
                execution_family
                not in execution_families
            ):
                raise ValueError(
                    "Analysis family references "
                    "an unknown execution family: "
                    f"{execution_family}"
                )

            if execution_family in mapping:
                raise ValueError(
                    "Execution family belongs to "
                    "multiple analysis families: "
                    f"{execution_family}"
                )

            mapping[
                execution_family
            ] = analysis_family

    missing_families = (
        execution_families
        - set(mapping)
    )

    if missing_families:
        raise ValueError(
            "Execution families are missing "
            "from analysis families: "
            f"{sorted(missing_families)}"
        )

    enabled_methods = [
        method_config
        for method_config
        in methods_config[
            "methods"
        ].values()
        if method_config.get(
            "enabled",
            True,
        )
    ]

    analysis_family_sizes = {
        analysis_family: sum(
            mapping[
                method_config[
                    "family"
                ]
            ]
            == analysis_family
            for method_config
            in enabled_methods
        )
        for analysis_family
        in analysis_families
    }

    singleton_families = sorted(
        analysis_family
        for (
            analysis_family,
            method_count,
        ) in analysis_family_sizes.items()
        if method_count < 2
    )

    if singleton_families:
        raise ValueError(
            "Analysis families require at "
            "least two enabled methods: "
            f"{singleton_families}"
        )

    return mapping


def resolve_method_complexity_map(
    methods_config: dict[str, Any],
) -> dict[str, str]:
    methods = methods_config[
        "methods"
    ]

    complexity_classes = (
        methods_config[
            "complexity_classes"
        ]
    )

    mapping = {}

    for (
        complexity_id,
        complexity_config,
    ) in complexity_classes.items():
        method_ids = list(
            complexity_config.get(
                "methods",
                [],
            )
        )

        if not method_ids:
            raise ValueError(
                "Complexity class has no "
                "methods: "
                f"{complexity_id}"
            )

        for method_id in method_ids:
            if method_id not in methods:
                raise ValueError(
                    "Complexity class references "
                    "an unknown method: "
                    f"{method_id}"
                )

            if method_id in mapping:
                raise ValueError(
                    "Method belongs to multiple "
                    "complexity classes: "
                    f"{method_id}"
                )

            mapping[
                method_id
            ] = complexity_id

    enabled_method_ids = {
        method_id
        for method_id, method_config
        in methods.items()
        if method_config.get(
            "enabled",
            True,
        )
    }

    missing_method_ids = (
        enabled_method_ids
        - set(mapping)
    )

    if missing_method_ids:
        raise ValueError(
            "Enabled methods are missing "
            "complexity classes: "
            f"{sorted(missing_method_ids)}"
        )

    return mapping


def ensure_project_directories() -> None:
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FOLDS_DATA_DIR,
        SUMMARY_RESULTS_DIR,
        METADATA_RESULTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)