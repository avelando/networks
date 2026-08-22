from functools import lru_cache
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


@lru_cache(maxsize=None)
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