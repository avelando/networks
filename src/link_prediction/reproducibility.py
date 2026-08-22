import json
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from link_prediction.config import METADATA_RESULTS_DIR

PACKAGES = [
    "networkx",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "PyYAML",
    "requests",
    "tqdm",
    "statsmodels",
]


def get_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def collect_environment_metadata() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "git_commit": get_git_commit(),
        "packages": {
            package: get_package_version(package)
            for package in PACKAGES
        },
    }


def save_environment_metadata(
    filename: str = "environment.json",
) -> Path:
    METADATA_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = METADATA_RESULTS_DIR / filename

    metadata = collect_environment_metadata()

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            sort_keys=True,
        )

    return output_path