import os
from collections.abc import (
    Callable,
    Sequence,
)
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from typing import Any

import psutil

from link_prediction.config import (
    load_experiment_config,
)

GIB = 1024**3


def available_logical_cpus() -> int:
    if hasattr(
        os,
        "sched_getaffinity",
    ):
        try:
            return max(
                1,
                len(
                    os.sched_getaffinity(
                        0
                    )
                ),
            )
        except OSError:
            pass

    try:
        affinity = (
            psutil.Process()
            .cpu_affinity()
        )

        if affinity:
            return max(
                1,
                len(affinity),
            )

    except (
        AttributeError,
        NotImplementedError,
        psutil.Error,
    ):
        pass

    return max(
        1,
        psutil.cpu_count(
            logical=True
        )
        or os.cpu_count()
        or 1,
    )


def available_physical_cpus() -> int:
    logical = (
        available_logical_cpus()
    )

    physical = (
        psutil.cpu_count(
            logical=False
        )
    )

    if physical is None:
        return max(
            1,
            logical // 2,
        )

    return max(
        1,
        min(
            logical,
            physical,
        ),
    )


def available_memory_bytes() -> int:
    return int(
        psutil.virtual_memory()
        .available
    )


def choose_process_count(
    logical_cpus: int,
    memory_bytes: int | None,
    *,
    physical_cpus: int | None = None,
    memory_per_worker_gib: float = 4.0,
    reserve_memory_gib: float = 2.0,
    hard_cap: int = 16,
    task_count: int | None = None,
) -> int:
    if logical_cpus < 1:
        raise ValueError(
            "logical_cpus must be at least 1."
        )

    if memory_per_worker_gib <= 0:
        raise ValueError(
            "memory_per_worker_gib must be positive."
        )

    if reserve_memory_gib < 0:
        raise ValueError(
            "reserve_memory_gib cannot be negative."
        )

    if hard_cap < 1:
        raise ValueError(
            "hard_cap must be at least 1."
        )

    if physical_cpus is None:
        cpu_limit = max(
            1,
            logical_cpus // 2,
        )
    else:
        cpu_limit = max(
            1,
            min(
                logical_cpus,
                physical_cpus,
            ),
        )

    memory_limit = hard_cap

    if memory_bytes is not None:
        reserve_bytes = int(
            reserve_memory_gib
            * GIB
        )

        usable_memory = max(
            0,
            memory_bytes
            - reserve_bytes,
        )

        worker_bytes = int(
            memory_per_worker_gib
            * GIB
        )

        memory_limit = max(
            1,
            usable_memory
            // worker_bytes,
        )

    worker_count = max(
        1,
        min(
            cpu_limit,
            memory_limit,
            hard_cap,
        ),
    )

    if task_count is not None:
        worker_count = min(
            worker_count,
            max(
                1,
                task_count,
            ),
        )

    return worker_count


def resolve_process_count(
    max_workers: int | str | None = None,
    *,
    profile: str = "method_benchmark",
    task_count: int | None = None,
) -> int:
    config = (
        load_experiment_config()
        .get(
            "execution",
            {},
        )
    )

    configured_workers = (
        config.get(
            "max_workers",
            "auto",
        )
    )

    if max_workers is None:
        max_workers = (
            configured_workers
        )

    environment_override = (
        os.getenv(
            "LINK_PREDICTION_MAX_WORKERS"
        )
    )

    if environment_override:
        try:
            max_workers = int(
                environment_override
            )
        except ValueError as error:
            raise ValueError(
                "LINK_PREDICTION_MAX_WORKERS "
                "must be an integer."
            ) from error

    if isinstance(
        max_workers,
        int,
    ):
        if max_workers < 1:
            raise ValueError(
                "max_workers must be at least 1."
            )

        if task_count is None:
            return max_workers

        return min(
            max_workers,
            max(
                1,
                task_count,
            ),
        )

    if max_workers != "auto":
        raise ValueError(
            "max_workers must be "
            "an integer, None, or 'auto'."
        )

    memory_profiles = (
        config.get(
            "memory_per_worker_gib",
            {},
        )
    )

    memory_per_worker = float(
        memory_profiles.get(
            profile,
            4.0,
        )
    )

    return choose_process_count(
        logical_cpus=
            available_logical_cpus(),
        physical_cpus=
            available_physical_cpus(),
        memory_bytes=
            available_memory_bytes(),
        memory_per_worker_gib=
            memory_per_worker,
        reserve_memory_gib=float(
            config.get(
                "reserve_memory_gib",
                2.0,
            )
        ),
        hard_cap=int(
            config.get(
                "hard_cap",
                16,
            )
        ),
        task_count=
            task_count,
    )


def run_process_tasks(
    worker: Callable[..., Any],
    tasks: Sequence[tuple[Any, ...]],
    *,
    max_workers: int | str | None = None,
    profile: str = "method_benchmark",
    label: str = "tasks",
) -> list[Any]:
    if not tasks:
        return []

    worker_count = (
        resolve_process_count(
            max_workers,
            profile=profile,
            task_count=len(tasks),
        )
    )

    print(
        f"{label}: "
        f"{len(tasks)} tasks, "
        f"{worker_count} workers"
    )

    if worker_count == 1:
        return [
            worker(
                *task
            )
            for task
            in tasks
        ]

    results: list[
        Any | None
    ] = [
        None
    ] * len(
        tasks
    )

    with ProcessPoolExecutor(
        max_workers=
            worker_count
    ) as executor:
        futures = {
            executor.submit(
                worker,
                *task,
            ): index
            for index, task
            in enumerate(
                tasks
            )
        }

        for future in (
            as_completed(
                futures
            )
        ):
            index = futures[
                future
            ]

            results[
                index
            ] = future.result()

    return results