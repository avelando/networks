import os
from pathlib import Path

import psutil


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
            return len(
                affinity
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


def cgroup_available_memory_bytes() -> int | None:
    memory_max = Path(
        "/sys/fs/cgroup/memory.max"
    )

    memory_current = Path(
        "/sys/fs/cgroup/memory.current"
    )

    if (
        memory_max.exists()
        and memory_current.exists()
    ):
        maximum_text = (
            memory_max
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

        if maximum_text != "max":
            maximum = int(
                maximum_text
            )

            current = int(
                memory_current
                .read_text(
                    encoding="utf-8"
                )
                .strip()
            )

            return max(
                0,
                maximum - current,
            )

    memory_limit = Path(
        "/sys/fs/cgroup/memory/"
        "memory.limit_in_bytes"
    )

    memory_usage = Path(
        "/sys/fs/cgroup/memory/"
        "memory.usage_in_bytes"
    )

    if (
        memory_limit.exists()
        and memory_usage.exists()
    ):
        maximum = int(
            memory_limit
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

        current = int(
            memory_usage
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

        return max(
            0,
            maximum - current,
        )

    return None


def available_memory_bytes() -> int:
    system_available = int(
        psutil.virtual_memory()
        .available
    )

    cgroup_available = (
        cgroup_available_memory_bytes()
    )

    if cgroup_available is None:
        return system_available

    return min(
        system_available,
        cgroup_available,
    )


def choose_process_count(
    logical_cpus: int,
    memory_bytes: int | None,
    memory_per_worker_gib: float = 4.0,
    reserve_memory_gib: float = 1.0,
    hard_cap: int = 8,
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

    cpu_limit = max(
        1,
        logical_cpus // 2,
    )

    memory_limit = hard_cap

    if memory_bytes is not None:
        reserved_memory = int(
            reserve_memory_gib
            * GIB
        )

        usable_memory = max(
            0,
            memory_bytes
            - reserved_memory,
        )

        memory_per_worker = int(
            memory_per_worker_gib
            * GIB
        )

        memory_limit = max(
            1,
            usable_memory
            // memory_per_worker,
        )

    return max(
        1,
        min(
            cpu_limit,
            memory_limit,
            hard_cap,
        ),
    )


def resolve_process_count(
    max_workers: int | str | None = "auto",
) -> int:
    if (
        max_workers is None
        or max_workers == "auto"
    ):
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

        else:
            return choose_process_count(
                logical_cpus=
                    available_logical_cpus(),
                memory_bytes=
                    available_memory_bytes(),
            )

    if isinstance(
        max_workers,
        int,
    ):
        if max_workers < 1:
            raise ValueError(
                "max_workers must be at least 1."
            )

        return max_workers

    raise ValueError(
        "max_workers must be an integer or 'auto'."
    )