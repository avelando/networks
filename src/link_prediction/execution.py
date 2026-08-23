import os

GIB = 1024 ** 3


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

    return max(
        1,
        os.cpu_count()
        or 1,
    )


def available_memory_bytes() -> int | None:
    try:
        pages = os.sysconf(
            "SC_AVPHYS_PAGES"
        )

        page_size = os.sysconf(
            "SC_PAGE_SIZE"
        )
    except (
        AttributeError,
        OSError,
        ValueError,
    ):
        return None

    return int(
        pages
    ) * int(
        page_size
    )


def choose_process_count(
    logical_cpus: int,
    memory_bytes: int | None,
    memory_per_worker_gib: float = 4.0,
    hard_cap: int = 8,
) -> int:
    if logical_cpus < 1:
        raise ValueError(
            "logical_cpus must be at least 1."
        )

    if (
        memory_per_worker_gib
        <= 0
    ):
        raise ValueError(
            "memory_per_worker_gib must be positive."
        )

    cpu_limit = max(
        1,
        logical_cpus // 2,
    )

    memory_limit = hard_cap

    if memory_bytes is not None:
        memory_per_worker = int(
            memory_per_worker_gib
            * GIB
        )

        memory_limit = max(
            1,
            memory_bytes
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