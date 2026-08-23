import pytest

from link_prediction.execution import (
    GIB,
    choose_process_count,
    resolve_process_count,
)


def test_choose_process_count_from_cpu():
    workers = (
        choose_process_count(
            logical_cpus=12,
            physical_cpus=6,
            memory_bytes=64 * GIB,
            reserve_memory_gib=2.0,
        )
    )

    assert workers == 6


def test_choose_process_count_respects_memory():
    workers = (
        choose_process_count(
            logical_cpus=12,
            physical_cpus=6,
            memory_bytes=8 * GIB,
            memory_per_worker_gib=4.0,
            reserve_memory_gib=2.0,
        )
    )

    assert workers == 1


def test_more_resources_allow_more_workers():
    workers = (
        choose_process_count(
            logical_cpus=32,
            physical_cpus=16,
            memory_bytes=128 * GIB,
            memory_per_worker_gib=6.0,
            reserve_memory_gib=2.0,
            hard_cap=16,
        )
    )

    assert workers == 16


def test_task_count_limits_workers():
    workers = (
        choose_process_count(
            logical_cpus=32,
            physical_cpus=16,
            memory_bytes=128 * GIB,
            task_count=3,
        )
    )

    assert workers == 3


def test_resolve_explicit_process_count():
    assert (
        resolve_process_count(
            3,
            task_count=10,
        )
        == 3
    )


def test_invalid_process_count():
    with pytest.raises(
        ValueError
    ):
        resolve_process_count(
            0
        )