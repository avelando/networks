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
            memory_bytes=64 * GIB,
        )
    )

    assert workers == 6


def test_choose_process_count_respects_memory():
    workers = (
        choose_process_count(
            logical_cpus=12,
            memory_bytes=8 * GIB,
            memory_per_worker_gib=4.0,
        )
    )

    assert workers == 2


def test_resolve_explicit_process_count():
    assert (
        resolve_process_count(
            3
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