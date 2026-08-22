# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for DualQueueThreadPool's adaptive batch-splitting mechanics.

These tests exercise the pool's internal scheduling (idle-thread tracking,
the smoothed _IdleEwma signal, and pop-time batch slicing) directly, without
any real disk I/O -- make_batch_fn stand-ins are pure and instantaneous.
"""

import math
import threading
import time
from collections.abc import Callable

import pytest

from vllm.v1.kv_offload.tiering.fs import thread_pool as tp_mod
from vllm.v1.kv_offload.tiering.fs.thread_pool import (
    DualQueueThreadPool,
    Task,
    _IdleEwma,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tasks(n: int) -> list[Task]:
    # `key`/`path` are unused by these tests; only `offset` (as a task index)
    # is inspected, so plain ints stand in for the real OffloadKey/path.
    return [Task(key=i, path="unused", offset=i) for i in range(n)]


def _recording_make_batch_fn(
    calls: list[list[int]],
) -> Callable[[list[Task]], Callable[[], None]]:
    """make_batch_fn stand-in that records the task indices it was handed
    and, when invoked, does nothing (no real I/O)."""

    def make_batch_fn(tasks: list[Task]) -> Callable[[], None]:
        calls.append([t.offset for t in tasks])
        return lambda: None

    return make_batch_fn


class _FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def fake_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(tp_mod.time, "monotonic", clock)
    return clock


# ---------------------------------------------------------------------------
# _IdleEwma: pure math, deterministic via a fake clock.
# ---------------------------------------------------------------------------


def test_idle_ewma_initial_value_equals_initial_raw(fake_clock):
    ewma = _IdleEwma(initial=4, tau=1.0)
    assert ewma.value() == 4.0


def test_idle_ewma_no_decay_without_elapsed_time(fake_clock):
    """update()/value() must not move the average until real time passes."""
    ewma = _IdleEwma(initial=4, tau=1.0)
    ewma.update(0)
    assert ewma.value() == pytest.approx(4.0)


def test_idle_ewma_decays_exponentially_with_elapsed_time(fake_clock):
    ewma = _IdleEwma(initial=8, tau=1.0)
    ewma.update(0)
    fake_clock.advance(1.0)  # exactly one time constant
    expected = 8 * math.exp(-1) + 0 * (1 - math.exp(-1))
    assert ewma.value() == pytest.approx(expected)


def test_idle_ewma_converges_to_raw_after_long_dwell(fake_clock):
    ewma = _IdleEwma(initial=0, tau=0.05)
    ewma.update(8)
    fake_clock.advance(10.0)  # 200 time constants
    assert ewma.value() == pytest.approx(8.0, abs=1e-6)


def test_idle_ewma_sequential_updates_match_manual_computation(fake_clock):
    ewma = _IdleEwma(initial=4, tau=1.0)
    fake_clock.advance(0.5)
    ewma.update(2)
    fake_clock.advance(0.5)
    actual = ewma.value()

    value, raw = 4.0, 4
    value = value * math.exp(-0.5) + raw * (1 - math.exp(-0.5))
    raw = 2
    value = value * math.exp(-0.5) + raw * (1 - math.exp(-0.5))
    assert actual == pytest.approx(value)


@pytest.mark.parametrize(
    "updates",
    [
        [(4, 0.01), (0, 0.2), (8, 0.001), (0, 0.5)],
        [(0, 0.0), (0, 0.0), (0, 1.0)],
    ],
)
def test_idle_ewma_never_negative_for_nonnegative_raw(fake_clock, updates):
    ewma = _IdleEwma(initial=0, tau=0.1)
    for raw, dt in updates:
        ewma.update(raw)
        fake_clock.advance(dt)
        assert ewma.value() >= 0.0


# ---------------------------------------------------------------------------
# DualQueueThreadPool: pop-time batch splitting.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_tasks,n_threads", [(1, 4), (5, 4), (37, 4), (100, 8)])
def test_batches_cover_every_task_exactly_once(n_tasks, n_threads):
    """Regression test for the pop-time slicing bug: every task index must be
    handed to make_batch_fn exactly once, with no drops or duplicates,
    regardless of how the job gets split across idle threads."""
    calls: list[list[int]] = []
    pool = DualQueueThreadPool(n_read_threads=n_threads, n_write_threads=0)
    try:
        pool.enqueue_load(
            job_id=1,
            make_batch_fn=_recording_make_batch_fn(calls),
            n_tasks=n_tasks,
            tasks=_make_tasks(n_tasks),
        )
        pool.wait_idle()
    finally:
        pool.shutdown()

    seen = [i for batch in calls for i in batch]
    assert sorted(seen) == list(range(n_tasks))


def test_job_larger_than_thread_count_splits_into_multiple_batches():
    """On a fully-idle pool, a job bigger than the thread count must not be
    swallowed whole by the first thread to pop it."""
    calls: list[list[int]] = []
    pool = DualQueueThreadPool(n_read_threads=4, n_write_threads=0)
    try:
        pool.enqueue_load(
            job_id=1,
            make_batch_fn=_recording_make_batch_fn(calls),
            n_tasks=16,
            tasks=_make_tasks(16),
        )
        pool.wait_idle()
    finally:
        pool.shutdown()

    assert len(calls) > 1
    assert all(len(batch) < 16 for batch in calls)


def test_single_thread_pool_takes_whole_job_when_idle_signal_is_low():
    """Baseline/contrast case: with only one thread ever idle (raw and EWMA
    both == 1), the job is not split at all."""
    calls: list[list[int]] = []
    pool = DualQueueThreadPool(n_read_threads=1, n_write_threads=0)
    try:
        pool.enqueue_load(
            job_id=1,
            make_batch_fn=_recording_make_batch_fn(calls),
            n_tasks=8,
            tasks=_make_tasks(8),
        )
        pool.wait_idle()
    finally:
        pool.shutdown()

    assert calls == [list(range(8))]


def test_smoothed_idle_signal_drives_split_not_instantaneous_count():
    """Regression test for the monopolization bug: a thread popping work must
    size its batch off the *smoothed* idle signal, not the instantaneous
    count -- otherwise a thread that just went idle (instantaneous count of
    1) swallows an entire job even though several threads were idle a
    moment ago.

    Uses a single real worker thread but seeds its EWMA to simulate "the
    pool was recently servicing with 4 idle threads" independent of the
    real (1) instantaneous count, avoiding a timing-dependent multi-thread
    race.
    """
    calls: list[list[int]] = []
    pool = DualQueueThreadPool(
        n_read_threads=1, n_write_threads=0, idle_ewma_tau_s=10.0
    )
    try:
        with pool._condition:
            assert pool._idle_read_threads == 1
            pool._idle_read_ewma._value = 4.0
            pool._idle_read_ewma._raw = pool._idle_read_threads
            pool._idle_read_ewma._last_t = time.monotonic()

        pool.enqueue_load(
            job_id=1,
            make_batch_fn=_recording_make_batch_fn(calls),
            n_tasks=8,
            tasks=_make_tasks(8),
        )
        pool.wait_idle()
    finally:
        pool.shutdown()

    # ceil(8 / 4) == 2: the first pop must leave work behind for whoever
    # becomes idle next, instead of taking all 8 tasks.
    assert len(calls[0]) == 2


def test_worker_activity_updates_idle_ewma_not_just_raw_count():
    """Going busy must move the smoothed signal, not just the raw counter."""
    pool = DualQueueThreadPool(n_read_threads=2, n_write_threads=0)
    started = threading.Event()

    def make_batch_fn(tasks: list[Task]) -> Callable[[], None]:
        def run() -> None:
            started.set()
            time.sleep(0.2)

        return run

    try:
        pool.enqueue_load(
            job_id=1, make_batch_fn=make_batch_fn, n_tasks=1, tasks=_make_tasks(1)
        )
        assert started.wait(timeout=5.0), "worker never picked up the task"

        deadline = time.monotonic() + 5.0
        with pool._condition:
            while pool._idle_read_threads != 1 and time.monotonic() < deadline:
                pool._condition.wait(timeout=0.05)
            assert pool._idle_read_threads == 1
            assert pool._idle_read_ewma.value() < 2.0
    finally:
        pool.wait_idle()
        pool.shutdown()


def test_idle_counts_return_to_full_after_jobs_complete():
    calls: list[list[int]] = []
    pool = DualQueueThreadPool(n_read_threads=3, n_write_threads=2)
    try:
        for job_id in range(20):
            pool.enqueue_load(
                job_id=job_id,
                make_batch_fn=_recording_make_batch_fn(calls),
                n_tasks=5,
                tasks=_make_tasks(5),
            )
        pool.wait_idle()

        with pool._condition:
            assert pool._idle_read_threads == 3
            assert pool._idle_write_threads == 2
    finally:
        pool.shutdown()
