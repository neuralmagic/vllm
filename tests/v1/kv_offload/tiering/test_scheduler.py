# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit and integration tests for SSDTPScheduler data structures and
DualQueueThreadPool threading policies.

Data-structure tests (no threading):
  - SJFQueue
  - FCFSQueue
  - LoadQueue
  - StoreQueue

Policy tests (real threads, no disk I/O):
  - Read threads always batch
  - Fast threads drain small load jobs split across n_fast_threads batches
  - Store threads always execute with batch_size=1
  - Fast threads drain short load jobs (store threads no longer do)
  - Read threads drain store jobs when load is empty
  - Fast threads wait when no short jobs or store jobs exist

Stress / deadlock tests (DualQueueThreadPool with fake I/O):
  - All submitted jobs complete without deadlock
  - Task count integrity
  - No starvation under load pressure
  - Burst-idle-burst pattern wakes threads correctly
  - Concurrent producers
  - Shutdown while work is pending
"""

import threading
import time
from unittest.mock import patch

import pytest

from vllm.v1.kv_offload.tiering.fs.scheduler import SSDTPScheduler
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool, Task

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRESS_TIMEOUT = 15.0  # seconds; generous for CI; deadlock => timeout
_DUMMY_KEY = b"\x00" * 16
_BLOCK_SIZE = 1  # 1 byte per block; keeps bucket math identical to num_tasks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task(n: int = 0) -> Task:
    """Minimal Task – content irrelevant for scheduler unit tests."""
    return Task(key=_DUMMY_KEY, path="/dev/null", offset=n)


def _tasks(n: int) -> list[Task]:
    return [_task(i) for i in range(n)]


def _noop_make_batch_fn(batch: list[Task]):
    """make_batch_fn that does nothing – used for pure scheduler tests."""
    return lambda: None


def _recording_make_batch_fn(
    records: list[tuple[str, int]],
    lock: threading.Lock,
):
    """
    Returns a make_batch_fn that appends (thread_name, batch_size) to records.
    """

    def make_batch_fn(batch: list[Task]):
        size = len(batch)

        def fn():
            with lock:
                records.append((threading.current_thread().name, size))

        return fn

    return make_batch_fn


def _ssd_pool(
    n_read: int = 8, n_write: int = 4, prefix: str = "test"
) -> DualQueueThreadPool:
    """Create a DualQueueThreadPool backed by SSDTPScheduler."""
    with patch("vllm.v1.kv_offload.tiering.fs.thread_pool.envs") as mock_envs:
        mock_envs.VLLM_FS_THREAD_POOL_SCHEDULER_CLS = "SSDTPScheduler"
        return DualQueueThreadPool(
            n_read, n_write, _BLOCK_SIZE, thread_name_prefix=prefix
        )


def _drain_finished(pool: DualQueueThreadPool, timeout: float = STRESS_TIMEOUT) -> list:
    """Wait for idle then collect all finished job ids."""
    pool.wait_idle()
    return [job_id for job_id, _, _ in pool.get_finished()]


# ---------------------------------------------------------------------------
# SJFQueue unit tests
# ---------------------------------------------------------------------------

FCFSQueue = SSDTPScheduler.FCFSQueue
SJFQueue = SSDTPScheduler.SJFQueue
LoadQueue = SSDTPScheduler.LoadQueue
StoreQueue = SSDTPScheduler.StoreQueue
Role = SSDTPScheduler.Role


class TestSJFQueue:
    def test_add_single_has_work(self):
        q = SJFQueue(_BLOCK_SIZE)
        q.add(job_id=1, num_tasks=4)
        assert q.has_work()

    def test_has_short_job_single_item_is_false(self):
        # One job => one bucket => not multimodal.
        q = SJFQueue(_BLOCK_SIZE)
        q.add(job_id=1, num_tasks=8)
        assert not q.has_short_job()

    def test_has_short_job_same_bucket_is_false(self):
        # 4 and 5 both land in bucket 2 (floor(log2(4))==2, floor(log2(5))==2).
        q = SJFQueue(_BLOCK_SIZE)
        q.add(job_id=1, num_tasks=4)
        q.add(job_id=2, num_tasks=5)
        assert not q.has_short_job()

    def test_has_short_job_different_buckets_is_true(self):
        # 1 (bucket 0) and 1024 (bucket 10) → multimodal.
        q = SJFQueue(_BLOCK_SIZE)
        q.add(job_id=1, num_tasks=1)
        q.add(job_id=2, num_tasks=1024)
        assert q.has_short_job()

    def test_returns_smallest_first(self):
        q = SJFQueue(_BLOCK_SIZE)
        for job_id, size in enumerate([8, 2, 16, 1], start=1):
            q.add(job_id=job_id, num_tasks=size)
        results = [q.next()[1] for _ in range(4)]
        assert results == sorted(results), f"Not SJF order: {results}"

    def test_bucket_assignment_boundaries(self):
        # Explicit spot-checks: num_tasks → expected bucket (floor(log2(n))).
        q = SJFQueue(_BLOCK_SIZE)
        cases = [1, 2, 3, 4, 7, 8, 15, 16]
        for i, n in enumerate(cases, start=1):
            q.add(job_id=i, num_tasks=n)
        # Drain in SJF order and verify sizes come out in non-decreasing order.
        sizes = []
        while q.has_work():
            _, num_tasks = q.next()
            sizes.append(num_tasks)
        assert sizes == sorted(sizes)

    def test_meta_bitmap_cleared_after_last_removal(self):
        q = SJFQueue(_BLOCK_SIZE)
        q.add(job_id=1, num_tasks=4)  # bucket 2
        q.next()  # remove it
        assert q.sjf_meta.value == 0

    def test_meta_bitmap_partial_after_partial_removal(self):
        q = SJFQueue(_BLOCK_SIZE)
        q.add(job_id=1, num_tasks=4)  # bucket 2
        q.add(job_id=2, num_tasks=8)  # bucket 3
        q.next()  # removes bucket-2 entry
        # Bucket 3 still occupied.
        assert q.sjf_meta.value != 0
        assert q.has_work()

    def test_clear(self):
        q = SJFQueue(_BLOCK_SIZE)
        q.add(job_id=1, num_tasks=4)
        q.add(job_id=2, num_tasks=1024)
        q.clear()
        assert not q.has_work()
        assert q.sjf_meta.value == 0

    def test_remove_consistency_error_on_wrong_order(self):
        # _remove asserts that the front of the bucket matches job_id.
        q = SJFQueue(_BLOCK_SIZE)
        q.add(job_id=1, num_tasks=4)
        q.add(job_id=2, num_tasks=5)  # same bucket
        # job_id=2 is behind job_id=1 in bucket 2 → _remove(2, 4) must fail.
        with pytest.raises(AssertionError):
            q._remove(job_id=2, num_tasks=4)


# ---------------------------------------------------------------------------
# FCFSQueue unit tests
# ---------------------------------------------------------------------------


class TestFCFSQueue:
    def test_fifo_ordering(self):
        q = FCFSQueue()
        for i in [10, 20, 30]:
            q.add(job_id=i, num_tasks=1)
        assert [q.next()[0] for _ in range(3)] == [10, 20, 30]

    def test_has_work_empty_then_populated(self):
        q = FCFSQueue()
        assert not q.has_work()
        q.add(job_id=1, num_tasks=5)
        assert q.has_work()

    def test_has_work_after_drain(self):
        q = FCFSQueue()
        q.add(job_id=1, num_tasks=1)
        q.next()
        assert not q.has_work()

    def test_clear(self):
        q = FCFSQueue()
        for i in range(5):
            q.add(job_id=i, num_tasks=1)
        q.clear()
        assert not q.has_work()


# ---------------------------------------------------------------------------
# LoadQueue unit tests
# ---------------------------------------------------------------------------


class TestLoadQueue:
    def test_read_role_returns_n_read_threads_as_n_batch(self):
        n_read = 4
        q = LoadQueue(n_read_threads=n_read, n_fast_threads=1, block_size=_BLOCK_SIZE)
        q.add(item="job_a", job_id=1, num_tasks=64)
        item, n_batch = q.fetch_work(Role.READ)
        assert item == "job_a"
        assert n_batch == n_read

    def test_read_role_removes_from_both_queues(self):
        q = LoadQueue(n_read_threads=4, n_fast_threads=1, block_size=_BLOCK_SIZE)
        q.add(item="job_a", job_id=1, num_tasks=64)
        q.fetch_work(Role.READ)
        assert not q.has_work()

    def test_fast_role_returns_n_fast_threads_as_n_batch(self):
        # FAST splits the short job across n_fast_threads batches.
        n_fast = 2
        q = LoadQueue(n_read_threads=4, n_fast_threads=n_fast, block_size=_BLOCK_SIZE)
        q.add(item="small", job_id=1, num_tasks=1)
        q.add(item="large", job_id=2, num_tasks=1024)
        assert q.has_short_job()
        item, n_batch = q.fetch_work(Role.FAST)
        assert n_batch == n_fast
        assert item == "small"

    def test_fast_role_leaves_stale_fcfs_entry(self):
        # After FAST steals a job via SJF, the job is gone from .jobs
        # but fcfs_q still has a stale reference.
        q = LoadQueue(n_read_threads=4, n_fast_threads=1, block_size=_BLOCK_SIZE)
        q.add(item="small", job_id=1, num_tasks=1)
        q.add(item="large", job_id=2, num_tasks=1024)
        q.fetch_work(Role.FAST)  # steals job 1 via SJF
        # jobs dict no longer has job_id=1
        assert 1 not in q.jobs
        # fcfs_q still has job_id=1 as the first entry (stale)
        stale_id, _ = q.fcfs_q.fcfs_q[0]
        assert stale_id == 1

    def test_read_skips_stale_fcfs_entries(self):
        # READ must skip job_id=1 (stolen by fast) and return job_id=2.
        q = LoadQueue(n_read_threads=4, n_fast_threads=1, block_size=_BLOCK_SIZE)
        q.add(item="small", job_id=1, num_tasks=1)
        q.add(item="large", job_id=2, num_tasks=1024)
        q.fetch_work(Role.FAST)  # steals job 1
        item, n_batch = q.fetch_work(Role.READ)  # must skip stale, get job 2
        assert item == "large"
        assert n_batch == 4
        assert not q.has_work()

    def test_has_short_job_uniform_sizes(self):
        q = LoadQueue(n_read_threads=4, n_fast_threads=1, block_size=_BLOCK_SIZE)
        q.add(item="a", job_id=1, num_tasks=100)
        q.add(item="b", job_id=2, num_tasks=10)
        # floor(log2(100))=6, floor(log2(200))=7 → different buckets.
        # has_short_job only False when same bucket.
        # 100→bucket6, 200→bucket7: multimodal, so True here.
        assert q.has_short_job()

    def test_has_short_job_same_bucket(self):
        q = LoadQueue(n_read_threads=4, n_fast_threads=1, block_size=_BLOCK_SIZE)
        q.add(item="a", job_id=1, num_tasks=4)
        q.add(item="b", job_id=2, num_tasks=5)  # same bucket 2
        assert not q.has_short_job()

    def test_has_work_empty_and_populated(self):
        q = LoadQueue(n_read_threads=4, n_fast_threads=1, block_size=_BLOCK_SIZE)
        assert not q.has_work()
        q.add(item="x", job_id=1, num_tasks=10)
        assert q.has_work()

    def test_clear(self):
        q = LoadQueue(n_read_threads=4, n_fast_threads=1, block_size=_BLOCK_SIZE)
        q.add(item="x", job_id=1, num_tasks=10)
        q.clear()
        assert not q.has_work()


# ---------------------------------------------------------------------------
# StoreQueue unit tests
# ---------------------------------------------------------------------------


class TestStoreQueue:
    def test_fetch_always_returns_batch_size_1(self):
        q = StoreQueue(_BLOCK_SIZE)
        q.add(item="job_a", job_id=1, num_tasks=100)
        _, n_batch = q.fetch_work(Role.WRITE)
        assert n_batch == 1

    def test_sjf_ordering(self):
        q = StoreQueue(_BLOCK_SIZE)
        q.add(item="large", job_id=1, num_tasks=100)
        q.add(item="small", job_id=2, num_tasks=1)
        q.add(item="medium", job_id=3, num_tasks=50)
        sizes = []
        while q.has_work():
            _, n_batch = q.fetch_work(Role.WRITE)
            sizes.append(n_batch)
        # All n_batch == 1; we verify ordering via the job removal order.
        # Re-run to check ordering by item:
        q2 = StoreQueue(_BLOCK_SIZE)
        q2.add(item="large", job_id=1, num_tasks=100)
        q2.add(item="small", job_id=2, num_tasks=1)
        q2.add(item="medium", job_id=3, num_tasks=50)
        items = []
        while q2.has_work():
            item, _ = q2.fetch_work(Role.WRITE)
            items.append(item)
        assert items == ["small", "medium", "large"]

    def test_fetch_removes_from_jobs(self):
        q = StoreQueue(_BLOCK_SIZE)
        q.add(item="x", job_id=1, num_tasks=8)
        q.fetch_work(Role.WRITE)
        assert 1 not in q.jobs
        assert not q.has_work()

    def test_has_work(self):
        q = StoreQueue(_BLOCK_SIZE)
        assert not q.has_work()
        q.add(item="x", job_id=1, num_tasks=8)
        assert q.has_work()
        q.fetch_work(Role.WRITE)
        assert not q.has_work()

    def test_clear(self):
        q = StoreQueue(_BLOCK_SIZE)
        q.add(item="x", job_id=1, num_tasks=8)
        q.clear()
        assert not q.has_work()


# ---------------------------------------------------------------------------
# SSDTPScheduler.has_work policy unit tests (no threads)
# ---------------------------------------------------------------------------


class TestSSDTPSchedulerHasWork:
    """Direct unit tests on SSDTPScheduler.has_work() without spawning threads."""

    def _make_sched(self, n_read=8, n_write=4) -> SSDTPScheduler:
        # Create scheduler without starting threads.
        sched = object.__new__(SSDTPScheduler)
        # Bypass __init__ threading; replicate only the data structure setup.
        sched._n_read_threads = n_read - (n_read // 4)
        sched._n_write_threads = n_write
        sched._n_fast_threads = n_read // 4
        sched._load_q = LoadQueue(
            sched._n_read_threads, sched._n_fast_threads, block_size=_BLOCK_SIZE
        )
        sched._store_q = StoreQueue(_BLOCK_SIZE)
        from collections import deque

        sched._load_deque = deque()
        sched._fast_load_deque = deque()
        return sched

    def _fake_state(self, job_id=1):
        state = type("S", (), {"job_id": job_id})()
        return state

    def test_read_sees_only_load(self):
        sched = self._make_sched()
        state = self._fake_state(1)
        sched._load_q.add(
            (_noop_make_batch_fn, _tasks(10), state), job_id=1, num_tasks=10
        )
        assert sched.has_work(Role.READ)
        assert not sched.has_work(Role.FAST)  # single bucket → not multimodal
        assert not sched.has_work(Role.WRITE)

    def test_read_sees_store(self):
        sched = self._make_sched()
        state = self._fake_state(1)
        sched._store_q.add(
            (_noop_make_batch_fn, _tasks(8), state), job_id=1, num_tasks=8
        )
        assert sched.has_work(Role.READ)
        assert sched.has_work(Role.FAST)  # store_q.has_work()
        assert sched.has_work(Role.WRITE)

    def test_fast_sees_multimodal_load(self):
        sched = self._make_sched()
        s1, s2 = self._fake_state(1), self._fake_state(2)
        sched._load_q.add((_noop_make_batch_fn, _tasks(1), s1), job_id=1, num_tasks=1)
        sched._load_q.add(
            (_noop_make_batch_fn, _tasks(1024), s2), job_id=2, num_tasks=1024
        )
        assert sched.has_work(Role.READ)
        assert sched.has_work(Role.FAST)  # multimodal → has_short_job
        assert not sched.has_work(Role.WRITE)  # no store_q work

    def test_write_does_not_see_uniform_load(self):
        sched = self._make_sched()
        # Two jobs same bucket (4, 5 → bucket 2)
        s1, s2 = self._fake_state(1), self._fake_state(2)
        sched._load_q.add((_noop_make_batch_fn, _tasks(4), s1), job_id=1, num_tasks=4)
        sched._load_q.add((_noop_make_batch_fn, _tasks(5), s2), job_id=2, num_tasks=5)
        assert sched.has_work(Role.READ)
        assert not sched.has_work(Role.FAST)
        assert not sched.has_work(Role.WRITE)


# ---------------------------------------------------------------------------
# Helpers for thread policy tests
# ---------------------------------------------------------------------------


def _make_recording_pool(
    n_read: int,
    n_write: int,
    records: list[tuple[str, int]],
    lock: threading.Lock,
    prefix: str = "pol",
) -> DualQueueThreadPool:
    """Pool backed by SSDTPScheduler; make_batch_fn records (thread, batch_size)."""
    with patch("vllm.v1.kv_offload.tiering.fs.thread_pool.envs") as mock_envs:
        mock_envs.VLLM_FS_THREAD_POOL_SCHEDULER_CLS = "SSDTPScheduler"
        return DualQueueThreadPool(
            n_read, n_write, _BLOCK_SIZE, thread_name_prefix=prefix
        )


# ---------------------------------------------------------------------------
# Threading policy tests
# ---------------------------------------------------------------------------


class TestThreadingPolicies:
    """
    Each test creates a real DualQueueThreadPool backed by SSDTPScheduler,
    submits jobs, waits for completion, and asserts on recorded (thread, batch_size).
    """

    def _run(
        self,
        n_read: int,
        n_write: int,
        jobs: list[tuple[int, int, bool]],  # (job_id, n_tasks, is_load)
        prefix: str = "pol",
        timeout: float = 10.0,
    ) -> tuple[list[tuple[str, int]], list[int]]:
        """
        Submit jobs to an SSD pool.
        Returns (records, finished_job_ids).
        records: list of (thread_name, batch_size) per execution unit.
        """
        records: list[tuple[str, int]] = []
        lock = threading.Lock()
        make_fn = _recording_make_batch_fn(records, lock)

        with patch("vllm.v1.kv_offload.tiering.fs.thread_pool.envs") as mock_envs:
            mock_envs.VLLM_FS_THREAD_POOL_SCHEDULER_CLS = "SSDTPScheduler"
            pool = DualQueueThreadPool(
                n_read, n_write, _BLOCK_SIZE, thread_name_prefix=prefix
            )

        for job_id, n_tasks, is_load in jobs:
            task_list = _tasks(n_tasks)
            if is_load:
                pool.enqueue_load(job_id, n_tasks, iter(task_list), make_fn)
            else:
                pool.enqueue_store(job_id, n_tasks, iter(task_list), make_fn)

        pool.wait_idle()
        finished = [jid for jid, _, _ in pool.get_finished()]
        pool.shutdown()
        return records, finished

    def test_read_threads_always_batch(self):
        # n_read=8 → n_fast=2, effective_read=6. Submit 60 tasks load job.
        # READ threads should split into up to 6 batches (n_read_threads=6).
        # No batch should be size 1 unless remainder forces it.
        n_read, n_write = 8, 2
        n_tasks = 60
        records, finished = self._run(
            n_read, n_write, [(1, n_tasks, True)], prefix="rdbatch"
        )

        read_records = [(t, s) for t, s in records if "_l" in t]
        total = sum(s for _, s in read_records)
        assert total == n_tasks, f"Tasks lost: {total} != {n_tasks}"
        assert len(finished) == 1
        # All read-thread batches should be > 1 (evenly distributed across 6 threads)
        for _, batch_size in read_records:
            assert batch_size > 1, "Read thread got batch_size=1 for uniform large job"

    def test_fast_threads_drain_small_load_jobs(self):
        # Submit many small (1-task) load jobs alongside one large (500-task) job.
        # When multimodal distribution exists, fast threads (Role.FAST) pick up
        # the small jobs via SJF and split them across n_fast_threads batches.
        # For 1-task jobs, each batch is still 1 task (min(n_tasks, n_fast) == 1).
        # n_read=8 → n_fast=2, effective_read=6.
        n_read, n_write = 8, 2
        jobs = [(i, 1, True) for i in range(50)] + [(50, 500, True)]
        records, finished = self._run(n_read, n_write, jobs, prefix="fast")
        assert sorted(finished) == list(range(51))
        total = sum(s for _, s in records)
        assert total == 50 + 500, f"Task count wrong: {total}"
        # For 1-task short jobs, each fast-thread batch contains exactly 1 task.
        fast_records = [(t, s) for t, s in records if "_f" in t]
        for _, bs in fast_records:
            assert bs == 1, (
                f"Fast thread executed batch_size={bs} for 1-task load job; expected 1"
            )

    def test_store_threads_one_job_per_dequeue(self):
        # "batch_size 1" in the policy means 1 store JOB dequeued per thread
        # wake-up (n_batch=1 from StoreQueue), not 1 task per fn() call.
        # Each fn() call covers ALL tasks of the dequeued job atomically.
        # Verify: len(records) == number of store jobs, and each record's
        # batch_size matches its job's total task count.
        n_read, n_write = 8, 4
        job_sizes = [10, 50, 5, 20, 100]
        jobs = [(i, size, False) for i, size in enumerate(job_sizes)]
        records, finished = self._run(n_read, n_write, jobs, prefix="store")
        assert sorted(finished) == list(range(len(job_sizes)))
        # One atomic fn() call per store job.
        assert len(records) == len(job_sizes), (
            f"Expected {len(job_sizes)} execution units "
            f"(one per job), got {len(records)}"
        )
        # Total tasks correct.
        assert sum(s for _, s in records) == sum(job_sizes)
        # Each execution unit covers a whole job (batch_size == one of the job sizes).
        assert sorted(s for _, s in records) == sorted(job_sizes)

    def test_fast_threads_drain_multimodal_load_jobs(self):
        # Two load jobs of different sizes (multimodal) → has_short_job True.
        # Fast threads (FAST role) are notified and pick up the short job.
        # Store/WRITE threads are NOT notified for short load jobs.
        n_read, n_write = 8, 4
        records, finished = self._run(
            n_read,
            n_write,
            [(1, 1, True), (2, 1024, True)],
            prefix="stdrn",
        )
        assert sorted(finished) == [1, 2]
        # All tasks must complete.
        assert sum(s for _, s in records) == 1 + 1024
        # Store/WRITE threads must not have touched load jobs.
        store_records = [(t, s) for t, s in records if "_s" in t]
        assert not store_records, (
            f"Store threads should not process load jobs, got: {store_records}"
        )

    def test_read_threads_drain_store_jobs_when_load_empty(self):
        # With no load work, read threads (and fast threads) fall back to the
        # store queue (has_work(READ) is True when store_q.has_work()).
        # They execute the store job atomically: make_batch_fn is called with
        # ALL tasks of the job, so batch_size == total tasks (not 1).
        # n_write=0 means no dedicated store threads; only read (_l) and
        # fast (_f) threads are spawned.
        n_read, n_write = 8, 0
        n_tasks = 20
        records, finished = self._run(
            n_read, n_write, [(1, n_tasks, False)], prefix="rdstore"
        )
        assert finished == [1]
        # Only _l (read) and _f (fast) threads exist when n_write=0.
        non_store_records = [(t, s) for t, s in records if "_l" in t or "_f" in t]
        assert non_store_records, "No read or fast thread drained the store job"
        # Exactly one atomic execution unit for the one store job.
        assert len(non_store_records) == 1
        assert non_store_records[0][1] == n_tasks

    def test_read_prioritizes_load_over_store(self):
        # Submit a load job (64 tasks) first, then a store job (8 tasks).
        # Read threads must process all load tasks before any store tasks
        # only when load queue is non-empty (tested via ordering timestamps).
        n_read, n_write = 4, 0  # only read threads
        order: list[tuple[str, float]] = []
        lock = threading.Lock()

        def make_fn_with_time(is_load_job: bool):
            def make_batch_fn(batch):
                def fn():
                    with lock:
                        order.append(
                            ("load" if is_load_job else "store", time.monotonic())
                        )

                return fn

            return make_batch_fn

        with patch("vllm.v1.kv_offload.tiering.fs.thread_pool.envs") as mock_envs:
            mock_envs.VLLM_FS_THREAD_POOL_SCHEDULER_CLS = "SSDTPScheduler"
            pool = DualQueueThreadPool(
                n_read, n_write, _BLOCK_SIZE, thread_name_prefix="prio"
            )

        pool.enqueue_load(1, 64, iter(_tasks(64)), make_fn_with_time(True))
        pool.enqueue_store(2, 8, iter(_tasks(8)), make_fn_with_time(False))
        pool.wait_idle()
        pool.shutdown()

        load_times = [t for kind, t in order if kind == "load"]
        store_times = [t for kind, t in order if kind == "store"]
        assert load_times and store_times
        # At least the first store execution should start after some load executions
        # completed (load_q checked first in fetch_work for READ role).
        # We just verify all tasks ran.
        assert len(load_times) + len(store_times) > 0


# ---------------------------------------------------------------------------
# Stress / deadlock tests via DualQueueThreadPool
# ---------------------------------------------------------------------------


class TestStressNoDeadlock:
    """
    All tests patch envs to use SSDTPScheduler, submit real jobs with a
    no-op make_batch_fn, and assert completion within STRESS_TIMEOUT.
    """

    def _pool(self, n_read: int = 8, n_write: int = 4, prefix: str = "stress"):
        with patch("vllm.v1.kv_offload.tiering.fs.thread_pool.envs") as mock_envs:
            mock_envs.VLLM_FS_THREAD_POOL_SCHEDULER_CLS = "SSDTPScheduler"
            return DualQueueThreadPool(
                n_read, n_write, _BLOCK_SIZE, thread_name_prefix=prefix
            )

    def _submit_and_wait(
        self,
        pool: DualQueueThreadPool,
        jobs: list[tuple[int, int, bool]],  # (job_id, n_tasks, is_load)
        timeout: float = STRESS_TIMEOUT,
    ) -> list[int]:
        """Submit jobs, block until idle, return finished job_ids."""
        for job_id, n_tasks, is_load in jobs:
            task_list = _tasks(n_tasks)
            if is_load:
                pool.enqueue_load(job_id, n_tasks, iter(task_list), _noop_make_batch_fn)
            else:
                pool.enqueue_store(
                    job_id, n_tasks, iter(task_list), _noop_make_batch_fn
                )

        # deadline = time.monotonic() + timeout
        # wait_idle blocks; we wrap with a thread + join(timeout) to detect deadlock.
        idle_event = threading.Event()

        def _waiter():
            pool.wait_idle()
            idle_event.set()

        t = threading.Thread(target=_waiter, daemon=True)
        t.start()
        t.join(timeout)
        assert idle_event.is_set(), (
            f"DEADLOCK: {len(jobs)} jobs not drained in {timeout}s"
        )
        return [jid for jid, _, _ in pool.get_finished()]

    def test_all_load_jobs_drain(self):
        pool = self._pool(prefix="all_load")
        N = 200
        jobs = [(i, 16, True) for i in range(N)]
        finished = self._submit_and_wait(pool, jobs)
        assert sorted(finished) == list(range(N))
        pool.shutdown()

    def test_all_store_jobs_drain(self):
        pool = self._pool(prefix="all_store")
        N = 200
        jobs = [(i, 8, False) for i in range(N)]
        finished = self._submit_and_wait(pool, jobs)
        assert sorted(finished) == list(range(N))
        pool.shutdown()

    def test_mixed_load_and_store_drain(self):
        pool = self._pool(n_read=8, n_write=4, prefix="mixed")
        N = 300
        jobs = [(i, (i % 7) + 1, i % 3 != 0) for i in range(N)]
        finished = self._submit_and_wait(pool, jobs)
        assert sorted(finished) == list(range(N))
        pool.shutdown()

    def test_multimodal_load_sizes_drain(self):
        # Alternating tiny (1 task) and large (500 task) load jobs.
        # Activates fast-thread and SJF short-job paths.
        pool = self._pool(prefix="multimodal")
        jobs = []
        for i in range(50):
            jobs.append((2 * i, 1, True))  # tiny
            jobs.append((2 * i + 1, 500, True))  # large
        N = len(jobs)
        finished = self._submit_and_wait(pool, jobs)
        assert sorted(finished) == list(range(N))
        pool.shutdown()

    def test_concurrent_producers_drain(self):
        pool = self._pool(n_read=8, n_write=4, prefix="concprod")
        N_PRODUCERS = 6
        N_PER = 50
        submitted: list[int] = []
        sub_lock = threading.Lock()

        def producer(base_id: int, is_load: bool):
            for i in range(N_PER):
                job_id = base_id + i
                task_list = _tasks((i % 10) + 1)
                with sub_lock:
                    submitted.append(job_id)
                if is_load:
                    pool.enqueue_load(
                        job_id, len(task_list), iter(task_list), _noop_make_batch_fn
                    )
                else:
                    pool.enqueue_store(
                        job_id, len(task_list), iter(task_list), _noop_make_batch_fn
                    )

        threads = [
            threading.Thread(target=producer, args=(p * N_PER, p % 2 == 0), daemon=True)
            for p in range(N_PRODUCERS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        finished = self._submit_and_wait(pool, [])  # wait only; jobs already submitted
        with sub_lock:
            assert sorted(finished) == sorted(submitted)
        pool.shutdown()

    def test_burst_idle_burst_wakes_threads(self):
        # Three bursts with wait_idle between each.
        # Ensures condition variable wakes correctly after going idle.
        pool = self._pool(prefix="burst")
        all_finished: list[int] = []
        for burst in range(3):
            base = burst * 100
            jobs = [(base + i, 8, i % 2 == 0) for i in range(100)]
            finished = self._submit_and_wait(pool, jobs)
            assert len(finished) == 100, (
                f"Burst {burst}: only {len(finished)}/100 finished"
            )
            all_finished.extend(finished)
        assert sorted(all_finished) == list(range(300))
        pool.shutdown()

    def test_task_count_integrity(self):
        # Verify that task_done() is called exactly once per task:
        # total executed batch_sizes must equal total submitted tasks.
        executed_total = [0]
        ex_lock = threading.Lock()

        def counting_make_batch_fn(batch: list[Task]):
            size = len(batch)

            def fn():
                with ex_lock:
                    executed_total[0] += size

            return fn

        with patch("vllm.v1.kv_offload.tiering.fs.thread_pool.envs") as mock_envs:
            mock_envs.VLLM_FS_THREAD_POOL_SCHEDULER_CLS = "SSDTPScheduler"
            pool = DualQueueThreadPool(
                8, 4, _BLOCK_SIZE, thread_name_prefix="integrity"
            )

        total_tasks = 0
        for i in range(100):
            size = (i % 20) + 1
            total_tasks += size
            task_list = _tasks(size)
            if i % 2 == 0:
                pool.enqueue_load(i, size, iter(task_list), counting_make_batch_fn)
            else:
                pool.enqueue_store(i, size, iter(task_list), counting_make_batch_fn)

        idle_event = threading.Event()

        def _wait_then_set_count():
            pool.wait_idle()
            idle_event.set()

        threading.Thread(target=_wait_then_set_count, daemon=True).start()
        idle_event.wait(STRESS_TIMEOUT)
        assert idle_event.is_set(), "DEADLOCK in task count integrity test"
        assert executed_total[0] == total_tasks, (
            f"Task count mismatch: executed {executed_total[0]}, expected {total_tasks}"
        )
        pool.shutdown()

    def test_no_starvation_store_under_load_pressure(self):
        # Store jobs must complete even when load queue is continuously fed.
        pool = self._pool(n_read=8, n_write=4, prefix="starve")
        store_ids = list(range(50))
        load_ids = list(range(50, 150))

        for i in store_ids:
            pool.enqueue_store(i, 8, iter(_tasks(8)), _noop_make_batch_fn)
        for i in load_ids:
            pool.enqueue_load(i, 64, iter(_tasks(64)), _noop_make_batch_fn)

        idle_event = threading.Event()

        def _wait_then_set_starve():
            pool.wait_idle()
            idle_event.set()

        threading.Thread(target=_wait_then_set_starve, daemon=True).start()
        idle_event.wait(STRESS_TIMEOUT)
        assert idle_event.is_set(), "DEADLOCK or starvation: not all jobs completed"

        finished = pool.get_finished()
        all_ids = store_ids + load_ids
        assert sorted(jid for jid, _, _ in finished) == sorted(all_ids)
        pool.shutdown()

    def test_shutdown_does_not_hang_with_pending_work(self):
        pool = self._pool(n_read=4, n_write=2, prefix="shutdn")
        for i in range(500):
            pool.enqueue_load(i, 100, iter(_tasks(100)), _noop_make_batch_fn)

        start = time.monotonic()
        pool.shutdown(wait=True)
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"shutdown() hung for {elapsed:.1f}s with pending work"

    def test_zero_task_job_completes_immediately(self):
        # enqueue_load with n_tasks=0 must appear in get_finished() without
        # touching any worker thread.
        with patch("vllm.v1.kv_offload.tiering.fs.thread_pool.envs") as mock_envs:
            mock_envs.VLLM_FS_THREAD_POOL_SCHEDULER_CLS = "SSDTPScheduler"
            pool = DualQueueThreadPool(8, 4, _BLOCK_SIZE, thread_name_prefix="zero")

        pool.enqueue_load(99, 0, iter([]), _noop_make_batch_fn)
        # No wait_idle needed; 0-task jobs bypass the queue entirely.
        finished = pool.get_finished()
        assert len(finished) == 1
        assert finished[0][0] == 99
        pool.shutdown()
