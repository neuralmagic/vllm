# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit, integration, and stress tests for the filesystem-tier I/O plumbing
below FileSystemTierManager:
  - vllm.fs_io_C (csrc/fs_io.cpp): batch_store_block, batch_load_block,
    get_status_snapshot.
  - vllm.v1.kv_offload.tiering.fs.thread_pool: DualQueueThreadPool, JobState,
    BatchResultsTracker.
  - vllm.v1.kv_offload.tiering.fs.thread_pool_deque: TPDequeBalancedBatch.

Sections:
  1. Failed keys reported correctly.
  2. Results correctly drained for long-running threads.
  3. TPDequeBalancedBatch.
  4. Stress tests.
"""

import itertools
import os
import threading
import time

import numpy as np
import pytest

from vllm.v1.kv_offload.base import OffloadKey, make_offload_key
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool, Task
from vllm.v1.kv_offload.tiering.fs.thread_pool_deque import TPDequeBalancedBatch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def key(n: int) -> OffloadKey:
    return make_offload_key(n.to_bytes(8, "big"), 0)


def _identity_make_batch_fn(batch):
    return batch


# ---------------------------------------------------------------------------
# Section 1: Failed keys reported correctly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fail_indices", [set(), {0}, {2}, {4}, {0, 2, 4}, {0, 1, 2, 3, 4}]
)
def test_batch_store_block_c_ext_reports_failed_indices(tmp_path, fail_indices):
    """fs_io_C.batch_store_block must mark status[i]=1 only for the indices
    that actually failed -- regardless of whether the failure is first,
    middle, last, or scattered -- and must still write every other block."""
    try:
        from vllm.fs_io_C import batch_store_block as batch_store_block_C
    except ImportError:
        pytest.skip("fs_io_C extension not built")

    n = 5
    tmp_paths, dest_paths, buffers = [], [], []
    for i in range(n):
        tp = tmp_path / f"t{i}.tmp"
        dp = tmp_path / f"d{i}.bin"
        if i in fail_indices:
            # Pre-create the temp file so O_CREAT|O_EXCL deterministically
            # fails with EEXIST for this index only.
            tp.write_bytes(b"")
        tmp_paths.append(str(tp))
        dest_paths.append(str(dp))
        buffers.append(bytes([i]) * 16)

    status = np.full(n, -1, dtype=np.int8)
    if fail_indices:
        with pytest.raises(OSError):
            batch_store_block_C(tmp_paths, dest_paths, buffers, False, status)
    else:
        batch_store_block_C(tmp_paths, dest_paths, buffers, False, status)

    assert set(np.nonzero(status == 1)[0].tolist()) == fail_indices
    assert set(np.nonzero(status == 0)[0].tolist()) == set(range(n)) - fail_indices
    assert -1 not in status.tolist(), "every status entry must be resolved"
    for i in range(n):
        if i not in fail_indices:
            assert os.path.exists(dest_paths[i])


@pytest.mark.parametrize("fail_indices", [{0}, {4}, {0, 2, 4}])
def test_batch_load_block_c_ext_reports_failed_indices(tmp_path, fail_indices):
    """Same contract as above, for batch_load_block: only the blocks whose
    source file is missing must be marked failed; the rest must load intact."""
    try:
        from vllm.fs_io_C import batch_load_block as batch_load_block_C
    except ImportError:
        pytest.skip("fs_io_C extension not built")

    n = 5
    block_size = 16
    paths = []
    for i in range(n):
        p = tmp_path / f"s{i}.bin"
        if i not in fail_indices:
            p.write_bytes(bytes([i]) * block_size)
        paths.append(str(p))  # missing file -> deterministic ENOENT

    buffers = [bytearray(block_size) for _ in range(n)]
    status = np.full(n, -1, dtype=np.int8)
    with pytest.raises(OSError):
        batch_load_block_C(paths, buffers, False, status)

    assert set(np.nonzero(status == 1)[0].tolist()) == fail_indices
    assert -1 not in status.tolist()
    for i in range(n):
        if i not in fail_indices:
            assert bytes(buffers[i]) == bytes([i]) * block_size


def test_dual_queue_thread_pool_reports_failed_keys_for_arbitrary_pattern():
    """DualQueueThreadPool + JobState must report exactly the keys whose task
    failed, independent of where the failure falls within the batch."""
    pool = DualQueueThreadPool(n_read_threads=2, n_write_threads=1)
    try:
        keys = [key(i) for i in range(6)]
        tasks = [Task(1, k, "dummy", 0) for k in keys]
        fail_idx = {1, 4}

        def make_batch_fn(batch):
            def fn(results):
                for i in range(len(batch)):
                    results[i] = 1 if i in fail_idx else 0

            return fn

        pool.enqueue_load(job_id=1, tasks=tasks, make_batch_fn=make_batch_fn)
        pool.wait_idle()
        finished = list(pool.get_finished())
        assert len(finished) == 1
        job_id, success, _, failed_keys = finished[0]
        assert job_id == 1
        assert not success
        assert set(failed_keys) == {keys[1], keys[4]}
    finally:
        pool.shutdown(wait=True)


def test_dual_queue_thread_pool_reports_success_with_no_failed_keys():
    """A fully successful job must report success=True and no failed keys."""
    pool = DualQueueThreadPool(n_read_threads=1, n_write_threads=1)
    try:
        keys = [key(i) for i in range(3)]
        tasks = [Task(1, k, "dummy", 0) for k in keys]

        def make_batch_fn(batch):
            def fn(results):
                results[:] = 0

            return fn

        pool.enqueue_store(job_id=1, tasks=tasks, make_batch_fn=make_batch_fn)
        pool.wait_idle()
        (finished_job,) = list(pool.get_finished())
        job_id, success, _, failed_keys = finished_job
        assert job_id == 1
        assert success
        assert failed_keys == []
    finally:
        pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Section 2: Results correctly drained for long-running threads
# ---------------------------------------------------------------------------


def test_get_status_snapshot_only_returns_resolved_prefix():
    """get_status_snapshot must return only the contiguous resolved prefix
    starting at `start`, never reading past a still-pending (-1) entry --
    the only race-free way to observe a status array a worker thread may
    still be writing to."""
    try:
        from vllm.fs_io_C import get_status_snapshot
    except ImportError:
        pytest.skip("fs_io_C extension not built")

    status = np.array([0, 1, 0, -1, -1], dtype=np.int8)
    assert get_status_snapshot(status, 0) == [0, 1, 0]
    assert get_status_snapshot(status, 3) == []  # index 3 still pending

    status[3] = 1  # the "long running" thread finally finishes index 3
    assert get_status_snapshot(status, 3) == [1]
    status[4] = 0
    assert get_status_snapshot(status, 3) == [1, 0]

    with pytest.raises(ValueError):
        get_status_snapshot(status, -1)
    with pytest.raises(ValueError):
        get_status_snapshot(status, len(status) + 1)
    with pytest.raises(ValueError):
        get_status_snapshot(np.array([0, 1], dtype=np.int16), 0)  # wrong itemsize


def test_pump_drains_fast_job_while_slow_job_shares_a_worker_thread():
    """Two jobs submitted back-to-back onto a single-threaded queue may be
    batched onto the same worker thread call (TPDequeBalancedBatch) -- or, at
    worst, processed sequentially by the sole thread. Either way, a slow
    block belonging to job B must not prevent job A's already-resolved
    result from being drained and reported."""
    pool = DualQueueThreadPool(n_read_threads=1, n_write_threads=0)
    gate = threading.Event()
    try:

        def make_batch_fn(batch):
            def fn(results):
                for i, t in enumerate(batch):
                    if t.job_id == 2:
                        gate.wait(timeout=5.0)  # job B is the "long running" one
                    results[i] = 0

            return fn

        job_a_tasks = [Task(1, key(0), "dummy", 0), Task(1, key(1), "dummy", 0)]
        job_b_tasks = [Task(2, key(2), "dummy", 0)]

        pool.enqueue_load(job_id=1, tasks=job_a_tasks, make_batch_fn=make_batch_fn)
        pool.enqueue_load(job_id=2, tasks=job_b_tasks, make_batch_fn=make_batch_fn)

        deadline = time.monotonic() + 5.0
        finished_ids: set[int] = set()
        while 1 not in finished_ids and time.monotonic() < deadline:
            for job_id, success, _, _ in pool.get_finished():
                finished_ids.add(job_id)
                assert success
            time.sleep(0.005)

        assert 1 in finished_ids, "job A's result was not drained while job B was slow"
        assert 2 not in finished_ids, "job B should still be blocked on the gate"

        gate.set()
        pool.wait_idle()
        for job_id, success, _, _ in pool.get_finished():
            finished_ids.add(job_id)
            assert success
        assert finished_ids == {1, 2}
    finally:
        gate.set()
        pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Section 3: TPDequeBalancedBatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_tasks", "n_threads", "expected_sizes"),
    [
        (7, 3, [3, 2, 2]),
        (6, 3, [2, 2, 2]),
        (2, 5, [1, 1, 0, 0, 0]),
        (0, 3, [0, 0, 0]),
        (1, 1, [1]),
    ],
)
def test_balanced_batch_splits_evenly(n_tasks, n_threads, expected_sizes):
    q = TPDequeBalancedBatch(n_threads)
    q.submit(list(range(n_tasks)), _identity_make_batch_fn)
    assert [len(sub) for sub in q._qs] == expected_sizes
    assert len(q) == n_tasks


def test_balanced_batch_fetch_round_robins_and_wraps():
    q = TPDequeBalancedBatch(n_threads=3)
    q.submit(list(range(6)), _identity_make_batch_fn)  # -> [2, 2, 2]
    batches = [q.fetch()[1] for _ in range(3)]
    assert batches == [[0, 1], [2, 3], [4, 5]]
    assert len(q) == 0

    # Cursor wrapped back to 0; 2 tasks across 3 threads -> one per thread,
    # so they land in separate per-thread queues and need two fetches.
    q.submit([6, 7], _identity_make_batch_fn)
    assert q.fetch()[1] == [6]
    assert q.fetch()[1] == [7]


def test_balanced_batch_len_tracks_pending_across_submit_and_fetch():
    q = TPDequeBalancedBatch(n_threads=2)
    q.submit([1, 2, 3], _identity_make_batch_fn)
    q.submit([4, 5], _identity_make_batch_fn)
    assert len(q) == 5
    q.fetch()
    q.fetch()
    assert len(q) == 0


def test_balanced_batch_clear_resets_state():
    q = TPDequeBalancedBatch(n_threads=2)
    q.submit([1, 2, 3, 4], _identity_make_batch_fn)
    q.clear()
    assert len(q) == 0
    assert q._fetch_cursor == 0
    assert q._make_batch_fn is None
    q.submit([9], lambda batch: batch)  # a different fn is fine post-clear


def test_balanced_batch_rejects_inconsistent_make_batch_fn():
    """Balanced batching merges jobs across submit() calls onto shared
    per-thread queues, so every submit must agree on the same make_batch_fn."""
    q = TPDequeBalancedBatch(n_threads=2)
    q.submit([1, 2], _identity_make_batch_fn)
    with pytest.raises(AssertionError):
        q.submit([3, 4], lambda batch: batch)


def test_balanced_batch_empty_is_falsy():
    q = TPDequeBalancedBatch(n_threads=4)
    assert len(q) == 0
    assert not q


# ---------------------------------------------------------------------------
# Section 4: Stress tests
# ---------------------------------------------------------------------------


def test_thread_pool_stress_many_concurrent_jobs_no_lost_or_duplicated_results():
    """Hammer DualQueueThreadPool with many concurrent jobs from many
    submitter threads; every job's completion must be reported exactly once,
    with correct success/failed_keys accounting, and no deadlock."""
    pool = DualQueueThreadPool(n_read_threads=8, n_write_threads=8)
    n_submitters = 16
    jobs_per_submitter = 50
    tasks_per_job = 5
    fail_every_nth = 7

    counter = itertools.count()
    counter_lock = threading.Lock()

    def make_batch_fn(batch):
        def fn(results):
            for i in range(len(batch)):
                with counter_lock:
                    n = next(counter)
                results[i] = 1 if n % fail_every_nth == 0 else 0

        return fn

    job_ids: list[int] = []
    jobs_lock = threading.Lock()

    def submitter(base: int) -> None:
        for j in range(jobs_per_submitter):
            job_id = base * jobs_per_submitter + j
            tasks = [
                Task(job_id, key(job_id * 1000 + t), "dummy", 0)
                for t in range(tasks_per_job)
            ]
            with jobs_lock:
                job_ids.append(job_id)
            enqueue = pool.enqueue_store if j % 2 == 0 else pool.enqueue_load
            enqueue(job_id=job_id, tasks=tasks, make_batch_fn=make_batch_fn)

    threads = [
        threading.Thread(target=submitter, args=(i,)) for i in range(n_submitters)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    try:
        finished: dict[int, tuple[bool, list]] = {}
        deadline = time.monotonic() + 30.0
        while len(finished) < len(job_ids) and time.monotonic() < deadline:
            for job_id, success, _, failed_keys in pool.get_finished():
                assert job_id not in finished, f"job {job_id} reported twice"
                finished[job_id] = (success, failed_keys)
            time.sleep(0.005)
    finally:
        pool.shutdown(wait=True)

    missing = len(job_ids) - len(finished)
    assert missing == 0, f"lost {missing} job completions under load"
    for job_id in job_ids:
        success, failed_keys = finished[job_id]
        assert success == (len(failed_keys) == 0)


def test_balanced_batch_stress_concurrent_submit_and_fetch_conserves_tasks():
    """TPDequeBalancedBatch is only used behind DualQueueThreadPool's single
    condition-variable lock; stress it the same way -- many threads
    submitting and fetching under one external lock -- and verify every
    submitted task is fetched exactly once."""
    n_threads = 6
    q = TPDequeBalancedBatch(n_threads)
    lock = threading.Lock()
    stop = threading.Event()

    submitted_total = 2000
    submitted_counter = itertools.count()
    fetched: list[int] = []

    def submitter() -> None:
        for _ in range(submitted_total // 10):
            n = next(submitted_counter)
            if n >= submitted_total:
                return
            batch = [n]
            with lock:
                q.submit(batch, _identity_make_batch_fn)

    def fetcher() -> None:
        while not stop.is_set() or len(q):
            with lock:
                if len(q) == 0:
                    batch = None
                else:
                    _, batch = q.fetch()
            if batch is None:
                time.sleep(0.0005)
                continue
            fetched.extend(batch)

    submitters = [threading.Thread(target=submitter) for _ in range(10)]
    fetchers = [threading.Thread(target=fetcher) for _ in range(n_threads)]
    for t in fetchers:
        t.start()
    for t in submitters:
        t.start()
    for t in submitters:
        t.join()
    stop.set()
    for t in fetchers:
        t.join(timeout=10.0)

    assert sorted(fetched) == list(range(submitted_total)), (
        "tasks were lost or duplicated under concurrent submit/fetch"
    )
