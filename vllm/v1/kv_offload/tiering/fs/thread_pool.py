# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Thread pool:
    Two work queues (load, store) and two sets of threads:
      - Load-priority threads: drain the load queue first, then the store queue.
      - Store-priority threads: drain the store queue first, then the load queue.
    Load jobs are enqueued to the load queue; store jobs to the store queue.

The queues themselves, and the raw per-block I/O, live in the ``fs_io_C``
C extension (see ``csrc/fs_io.cpp``): a ``Pool`` holds a load ``WorkQueue``
and a store ``WorkQueue`` (each with its own mutex), plus a ``ResultQueue``
(its own mutex). Worker threads are still plain ``threading.Thread``
objects; each wake makes one call into ``wait_and_run()``, which releases
the GIL once and drains items, pushing one ``Result`` per finished item.
Once both queues run dry it blocks (without spinning) on a C-side condition
variable for up to a 30s idle timeout, so a steady trickle of small jobs
doesn't force a GIL round-trip between every one; ``shutdown()`` wakes any
idling worker immediately via ``request_stop()``. Job-level aggregation
(``JobState``) stays in Python and is driven by ``get_finished()`` draining
the C ``ResultQueue`` -- decoupled from, and asynchronous to, whatever the
worker threads are doing.
"""

import threading
import time
from collections import deque
from enum import Enum

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.base import JobId

try:
    from vllm.fs_io_C import (  # pyright: ignore[reportMissingImports]
        clear_work_queue,
        create_pool,
        destroy_pool,
        pop_all_results,
        push_load,
        push_store,
        queue_nonempty,
        request_stop,
        wait_and_run,
    )

    _HAS_FSIO_POOL_C = True
except ImportError:
    _HAS_FSIO_POOL_C = False

logger = init_logger(__name__)

# Polling interval for wait_idle(): job completion is only discovered when
# something drains the async ResultQueue, so wait_idle() drains in a loop
# instead of being notified instantly.
_WAIT_IDLE_POLL_INTERVAL_S = 0.001


class Priority(Enum):
    READ = 1
    WRITE = 2
    WRITE_EXCL = 3
    READ_EXCL = 4


class JobState:
    """
    Completion tracker for a set of per-block I/O tasks belonging to one job.

    ``task_done()`` is only ever called from ``get_finished()``'s drain loop,
    which runs on a single thread (the scheduler thread that also enqueues
    jobs) -- so, unlike a tracker fed directly by worker threads, this needs
    no lock.
    """

    __slots__ = (
        "_job_id",
        "_n_tasks",
        "_completed",
        "_success",
        "_transfer_time",
        "_failed_indices",
    )

    def __init__(self, job_id: JobId, n_tasks: int) -> None:
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._transfer_time = 0.0
        self._failed_indices: list[int] = []

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def task_done(
        self, index: int, success: bool, transfer_time: float
    ) -> tuple[bool, bool, float, list[int]]:
        """Record one block's completion.

        Returns (job_finished, overall_success, total_transfer_time,
        failed_indices). Blocks may complete in arbitrary order across
        threads, so failures are tracked by exact index rather than as a
        "first N succeeded" prefix count.
        """
        self._completed += 1
        self._transfer_time += transfer_time
        if not success:
            self._success = False
            self._failed_indices.append(index)
        return (
            self._completed == self._n_tasks,
            self._success,
            self._transfer_time,
            self._failed_indices,
        )


class DualQueueThreadPool:
    """
    Thread pool with two work queues (load and store) and two thread groups.

    Load-priority threads drain the load queue first, then fall back to the
    store queue.  Store-priority threads do the reverse.  Both queues, and
    the results queue, live in the ``fs_io_C`` pool (see module docstring).
    """

    def __init__(
        self,
        n_read_threads: int,
        n_write_threads: int,
        n_read_excl_threads: int,
        n_write_excl_threads: int,
        primary_kv_view: memoryview,
        block_size: int,
        use_o_direct: bool = True,
        thread_name_prefix: str = "fs_secondary_tier",
    ) -> None:
        if not _HAS_FSIO_POOL_C:
            raise ImportError(
                "DualQueueThreadPool requires the vllm.fs_io_C extension "
                "(work-stealing I/O queues live in C++); it was not built."
            )
        self._pool = create_pool(primary_kv_view, block_size, use_o_direct)
        self._condition = threading.Condition(threading.Lock())
        self._stop = False
        self._shutdown_done = False
        self._threads: list[threading.Thread] = []
        self._jobs: dict[JobId, JobState] = {}
        self._finished_q: deque[tuple[JobId, bool, float, list[int]]] = deque()
        self._inflight_jobs = 0  # guarded by _condition

        for i in range(n_read_threads):
            t = threading.Thread(
                target=self._worker,
                args=(Priority.READ,),
                name=f"{thread_name_prefix}_l{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(n_write_threads):
            t = threading.Thread(
                target=self._worker,
                args=(Priority.WRITE,),
                name=f"{thread_name_prefix}_s{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(n_write_excl_threads):
            t = threading.Thread(
                target=self._worker,
                args=(Priority.WRITE_EXCL,),
                name=f"{thread_name_prefix}_se{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(n_read_excl_threads):
            t = threading.Thread(
                target=self._worker,
                args=(Priority.READ_EXCL,),
                name=f"{thread_name_prefix}_le{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def enqueue_load(
        self,
        job_id: JobId,
        paths: list[str],
        offsets: list[int],
    ) -> None:
        """Enqueue every block of a load job (high-priority for load-priority
        threads) in one shot."""
        if not paths:
            # No block will ever produce a Result for this job, so it must be
            # resolved immediately rather than left waiting forever.
            self._finished_q.append((job_id, True, 0.0, []))
            return
        self._jobs[job_id] = JobState(job_id, len(paths))
        push_load(self._pool, job_id, paths, offsets)
        with self._condition:
            self._inflight_jobs += 1
            self._condition.notify(len(paths))

    def enqueue_store(
        self,
        job_id: JobId,
        paths: list[str],
        offsets: list[int],
    ) -> None:
        """Enqueue every block of a store job (high-priority for
        store-priority threads) in one shot."""
        if not paths:
            self._finished_q.append((job_id, True, 0.0, []))
            return
        self._jobs[job_id] = JobState(job_id, len(paths))
        push_store(self._pool, job_id, paths, offsets)
        with self._condition:
            self._inflight_jobs += 1
            self._condition.notify(len(paths))

    def _drain_results(self) -> None:
        """Pull every currently-available Result out of the C ResultQueue and
        fold it into the corresponding JobState, moving finished jobs into
        ``_finished_q``. Must only be called from the single thread that also
        owns ``_jobs`` (the scheduler thread) -- see JobState's docstring."""
        for job_id, index, err, transfer_time in pop_all_results(self._pool):
            state = self._jobs.get(job_id)
            if state is None:
                # Can happen after shutdown() clears in-flight bookkeeping
                # while a worker's already-popped item is still finishing.
                continue
            success = err == 0
            job_finished, job_success, total_time, failed_indices = state.task_done(
                index, success, transfer_time
            )
            if not success:
                logger.error(
                    "Job %s block %d I/O failed (errno=%d)", job_id, index, err
                )
            if job_finished:
                del self._jobs[job_id]
                self._finished_q.append(
                    (job_id, job_success, total_time, list(failed_indices))
                )
                # No waiter cares about this via self._condition: workers
                # only wait on queue_nonempty/_stop, and wait_idle() polls
                # rather than waiting on the condition variable. The lock is
                # still needed since _inflight_jobs is read under it from
                # other threads (wait_idle(), shutdown()).
                with self._condition:
                    self._inflight_jobs -= 1

    def get_finished(self) -> list[tuple[JobId, bool, float, list[int]]]:
        """Drain the C ResultQueue and return every job that finished since
        the last call, as (job_id, success, transfer_time, failed_indices)."""
        self._drain_results()
        jobs = []
        while self._finished_q:
            jobs.append(self._finished_q.popleft())
        return jobs

    def wait_idle(self) -> None:
        """Block until there are no in-flight jobs.

        Job completion is only discovered by draining the C ResultQueue, so
        this polls (drain, check, sleep) instead of being notified instantly.
        After this returns, every submitted job has had its last block
        finish, so no worker thread is still touching the shared buffer. Note:
        completed jobs may still be sitting in ``_finished_q`` waiting for
        ``get_finished()`` to drain them.
        """
        while True:
            self._drain_results()
            with self._condition:
                if self._inflight_jobs == 0:
                    return
            time.sleep(_WAIT_IDLE_POLL_INTERVAL_S)

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            if self._shutdown_done:
                return
            self._shutdown_done = True
            self._stop = True
            clear_work_queue(self._pool)
            # Wake any worker currently idling inside wait_and_run()'s C-side
            # timeout wait -- without this it would only return once that
            # timeout elapses.
            request_stop(self._pool)
            # Cancelled tasks will not decrement _inflight_jobs; reset it so a
            # subsequent wait_idle() returns instead of hanging.
            self._inflight_jobs = 0
            self._condition.notify_all()
        if wait:
            for t in self._threads:
                t.join()
            self._jobs.clear()
            # Only safe once every worker has actually returned: destroy_pool
            # releases the pinned buffer immediately, and a worker still
            # inside wait_and_run() is touching it without the GIL. With
            # wait=False we deliberately leak the pool rather than risk a
            # use-after-free race with a still-running worker.
            destroy_pool(self._pool)

    def _worker(self, priority: Priority) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop or queue_nonempty(self._pool, priority)
                )
                if self._stop:
                    return
            wait_and_run(self._pool, priority)
