# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Thread pool:
    Two queues (load, store) and two sets of threads:
      - Load-priority threads: drain the load queue first, then the store queue.
      - Store-priority threads: drain the store queue first, then the load queue.
    Load jobs are enqueued to the load queue; store jobs to the store queue.

Design notes — per-role ThreadWakers:
    Each thread role (READ / FAST / WRITE for SSDTPScheduler; load-priority /
    store-priority for NoBatch/Batch) gets its own ThreadWaker.  Workers block
    exclusively on their role's waker.wait(), so waker.notify() calls from
    submit() can ONLY wake threads eligible to process the incoming job.

    The scheduler owns the wakers: make_wakers() creates them, and submit()
    calls notify() directly.  The thread pool only stores per-thread waker refs
    for shutdown (waker.stop() delivery).  No token routing lives in the pool.

    ThreadWaker wraps a SimpleQueue backed by a POSIX semaphore: wait() parks
    the thread at the C level (GIL fully released).  notify() wakes exactly one
    waiting thread; if all threads are busy the semaphore counter increments so
    no wakeup is ever lost.

    Idle detection uses a dedicated threading.Condition (_idle_cond) that is
    only acquired by the main thread (wait_idle) and workers when a job
    finishes.  This separates the hot-path scheduler lock from the rarely
    used idle-wait path.
"""

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.tiering.base import JobId
from vllm.v1.kv_offload.tiering.fs.scheduler import (
    SCHEDULER_CLASS_MAPPING,
    ThreadWaker,
    TPScheduler,
)

logger = init_logger(__name__)


@dataclass
class Task:
    """
    I/O Task inputs
    """

    key: OffloadKey
    path: str
    offset: int


class JobState:
    """
    Thread-safe completion tracker for a set of per-block I/O tasks.

    Each task calls task_done(success) when it finishes.
    """

    __slots__ = (
        "_job_id",
        "_n_tasks",
        "_completed",
        "_success",
        "_transfer_time",
        "_lock",
    )

    def __init__(self, job_id: JobId, n_tasks: int) -> None:
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._transfer_time = 0.0
        self._lock = threading.Lock()

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def task_done(
        self, batch_size: int, success: bool, transfer_time: float
    ) -> tuple[bool, bool, float]:
        """Returns if job completed and success flag"""
        with self._lock:
            self._completed += batch_size
            self._transfer_time += transfer_time
            if not success:
                self._success = False
            return self._completed == self._n_tasks, self._success, self._transfer_time


class DualQueueThreadPool:
    """
    Thread pool with two task queues (load and store) and two thread groups.

    Load-priority threads drain the load queue first, then fall back to the
    store queue.  Store-priority threads do the reverse.

    Workers block on their role-specific SimpleQueue[_WORK | _STOP]:
      - submit() returns {role → n_tokens}; the pool puts tokens into the
        matching role queue — only eligible threads are woken.
      - Each get() wakes exactly one thread via a POSIX semaphore; no GIL
        storm, no spurious cross-role wakeups.
      - Idle detection is a separate Condition so wait_idle() never wakes
        worker threads.
    """

    def __init__(
        self,
        n_read_threads: int,
        n_write_threads: int,
        block_size: int,
        thread_name_prefix: str = "fs_secondary_tier",
    ) -> None:
        self._n_read_threads = n_read_threads
        self._n_write_threads = n_write_threads

        # Guards scheduler internal state (submit / has_work / fetch_work).
        # Not held during the actual I/O fn() call.
        self._sched_lock = threading.Lock()

        # Idle detection — only used by wait_idle() and job-completion paths.
        self._idle_lock = threading.Lock()
        self._idle_cond = threading.Condition(self._idle_lock)
        self._inflight_jobs = 0  # guarded by _idle_cond

        self._stop = False
        self._finished_q: deque[tuple[JobId, bool, float]] = deque()

        assert self.total_threads > 0, "ThreadPool needs at least one thread"

        scheduler_cls = SCHEDULER_CLASS_MAPPING[envs.VLLM_FS_THREAD_POOL_SCHEDULER_CLS]
        self._scheduler: TPScheduler = scheduler_cls(
            n_read_threads, n_write_threads, block_size
        )

        # make_wakers() initialises the scheduler's internal role→waker mapping
        # and returns it so make_threads() can hand each thread its own waker.
        wakers = self._scheduler.make_wakers()

        # (thread, its_waker) pairs — used during shutdown to call waker.stop()
        # for the exact waker each thread is blocking on.
        thread_waker_pairs = self._scheduler.make_threads(
            self._worker, thread_name_prefix, wakers
        )
        self._threads: list[threading.Thread] = [t for t, _ in thread_waker_pairs]
        self._thread_wakers: list[ThreadWaker] = [w for _, w in thread_waker_pairs]

    @property
    def total_threads(self) -> int:
        return self._n_read_threads + self._n_write_threads

    def _batch_tasks(
        self,
        tasks: list[Task],
        n_threads: int,
    ) -> Iterator[list[Task]]:
        """
        Batch tasks so that the request's tasks are split evenly across the
        n_threads.
        """
        assert n_threads > 0

        n_tasks = len(tasks)
        q, r = divmod(n_tasks, n_threads)
        batch_sizes = [q + 1 if i < r else q for i in range(n_threads)]
        assert sum(batch_sizes) == n_tasks
        start = 0
        for bs in batch_sizes[: min(n_tasks, n_threads)]:
            yield tasks[start : start + bs]
            start += bs

    def _enqueue(
        self,
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
        job_id: JobId,
        tasks: Iterable[Task],
        n_tasks: int,
        is_load: bool,
    ) -> None:
        """Batch `tasks` and append (fn, state, batch_size) entries to `queue`."""
        if n_tasks == 0:
            self._finished_q.append((job_id, True, 0.0))
            return
        state = JobState(job_id, n_tasks)
        task_lst = list(tasks)  # Materialize tasks outside locks
        assert len(task_lst) == n_tasks, "Unaccounted tasks"

        # Increment inflight before submit so wait_idle() can't return early.
        with self._idle_cond:
            self._inflight_jobs += 1

        # submit() routes _WORK tokens into the right role queues internally.
        with self._sched_lock:
            self._scheduler.submit(state, make_batch_fn, task_lst, is_load)

    def enqueue_load(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Task],
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
    ) -> None:
        """Enqueue load tasks for a job (high-priority for load-priority threads)."""
        self._enqueue(
            make_batch_fn,
            job_id,
            tasks,
            n_tasks=n_tasks,
            is_load=True,
        )

    def enqueue_store(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Task],
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
    ) -> None:
        """Enqueue store tasks for a job (high-priority for store-priority threads)."""
        self._enqueue(
            make_batch_fn,
            job_id,
            tasks,
            n_tasks=n_tasks,
            is_load=False,
        )

    def get_finished(self) -> list[tuple[JobId, bool, float]]:
        # No lock needed: deque is thread-safe for concurrent append/popleft,
        # and the manager is the sole popper.
        jobs = []
        while self._finished_q:
            jobs.append(self._finished_q.popleft())
        return jobs

    def wait_idle(self) -> None:
        """Block until there are no in-flight jobs.

        After this returns, every submitted job has had its last task
        finish, so no worker thread is still copying data. Note:
        completed jobs may still be sitting in ``_finished_q`` waiting
        for ``get_finished()`` to drain them.
        """
        with self._idle_cond:
            self._idle_cond.wait_for(lambda: self._inflight_jobs == 0)

    def shutdown(self, wait: bool = True) -> None:
        with self._sched_lock:
            self._stop = True
            self._scheduler.clear()

        # Reset inflight so wait_idle() returns if called after shutdown.
        with self._idle_cond:
            self._inflight_jobs = 0
            self._idle_cond.notify_all()

        # Call stop() on each thread's own waker so only that thread receives
        # the signal.  This guarantees each thread exits exactly once regardless
        # of whether role wakers are shared or distinct.
        for waker in self._thread_wakers:
            waker.stop()

        if wait:
            for t in self._threads:
                t.join()

    def _worker(self, args: Any, my_waker: ThreadWaker) -> None:
        """Worker loop: block on this thread's role waker, fetch work, execute I/O."""
        while True:
            # Blocks at C level (POSIX sem_wait); GIL fully released.
            # Only notify()/stop() on this role's waker can wake this thread.
            if not my_waker.wait():
                return  # stop() was called — exit cleanly

            # Verify there is work for our role.  With per-role wakers this
            # check should almost always pass; it guards the rare race where
            # another thread of the same role consumed the work between the
            # notify() and this wait().
            with self._sched_lock:
                if not self._scheduler.has_work(args):
                    continue
                result = self._scheduler.fetch_work(args)

            if result is None:
                # Defensive: fetch_work returned nothing despite has_work=True.
                continue

            fn, batch_size, state = result
            try:
                start_time = time.monotonic()
                fn()
                transfer_time = time.monotonic() - start_time
                job_finished, success, total_time = state.task_done(
                    batch_size, True, transfer_time
                )
            except Exception as exc:
                transfer_time = time.monotonic() - start_time
                logger.error(
                    "Job %s block I/O failed: %s",
                    state.job_id,
                    exc,
                )
                job_finished, success, total_time = state.task_done(
                    batch_size, False, transfer_time
                )

            if job_finished:
                # Append before decrementing inflight so get_finished() sees
                # the result as soon as wait_idle() returns.
                self._finished_q.append((state.job_id, success, total_time))
                with self._idle_cond:
                    self._inflight_jobs -= 1
                    if self._inflight_jobs == 0:
                        self._idle_cond.notify_all()
