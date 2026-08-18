# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Thread pool:
    Two queues (load, store) and two sets of threads:
      - Load-priority threads: drain the load queue first, then the store queue.
      - Store-priority threads: drain the store queue first, then the load queue.
    Load jobs are enqueued to the load queue; store jobs to the store queue.
"""

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.tiering.base import JobId
from vllm.v1.kv_offload.tiering.fs.timing_debug import TimingRecorder

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
        "_kind",
        "_enqueue_time",
        "_dequeued_count",
        "_first_dequeue_time",
        "_last_dequeue_time",
    )

    def __init__(
        self, job_id: JobId, n_tasks: int, enqueue_time: float, kind: str = ""
    ) -> None:
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._transfer_time = 0.0
        self._lock = threading.Lock()
        # Diagnostic-only fields for TimingRecorder (see timing_debug.py).
        self._kind = kind
        self._enqueue_time = enqueue_time
        self._dequeued_count = 0
        self._first_dequeue_time: float | None = None
        self._last_dequeue_time: float | None = None

    @property
    def job_id(self) -> JobId:
        return self._job_id

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def enqueue_time(self) -> float:
        return self._enqueue_time

    @property
    def n_batches_dequeued(self) -> int:
        with self._lock:
            return self._dequeued_count

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

    def record_dequeue(self, t: float) -> tuple[float, int]:
        """Record that one of this job's batches was picked up by a worker
        at time `t`. Returns (queue_wait, batch_no), where batch_no is the
        0-indexed pickup order among this job's batches."""
        with self._lock:
            batch_no = self._dequeued_count
            self._dequeued_count += 1
            if self._first_dequeue_time is None:
                self._first_dequeue_time = t
            if self._last_dequeue_time is None or t > self._last_dequeue_time:
                self._last_dequeue_time = t
            return t - self._enqueue_time, batch_no

    def pickup_span(self) -> tuple[float, float]:
        """Returns (queue_wait_first, pickup_spread) once the job has
        finished, i.e. once every batch has been dequeued at least once."""
        with self._lock:
            first = self._first_dequeue_time
            last = self._last_dequeue_time
            assert first is not None and last is not None, (
                "pickup_span() called before any batch was dequeued"
            )
            return first - self._enqueue_time, last - first


class DualQueueThreadPool:
    """
    Thread pool with two task queues (load and store) and two thread groups.

    Load-priority threads drain the load queue first, then fall back to the
    store queue.  Store-priority threads do the reverse.  Both queues share
    a single condition variable.
    """

    def __init__(
        self,
        n_read_threads: int,
        n_write_threads: int,
        thread_name_prefix: str = "fs_secondary_tier",
    ) -> None:
        self._n_read_threads = n_read_threads
        self._n_write_threads = n_write_threads
        self._load_q: deque = deque()
        self._store_q: deque = deque()
        self._condition = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []
        self._finished_q: deque[tuple[JobId, bool, float]] = deque()
        self._inflight_jobs = 0  # guarded by _condition
        self._timing = TimingRecorder(envs.VLLM_KV_OFFLOAD_FS_TIMING_LOG)

        assert self.total_threads > 0, "ThreadPool needs at least one thread"

        for i in range(self._n_read_threads):
            t = threading.Thread(
                target=self._worker,
                args=(True,),
                name=f"{thread_name_prefix}_l{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(self._n_write_threads):
            t = threading.Thread(
                target=self._worker,
                args=(False,),
                name=f"{thread_name_prefix}_s{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

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
        queue: deque,
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
        job_id: JobId,
        tasks: Iterable[Task],
        n_tasks: int,
        n_threads: int,
        kind: str,
    ) -> None:
        """Batch `tasks` and append (fn, state, batch_size) entries to `queue`."""
        t_enqueue = time.monotonic()
        if n_tasks == 0:
            self._finished_q.append((job_id, True, 0.0))
            return
        state = JobState(job_id, n_tasks, t_enqueue, kind)
        task_lst = list(tasks)  # Materialize tasks out of self._condition
        assert len(task_lst) == n_tasks, "Unaccounted tasks"
        n_batches = 0
        timing_enabled = self._timing.enabled
        with self._condition:
            self._inflight_jobs += 1
            queue_depth_before = len(queue) if timing_enabled else 0
            for batch in self._batch_tasks(task_lst, n_threads):
                queue.append((make_batch_fn(batch), len(batch), state))
                n_batches += 1
            self._condition.notify(n_batches)
        if timing_enabled:
            self._timing.record(
                "E", t_enqueue, job_id, kind, n_tasks, n_batches, queue_depth_before
            )

    def enqueue_load(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Task],
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
    ) -> None:
        """Enqueue load tasks for a job (high-priority for load-priority threads)."""

        self._enqueue(
            self._load_q,
            make_batch_fn,
            job_id,
            tasks,
            n_tasks=n_tasks,
            n_threads=self._n_read_threads
            if self._n_read_threads > 0
            else self.total_threads,
            kind="load",
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
            self._store_q,
            make_batch_fn,
            job_id,
            tasks,
            n_tasks=n_tasks,
            n_threads=self._n_write_threads
            if self._n_write_threads > 0
            else self.total_threads,
            kind="store",
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
        with self._condition:
            self._condition.wait_for(lambda: self._inflight_jobs == 0)

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._stop = True
            self._load_q.clear()
            self._store_q.clear()
            # Cancelled tasks will not decrement _inflight_jobs; reset it so a
            # subsequent wait_idle() returns instead of hanging.
            self._inflight_jobs = 0
            self._condition.notify_all()
        if wait:
            for t in self._threads:
                t.join()
        self._timing.close()

    def _worker(self, load_priority: bool) -> None:
        thread_name = threading.current_thread().name
        timing_enabled = self._timing.enabled
        # Wait for tasks, process from primary queue first, fall back to secondary.
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop or self._load_q or self._store_q
                )
                if self._stop:
                    return
                primary = self._load_q if load_priority else self._store_q
                secondary = self._store_q if load_priority else self._load_q
                fn, batch_size, state = (
                    primary.popleft() if primary else secondary.popleft()
                )
                start_time = time.monotonic()
                if timing_enabled:
                    queue_wait, batch_no = state.record_dequeue(start_time)
                    queue_depth_after = len(self._load_q) + len(self._store_q)
            if timing_enabled:
                self._timing.record(
                    "D",
                    start_time,
                    state.job_id,
                    thread_name,
                    batch_no,
                    batch_size,
                    queue_wait,
                    queue_depth_after,
                )
            try:
                fn()
                transfer_time = time.monotonic() - start_time
                batch_success = True
                job_finished, success, total_time = state.task_done(
                    batch_size, True, transfer_time
                )
            except Exception as exc:
                transfer_time = time.monotonic() - start_time
                batch_success = False
                logger.error(
                    "Job %s block I/O failed: %s",
                    state.job_id,
                    exc,
                )
                job_finished, success, total_time = state.task_done(
                    batch_size, False, transfer_time
                )

            if timing_enabled:
                self._timing.record(
                    "F",
                    time.monotonic(),
                    state.job_id,
                    thread_name,
                    batch_no,
                    transfer_time,
                    int(batch_success),
                )

            if job_finished:
                if timing_enabled:
                    job_finish_time = time.monotonic()
                    queue_wait_first, pickup_spread = state.pickup_span()
                    n_batches = state.n_batches_dequeued
                    first_dequeue_time = state.enqueue_time + queue_wait_first
                    span = job_finish_time - first_dequeue_time
                    parallel_efficiency = (
                        total_time / (span * n_batches)
                        if span > 0 and n_batches > 0
                        else 1.0
                    )
                    self._timing.record(
                        "J",
                        job_finish_time,
                        state.job_id,
                        state.kind,
                        n_batches,
                        queue_wait_first,
                        pickup_spread,
                        span,
                        total_time,
                        f"{parallel_efficiency:.4f}",
                    )
                with self._condition:
                    self._finished_q.append((state.job_id, success, total_time))
                    self._inflight_jobs -= 1
                    self._condition.notify_all()
