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
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.base import JobId

logger = init_logger(__name__)


@dataclass(slots=True)
class JobStats:
    """Stats snapshot produced when a job completes."""

    is_store: bool
    n_tasks: int = 0
    elapsed_ms: float = 0.0
    bytes_transferred: int = 0
    io_elapsed_s: float = 0.0


class JobState:
    """
    Thread-safe completion tracker for a set of per-block I/O tasks.

    Each task calls task_done(success) when it finishes. Records wall-clock
    elapsed time from enqueue to last-task completion.
    """

    __slots__ = (
        "_job_id",
        "_n_tasks",
        "_completed",
        "_success",
        "_lock",
        "_start_time",
        "_elapsed_ms",
        "_bytes_transferred",
        "_io_elapsed_s",
        "is_store",
    )

    def __init__(self, job_id: JobId, n_tasks: int, is_store: bool) -> None:
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._lock = threading.Lock()
        self._start_time: float = time.perf_counter()
        self._elapsed_ms: float = 0.0
        self._bytes_transferred: int = 0
        self._io_elapsed_s: float = 0.0
        self.is_store: bool = is_store

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def task_done(
        self,
        success: bool,
        task_bytes: int = 0,
        task_elapsed_s: float = 0.0,
    ) -> tuple[bool, bool]:
        """Returns if job completed and success flag"""
        with self._lock:
            self._completed += 1
            if success:
                self._bytes_transferred += task_bytes
                self._io_elapsed_s += task_elapsed_s
            else:
                self._success = False
            job_finished = self._completed == self._n_tasks
            if job_finished:
                self._elapsed_ms = (time.perf_counter() - self._start_time) * 1e3
            return job_finished, self._success

    def to_stats(self) -> JobStats:
        return JobStats(
            is_store=self.is_store,
            n_tasks=self._n_tasks,
            elapsed_ms=self._elapsed_ms,
            bytes_transferred=self._bytes_transferred,
            io_elapsed_s=self._io_elapsed_s,
        )


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
        self._load_q: deque = deque()
        self._store_q: deque = deque()
        self._condition = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []
        self._finished_q: deque[tuple[JobId, bool, JobStats]] = deque()

        for i in range(n_read_threads):
            t = threading.Thread(
                target=self._worker,
                args=(True,),
                name=f"{thread_name_prefix}_l{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(n_write_threads):
            t = threading.Thread(
                target=self._worker,
                args=(False,),
                name=f"{thread_name_prefix}_s{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def enqueue_load(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Callable],
    ) -> None:
        """Enqueue load tasks for a job (high-priority for load-priority threads)."""
        state = JobState(job_id, n_tasks, is_store=False)
        with self._condition:
            for fn in tasks:
                self._load_q.append((fn, state))
            self._condition.notify(n_tasks)

    def enqueue_store(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Callable],
    ) -> None:
        """Enqueue store tasks for a job (high-priority for store-priority threads)."""
        state = JobState(job_id, n_tasks, is_store=True)
        with self._condition:
            for fn in tasks:
                self._store_q.append((fn, state))
            self._condition.notify(n_tasks)

    def get_finished(self) -> list[tuple[JobId, bool, JobStats]]:
        jobs = []
        while self._finished_q:
            jobs.append(self._finished_q.popleft())
        return jobs

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._stop = True
            self._load_q.clear()
            self._store_q.clear()
            self._condition.notify_all()
        if wait:
            for t in self._threads:
                t.join()

    def _worker(self, load_priority: bool) -> None:
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
                task, state = primary.popleft() if primary else secondary.popleft()
            try:
                t0 = time.perf_counter()
                task_bytes = task()
                task_s = time.perf_counter() - t0
                job_finished, success = state.task_done(True, task_bytes, task_s)
            except Exception as exc:
                logger.error(
                    "Job %s block I/O failed: %s",
                    state.job_id,
                    exc,
                )
                job_finished, success = state.task_done(False)

            if job_finished:
                self._finished_q.append((state.job_id, success, state.to_stats()))
