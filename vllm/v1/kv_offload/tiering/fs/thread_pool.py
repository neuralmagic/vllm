# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Thread pool:
    Two queues (load, store) and two sets of threads:
      - Load-priority threads: drain the load queue first, then the store queue.
      - Store-priority threads: drain the store queue first, then the load queue.
    Load jobs are enqueued to the load queue; store jobs to the store queue.
"""

import errno as _errno
import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.base import JobId

logger = init_logger(__name__)


@dataclass
class TaskResult:
    """Outcome of a single per-block I/O task."""

    key: Any
    success: bool
    enospc: bool = False
    existed: bool = False  # store_block returned None (file was already present)


class JobState:
    """
    Thread-safe completion tracker for a set of per-block I/O tasks.

    Each task calls task_done(result) when it finishes.
    """

    __slots__ = ("_job_id", "_n_tasks", "_completed", "_results", "_lock")

    def __init__(self, job_id: JobId, n_tasks: int) -> None:
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._results: list[TaskResult] = []
        self._lock = threading.Lock()

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def task_done(self, result: TaskResult) -> bool:
        """Append result and return True when all tasks for this job are done."""
        with self._lock:
            self._results.append(result)
            self._completed += 1
            return self._completed == self._n_tasks

    @property
    def results(self) -> list[TaskResult]:
        return self._results


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
        self._finished_q: deque[tuple[JobId, list[TaskResult]]] = deque()

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
        tasks: Iterable[Callable],
        task_meta: list[Any],
    ) -> None:
        """Enqueue load tasks for a job (high-priority for load-priority threads)."""
        state = JobState(job_id, len(task_meta))
        with self._condition:
            for fn, meta in zip(tasks, task_meta):
                self._load_q.append((fn, meta, state))
            self._condition.notify(len(task_meta))

    def enqueue_store(
        self,
        job_id: JobId,
        tasks: Iterable[Callable],
        task_meta: list[Any],
    ) -> None:
        """Enqueue store tasks for a job (high-priority for store-priority threads)."""
        state = JobState(job_id, len(task_meta))
        with self._condition:
            for fn, meta in zip(tasks, task_meta):
                self._store_q.append((fn, meta, state))
            self._condition.notify(len(task_meta))

    def get_finished(self) -> list[tuple[JobId, list[TaskResult]]]:
        """Return completed jobs as (job_id, task_results) pairs."""
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
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop or self._load_q or self._store_q
                )
                if self._stop:
                    return
                primary = self._load_q if load_priority else self._store_q
                secondary = self._store_q if load_priority else self._load_q
                task, meta, state = (
                    primary.popleft() if primary else secondary.popleft()
                )
            try:
                ret = task()
                # store_block returns None when the file already existed;
                # load_block always returns True on success.
                existed = ret is None
                job_finished = state.task_done(
                    TaskResult(key=meta, success=True, existed=existed)
                )
            except OSError as exc:
                logger.error("Job %s block I/O failed: %s", state.job_id, exc)
                job_finished = state.task_done(
                    TaskResult(
                        key=meta,
                        success=False,
                        enospc=exc.errno == _errno.ENOSPC,
                    )
                )
            except Exception as exc:
                logger.error("Job %s block I/O failed: %s", state.job_id, exc)
                job_finished = state.task_done(TaskResult(key=meta, success=False))

            if job_finished:
                self._finished_q.append((state.job_id, state.results))
