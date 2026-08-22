# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Thread pool:
    Two queues (load, store) and two sets of threads:
      - Load-priority threads: drain the load queue first, then the store queue.
      - Store-priority threads: drain the store queue first, then the load queue.
    Load jobs are enqueued to the load queue; store jobs to the store queue.
"""

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.tiering.base import JobId

logger = init_logger(__name__)


@dataclass
class Task:
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
        self, success: bool, batch_size: int, transfer_time: float
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
        self._finished_q: deque[tuple[JobId, bool, float]] = deque()
        self._inflight_jobs = 0  # guarded by _condition
        self._idle_read_threads = n_read_threads
        self._idle_write_threads = n_write_threads

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

    @property
    def _idle_threads(self):
        return self._idle_read_threads + self._idle_write_threads

    def enqueue_load(
        self,
        job_id: JobId,
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
        n_tasks: int,
        tasks: Iterable[Task],
    ) -> None:
        pass
        self._enqueue(self._load_q, job_id, make_batch_fn, n_tasks, tasks)

    def enqueue_store(
        self,
        job_id: JobId,
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
        n_tasks: int,
        tasks: Iterable[Task],
    ) -> None:
        pass
        self._enqueue(self._store_q, job_id, make_batch_fn, n_tasks, tasks)

    @dataclass
    class ThreadWork:
        make_batch_fn: Callable[[list[Task]], Callable[[], None]]
        tasks: list[Task]
        state: JobState
        consumed: int

    def _enqueue(
        self,
        queue: deque,
        job_id: JobId,
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
        n_tasks: int,
        tasks: Iterable[Task],
    ) -> None:
        if n_tasks == 0:
            self._finished_q.append((job_id, True, 0.0))
            return
        state = JobState(job_id, n_tasks)
        task_lst = list(tasks)  # Materialize tasks out of self._condition
        assert len(task_lst) == n_tasks, "Unaccounted tasks"
        with self._condition:
            self._inflight_jobs += 1
            queue.append(self.ThreadWork(make_batch_fn, task_lst, state, 0))
            self._condition.notify(n_tasks)

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

    def _worker(self, load_priority: bool) -> None:
        # Wait for tasks, process from primary queue first, fall back to secondary.

        # return fn, state and n_tasks
        def _fetch_batch():
            # options:
            # 1. drain the entire queue!
            # 2. drain part of the queue based on the job!
            # job based draining is better - we need to parallelize as much
            # as possible!
            primary = self._load_q if load_priority else self._store_q
            secondary = self._store_q if load_priority else self._load_q
            if load_priority:
                primary, secondary = self._load_q, self._store_q
                queue = primary if primary else secondary
                idle_threads = (
                    self._idle_read_threads if primary else self._idle_threads
                )
            else:
                primary, secondary = self._store_q, self._load_q
                queue = primary if primary else secondary
                idle_threads = (
                    self._idle_write_threads if primary else self._idle_threads
                )

            # peek without popping
            work = queue[0]
            remaining = len(work.tasks) - work.consumed
            batch_size = math.ceil(remaining / idle_threads)
            fn = work.make_batch_fn(
                work.tasks[work.consumed : work.consumed + batch_size]
            )

            work.consumed += batch_size
            if work.consumed == len(work.tasks):
                queue.popleft()

            return fn, batch_size, work.state

        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop or self._load_q or self._store_q
                )
                if self._stop:
                    return
                fn, batch_size, state = _fetch_batch()

                if load_priority:
                    self._idle_read_threads -= 1
                else:
                    self._idle_write_threads -= 1
            try:
                start_time = time.monotonic()
                fn()
                transfer_time = time.monotonic() - start_time
                job_finished, success, total_time = state.task_done(
                    True, batch_size, transfer_time
                )
            except Exception as exc:
                transfer_time = time.monotonic() - start_time
                logger.error(
                    "Job %s block I/O failed: %s",
                    state.job_id,
                    exc,
                )
                job_finished, success, total_time = state.task_done(
                    False, batch_size, transfer_time
                )

            if job_finished:
                with self._condition:
                    self._finished_q.append((state.job_id, success, total_time))
                    self._inflight_jobs -= 1
                    self._condition.notify_all()

            with self._condition:
                if load_priority:
                    self._idle_read_threads += 1
                else:
                    self._idle_write_threads += 1
