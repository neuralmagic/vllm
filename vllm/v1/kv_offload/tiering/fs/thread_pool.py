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
from enum import Enum
from queue import PriorityQueue

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.tiering.base import JobId

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
        "_is_load",
        "_completed",
        "_success",
        "_transfer_time",
        "_lock",
    )

    def __init__(self, job_id: JobId, n_tasks: int, is_load: bool) -> None:
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._is_load = is_load
        self._completed = 0
        self._success = True
        self._transfer_time = 0.0
        self._lock = threading.Lock()

    @property
    def job_id(self) -> JobId:
        return self._job_id

    @property
    def is_load(self) -> bool:
        return self._is_load

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


class Role(Enum):
    LOAD = 1
    STORE = 2
    LOAD_FAST = 3  # fast lane for quick jobs
    STORE_FAST = 4  # fast lane for quick jobs


@dataclass
class WorkItem:
    state: JobState
    tasks: list[Task]
    make_batch_fn: Callable | None = None
    fn: Callable | None = None

    def is_materialized(self):
        return self.fn is not None

    def materialize(self):
        if self.is_materialized():
            return
        assert self.make_batch_fn is not None
        self.fn = self.make_batch_fn(self.tasks)
        self.make_batch_fn = None

    @property
    def n_tasks(self):
        return len(self.tasks)

    def as_work(self):
        assert self.is_materialized()
        return self.fn, self.n_tasks, self.state


class DualQueueThreadPool:
    """
    Thread pool with two task queues (load and store) and two thread groups.

    Load-priority threads drain the load queue first, then fall back to the
    store queue.  Store-priority threads do the reverse.  Both queues share
    a single condition variable.
    """

    def __init__(
        self,
        n_read_threads: int,  # batched. big enough to drive bw
        n_read_fast_threads: int,  # small; absorb small jobs
        n_write_threads: int,  # not batched. big enough to absorb writes
        n_write_fast_threads: int,  # smalle; absort small jobs
        thread_name_prefix: str = "fs_secondary_tier",
    ) -> None:
        self._n_read_threads = n_read_threads
        self._n_write_threads = n_write_threads
        self._n_read_fast_threads = n_read_fast_threads
        self._n_write_fast_threads = n_write_fast_threads
        self._load_q: deque = deque()
        self._load_fast_q: PriorityQueue = PriorityQueue()
        self._store_q: deque = deque()
        self._store_fast_q: PriorityQueue = PriorityQueue()
        self._condition = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []
        self._finished_q: deque[tuple[JobId, bool, float]] = deque()
        self._inflight_jobs = 0  # guarded by _condition

        self._inflight_load_jobs = 0
        self._inflight_load_tasks = 0
        self._inflight_store_jobs = 0
        self._inflight_store_tasks = 0

        assert self.total_threads > 0, "ThreadPool needs at least one thread"

        for i in range(self._n_read_threads):
            t = threading.Thread(
                target=self._worker,
                args=(Role.LOAD,),
                name=f"{thread_name_prefix}_l{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(self._n_write_threads):
            t = threading.Thread(
                target=self._worker,
                args=(Role.STORE,),
                name=f"{thread_name_prefix}_s{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(self._n_read_fast_threads):
            t = threading.Thread(
                target=self._worker,
                args=(Role.LOAD_FAST,),
                name=f"{thread_name_prefix}_lf{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(self._n_write_fast_threads):
            t = threading.Thread(
                target=self._worker,
                args=(Role.STORE_FAST,),
                name=f"{thread_name_prefix}_sf{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    @property
    def total_threads(self) -> int:
        return (
            self._n_read_threads
            + self._n_write_threads
            + self._n_read_fast_threads
            + self._n_write_fast_threads
        )

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

    def _batch_workitem(
        self,
        w: WorkItem,
        n_threads: int,
        materialize: bool = True,
    ):
        assert n_threads > 0
        assert not w.is_materialized()
        for b in self._batch_tasks(w.tasks, n_threads):
            wi = WorkItem(
                state=w.state, tasks=b, make_batch_fn=w.make_batch_fn, fn=w.fn
            )
            if materialize:
                wi.materialize()
            yield wi

    def _enqueue(
        self,
        queue: deque,
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
        state = JobState(job_id, n_tasks, is_load)
        task_lst = list(tasks)  # Materialize tasks out of self._condition
        assert len(task_lst) == n_tasks, "Unaccounted tasks"
        with self._condition:
            self._inflight_jobs += 1

            num_work = 0
            wi = WorkItem(state=state, tasks=task_lst, make_batch_fn=make_batch_fn)
            if is_load:
                self._inflight_load_jobs += 1
                self._inflight_load_tasks += n_tasks
                tasks_per_job = self._inflight_load_tasks / self._inflight_load_jobs
                if n_tasks < tasks_per_job:
                    self._load_fast_q.put(
                        (n_tasks, state.job_id, wi)
                    )  # not materialized
                    num_work += 1
                else:
                    for w in self._batch_workitem(wi, self._n_read_threads):
                        self._load_q.append(w)  # materialized
                        num_work += 1
            else:
                self._inflight_store_jobs += 1
                self._inflight_store_tasks += n_tasks
                tasks_per_job = self._inflight_store_tasks / self._inflight_store_jobs
                if n_tasks < tasks_per_job:
                    wi.materialize()
                    self._store_fast_q.put((n_tasks, state.job_id, wi))  # materialized
                    num_work += 1
                else:
                    wi.materialize()
                    self._store_q.append(wi)  # materialized
                    num_work += 1
            self._condition.notify_all()

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
            n_tasks,
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
            self._store_q,
            make_batch_fn,
            job_id,
            tasks,
            n_tasks,
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

    def _worker(self, role: Role) -> None:
        # Wait for tasks, process from primary queue first, fall back to secondary.

        def fetch_work():
            if role == Role.LOAD:
                if self._load_q:
                    return self._load_q.popleft().as_work()
                assert not self._load_fast_q.empty()
                print(f"Warning : {role} fetching from fast q")
                _, _, wi = self._load_fast_q.get()
                assert not wi.is_materialized()
                for w in self._batch_workitem(wi, self._n_read_threads):
                    self._load_q.append(w)
                return self._load_q.popleft().as_work()
            if role == Role.STORE:
                if self._store_q:
                    return self._store_q.popleft().as_work()
                print(f"Warning : {role} fetching from fast q")
                assert not self._store_fast_q.empty()
                _, _, wi = self._store_fast_q.get()
                return wi.as_work()
            if role == Role.LOAD_FAST:
                assert not self._load_fast_q.empty()
                _, _, wi = self._load_fast_q.get()
                wi.materialize()
                return wi.as_work()
            if role == Role.STORE_FAST:
                assert not self._store_fast_q.empty()
                _, _, wi = self._store_fast_q.get()
                return wi.as_work()

        def has_work():
            if role == Role.LOAD:
                return self._load_q or not self._load_fast_q.empty()
            if role == Role.STORE:
                return self._store_q or not self._store_fast_q.empty()
            if role == Role.LOAD_FAST:
                return not self._load_fast_q.empty()
            if role == Role.STORE_FAST:
                return not self._store_fast_q.empty()

        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stop or has_work())
                if self._stop:
                    return
                fn, batch_size, state = fetch_work()
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
                with self._condition:
                    self._finished_q.append((state.job_id, success, total_time))
                    if state.is_load:
                        self._inflight_load_jobs -= 1
                        self._inflight_load_tasks -= state._completed
                    else:
                        self._inflight_store_jobs -= 1
                        self._inflight_store_tasks -= state._completed
                    self._inflight_jobs -= 1
                    self._condition.notify_all()
