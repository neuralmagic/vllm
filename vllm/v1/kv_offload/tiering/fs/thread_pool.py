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


class Phase(Enum):
    LOAD = 1
    STORE = 2
    NONE = 3


class Role(Enum):
    PHASED = 1
    FAST = 2


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
    def is_load(self):
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


@dataclass
class WorkItem:
    state: JobState
    tasks: list[Task]
    make_batch_fn: Callable | None = None
    fn: Callable | None = None

    @property
    def n_tasks(self):
        return len(self.tasks)

    @property
    def is_load(self):
        return self.state.is_load

    def is_materialized(self):
        return self.fn is not None

    def materialize(self):
        if self.is_materialized():
            return self
        assert self.make_batch_fn is not None
        self.fn = self.make_batch_fn(self.tasks)
        return self

    def as_work(self):
        assert self.is_materialized()
        return self.fn, len(self.tasks), self.state


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
        n_fast_threads: int,
        thread_name_prefix: str = "fs_secondary_tier",
    ) -> None:
        self._n_read_threads = n_read_threads
        self._n_write_threads = n_write_threads
        self._n_fast_threads = n_fast_threads
        self._load_q: PriorityQueue = PriorityQueue()
        self._store_q: PriorityQueue = PriorityQueue()
        self._wq: deque = deque()
        self._condition = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []
        self._finished_q: deque[tuple[JobId, bool, float]] = deque()
        self._inflight_jobs = 0  # guarded by _condition

        self._phase = Phase.NONE
        self._phase_jobs: set[JobId] = set()

        assert self.total_threads > 0, "ThreadPool needs at least one thread"

        for i in range(self._n_read_threads):
            t = threading.Thread(
                target=self._worker,
                args=(Role.PHASED,),
                name=f"{thread_name_prefix}_l{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(self._n_write_threads):
            t = threading.Thread(
                target=self._worker,
                args=(Role.PHASED,),
                name=f"{thread_name_prefix}_s{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(self._n_fast_threads):
            t = threading.Thread(
                target=self._worker,
                args=(Role.FAST,),
                name=f"{thread_name_prefix}_fast{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def switch_phase(self):
        assert not self._wq
        if self._phase == Phase.NONE:
            if not self._load_q.empty():
                self._phase = Phase.LOAD
            elif not self._store_q.empty():
                self._phase = Phase.STORE
            return
        elif self._phase == Phase.LOAD and not self._store_q.empty():
            self._phase = Phase.STORE
        elif self._phase == Phase.STORE and not self._load_q.empty():
            self._phase = Phase.LOAD
        if self._phase == Phase.LOAD:
            assert not self._load_q.empty()
        if self._phase == Phase.STORE:
            assert not self._store_q.empty()
        return self._phase

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

    def _batch_work_item(self, wi: WorkItem, n_threads: int):
        assert not wi.is_materialized()
        for b in self._batch_tasks(wi.tasks, n_threads):
            yield WorkItem(
                state=wi.state, tasks=b, make_batch_fn=wi.make_batch_fn, fn=wi.fn
            )

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
        state = JobState(job_id, n_tasks, is_load)
        task_lst = list(tasks)  # Materialize tasks out of self._condition
        assert len(task_lst) == n_tasks, "Unaccounted tasks"

        wi = WorkItem(state, task_lst, make_batch_fn)
        with self._condition:
            self._inflight_jobs += 1
            if is_load:
                self._load_q.put((wi.n_tasks, wi.state.job_id, wi))
            else:
                self._store_q.put((wi.n_tasks, wi.state.job_id, wi))
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
        with self._condition:
            self._condition.wait_for(lambda: self._inflight_jobs == 0)

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._stop = True
            while not self._load_q.empty():
                self._load_q.get()
            while not self._store_q.empty():
                self._store_q.get()
            # Cancelled tasks will not decrement _inflight_jobs; reset it so a
            # subsequent wait_idle() returns instead of hanging.
            self._inflight_jobs = 0
            self._condition.notify_all()
        if wait:
            for t in self._threads:
                t.join()

    def _worker(self, role: Role) -> None:
        # Wait for tasks, process from primary queue first, fall back to secondary.

        def has_work():
            if role == Role.FAST:
                return not self._load_q.empty() or not self._store_q.empty()
            if self._wq:
                return True
            # wait till phase work is complete
            return len(self._phase_jobs) == 0 and (
                not self._load_q.empty() or not self._store_q.empty()
            )

        def populate_wq(q: PriorityQueue):
            assert not self._phase_jobs
            while not q.empty():
                _, _, wi = q.get()
                for b in self._batch_work_item(wi, self.total_threads):
                    self._wq.append(b.materialize().as_work())
                self._phase_jobs.add(wi.state.job_id)

        def fetch_work():
            if role == Role.FAST:
                # Fast lane: always prefer the globally smallest job
                if self._load_q.empty():
                    _, _, wi = self._store_q.get()
                    return wi.materialize().as_work()
                elif self._store_q.empty():
                    _, _, wi = self._load_q.get()
                    return wi.materialize().as_work()
                # both have work
                load_min = self._load_q.queue[0][0]
                store_min = self._store_q.queue[0][0]
                if load_min < store_min:
                    _, _, wi = self._load_q.get()
                    return wi.materialize().as_work()
                else:
                    _, _, wi = self._store_q.get()
                    return wi.materialize().as_work()

            if self._wq:
                return self._wq.popleft()
            # run out of work
            self.switch_phase()
            assert self._phase != Phase.NONE
            if self._phase == Phase.LOAD:
                assert not self._load_q.empty()
                populate_wq(self._load_q)
            else:
                assert self._phase == Phase.STORE
                assert not self._store_q.empty()
                populate_wq(self._store_q)
            return self._wq.popleft()

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
                    self._phase_jobs.discard(state.job_id)
                    self._inflight_jobs -= 1
                    self._condition.notify_all()
