# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Thread pool:
    Two queues (load, store) and two sets of threads:
      - Load-priority threads: drain the load queue first, then the store queue.
      - Store-priority threads: drain the store queue first, then the load queue.
    Load jobs are enqueued to the load queue; store jobs to the store queue.
"""

import itertools
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.base import JobId

logger = init_logger(__name__)


@dataclass
class Task:
    """
    I/O Task inputs
    """

    path: str
    offset: int


@dataclass(frozen=True)
class FinishedJob:
    """Result and timing information for one completed job."""

    job_id: JobId
    success: bool
    is_load: bool
    n_tasks: int
    # Time from enqueue to the job's last task completing.
    job_duration: float
    # Time from enqueue to a worker thread picking up the job's first batch.
    queueing_delay: float
    # Time from a worker picking up the first batch to the last task
    # completing, i.e. job_duration - queueing_delay.
    execution_time: float


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
        "_lock",
        "_enqueue_time",
        "_first_batch_time",
        "_is_load",
    )

    def __init__(self, job_id: JobId, n_tasks: int, is_load: bool) -> None:
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._lock = threading.Lock()
        self._enqueue_time = time.monotonic()
        self._first_batch_time: float | None = None
        self._is_load = is_load

    @property
    def job_id(self) -> JobId:
        return self._job_id

    @property
    def is_load(self) -> bool:
        return self._is_load

    @property
    def n_tasks(self) -> int:
        return self._n_tasks

    @property
    def enqueue_time(self) -> float:
        return self._enqueue_time

    @property
    def first_batch_time(self) -> float | None:
        return self._first_batch_time

    def mark_batch_start(self) -> bool:
        """Record, once, when the first batch of this job started executing.

        Multiple threads may pop different batches of the same job at
        nearly the same time; only the first call sets the timestamp.

        Returns:
            True if this call was the one that set the timestamp (i.e. the
            job just transitioned from queued to executing), False otherwise.
        """
        with self._lock:
            if self._first_batch_time is None:
                self._first_batch_time = time.monotonic()
                return True
            return False

    def task_done(self, batch_size: int, success: bool) -> tuple[bool, bool]:
        """Returns if job completed and success flag"""
        with self._lock:
            self._completed += batch_size
            if not success:
                self._success = False
            return self._completed == self._n_tasks, self._success


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
        self._finished_q: deque[FinishedJob] = deque()
        # All of the following are guarded by _condition.
        self._inflight_read_jobs = 0
        self._inflight_write_jobs = 0
        self._active_read_threads = 0
        self._active_write_threads = 0
        self._active_read_jobs = 0
        self._active_write_jobs = 0

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

    def _batch_tasks(
        self,
        tasks: Iterable[Task],
        n_tasks: int,
        n_threads: int,
    ) -> Iterator[list[Task]]:
        """
        Batch tasks so that the request's tasks are split evenly across the
        n_threads.
        """
        tasks = iter(tasks)
        q, r = divmod(n_tasks, n_threads)
        batch_sizes = [q + 1 if i < r else q for i in range(n_threads)]
        for bs in batch_sizes[: min(n_tasks, n_threads)]:
            batch = list(itertools.islice(tasks, bs))
            assert len(batch) == bs
            yield batch

        assert next(tasks, None) is None, "Unaccounted tasks"

    def _enqueue(
        self,
        queue: deque,
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
        job_id: JobId,
        tasks: Iterable[Task],
        n_tasks: int,
        n_threads: int,
    ) -> None:
        """Batch `tasks` and append (fn, state, batch_size) entries to `queue`."""
        is_load = queue is self._load_q
        if n_tasks == 0:
            self._finished_q.append(
                FinishedJob(
                    job_id=job_id,
                    success=True,
                    is_load=is_load,
                    n_tasks=0,
                    job_duration=0.0,
                    queueing_delay=0.0,
                    execution_time=0.0,
                )
            )
            return
        state = JobState(job_id, n_tasks, is_load=is_load)
        n_batches = 0
        with self._condition:
            if is_load:
                self._inflight_read_jobs += 1
            else:
                self._inflight_write_jobs += 1
            for batch in self._batch_tasks(tasks, n_tasks, n_threads):
                queue.append((make_batch_fn(batch), len(batch), state))
                n_batches += 1
            self._condition.notify(n_batches)

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
            n_threads=self._n_read_threads,
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
            n_threads=self._n_write_threads,
        )

    def get_finished(self) -> list[FinishedJob]:
        """Returns the list of jobs that have fully completed since the
        last call."""
        # No lock needed: deque is thread-safe for concurrent append/popleft,
        # and the manager is the sole popper.
        jobs = []
        while self._finished_q:
            jobs.append(self._finished_q.popleft())
        return jobs

    @property
    def num_inflight_read_jobs(self) -> int:
        """Number of load jobs submitted but not yet fully completed.

        Includes jobs still waiting in the load/store queues as well as
        jobs currently being executed by a worker thread.
        """
        with self._condition:
            return self._inflight_read_jobs

    @property
    def num_inflight_write_jobs(self) -> int:
        """Number of store jobs submitted but not yet fully completed.

        Includes jobs still waiting in the load/store queues as well as
        jobs currently being executed by a worker thread.
        """
        with self._condition:
            return self._inflight_write_jobs

    @property
    def num_active_read_threads(self) -> int:
        """Number of worker threads currently executing a load batch."""
        with self._condition:
            return self._active_read_threads

    @property
    def num_active_write_threads(self) -> int:
        """Number of worker threads currently executing a store batch."""
        with self._condition:
            return self._active_write_threads

    @property
    def num_active_read_jobs(self) -> int:
        """Number of distinct load jobs currently executing.

        Unlike num_active_read_threads, a job with multiple batches running
        concurrently on multiple threads is only counted once here.
        """
        with self._condition:
            return self._active_read_jobs

    @property
    def num_active_write_jobs(self) -> int:
        """Number of distinct store jobs currently executing.

        Unlike num_active_write_threads, a job with multiple batches running
        concurrently on multiple threads is only counted once here.
        """
        with self._condition:
            return self._active_write_jobs

    def wait_idle(self) -> None:
        """Block until there are no in-flight jobs.

        After this returns, every submitted job has had its last task
        finish, so no worker thread is still copying data. Note:
        completed jobs may still be sitting in ``_finished_q`` waiting
        for ``get_finished()`` to drain them.
        """
        with self._condition:
            self._condition.wait_for(
                lambda: self._inflight_read_jobs == 0 and self._inflight_write_jobs == 0
            )

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._stop = True
            self._load_q.clear()
            self._store_q.clear()
            # Cancelled tasks will not decrement _inflight_{read,write}_jobs;
            # reset them so a subsequent wait_idle() returns instead of
            # hanging.
            self._inflight_read_jobs = 0
            self._inflight_write_jobs = 0
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
                fn, batch_size, state = (
                    primary.popleft() if primary else secondary.popleft()
                )
                if state.is_load:
                    self._active_read_threads += 1
                else:
                    self._active_write_threads += 1
            is_first_batch = state.mark_batch_start()
            if is_first_batch:
                with self._condition:
                    if state.is_load:
                        self._active_read_jobs += 1
                    else:
                        self._active_write_jobs += 1
            try:
                fn()
                job_finished, success = state.task_done(batch_size, True)
            except Exception as exc:
                logger.error(
                    "Job %s block I/O failed: %s",
                    state.job_id,
                    exc,
                )
                job_finished, success = state.task_done(batch_size, False)
            finally:
                with self._condition:
                    if state.is_load:
                        self._active_read_threads -= 1
                    else:
                        self._active_write_threads -= 1

            if job_finished:
                now = time.monotonic()
                job_duration = now - state.enqueue_time
                queueing_delay = (state.first_batch_time or now) - state.enqueue_time
                execution_time = job_duration - queueing_delay
                with self._condition:
                    self._finished_q.append(
                        FinishedJob(
                            job_id=state.job_id,
                            success=success,
                            is_load=state.is_load,
                            n_tasks=state.n_tasks,
                            job_duration=job_duration,
                            queueing_delay=queueing_delay,
                            execution_time=execution_time,
                        )
                    )
                    if state.is_load:
                        self._inflight_read_jobs -= 1
                        self._active_read_jobs -= 1
                    else:
                        self._inflight_write_jobs -= 1
                        self._active_write_jobs -= 1
                    self._condition.notify_all()
