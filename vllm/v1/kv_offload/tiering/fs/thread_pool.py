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


class _IdleEwma:
    """Time-weighted exponentially-weighted moving average of a live
    idle-thread count.

    A single popping thread only ever observes the *instantaneous* idle
    count -- in a steady stream of small jobs, threads finish staggered
    one at a time, so that instantaneous count is almost always "just me"
    even when the pool has real spare capacity on average. Sizing batch
    splits off this smoothed signal instead avoids that: a thread that
    just became idle sees "typically N have been idle recently" rather
    than "only I am idle right now", so it takes a smaller slice and
    leaves the rest for the next (possibly also-staggered) idle thread to
    pick up, instead of swallowing an entire job by itself.

    Time-weighted (not sample-weighted): decay is applied based on
    wall-clock time elapsed since the last update, so a count that's been
    held steady for a while contributes proportionally more than one that
    just changed a moment ago -- same idea as Linux's per-entity load
    tracking (PELT).
    """

    __slots__ = ("_tau", "_value", "_raw", "_last_t")

    def __init__(self, initial: int, tau: float) -> None:
        self._tau = tau
        self._value = float(initial)
        self._raw = initial
        self._last_t = time.monotonic()

    def _decay_to_now(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_t
        if elapsed > 0:
            decay = math.exp(-elapsed / self._tau)
            self._value = self._value * decay + self._raw * (1 - decay)
            self._last_t = now

    def update(self, new_raw: int) -> None:
        """Must be called while holding the pool's lock, right after the
        underlying idle-thread counter changes to `new_raw`. Decays the
        average up to now using the *previous* raw level (i.e. how long
        that previous level was actually held), then starts tracking the
        new level going forward."""
        self._decay_to_now()
        self._raw = new_raw

    def value(self) -> float:
        """Must be called while holding the pool's lock."""
        self._decay_to_now()
        return self._value


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
        idle_ewma_tau_s: float = 0.2,
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
        # Smoothed idle-thread signal used to size batch splits -- see
        # _IdleEwma. Guarded by _condition, like the raw counts above.
        self._idle_read_ewma = _IdleEwma(n_read_threads, tau=idle_ewma_tau_s)
        self._idle_write_ewma = _IdleEwma(n_write_threads, tau=idle_ewma_tau_s)

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

    def _idle_ewma_total(self) -> float:
        """Combined (read+write) smoothed idle-thread count. Must be called
        while holding _condition."""
        return self._idle_read_ewma.value() + self._idle_write_ewma.value()

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
                idle_ewma = (
                    self._idle_read_ewma.value() if primary else self._idle_ewma_total()
                )
            else:
                primary, secondary = self._store_q, self._load_q
                queue = primary if primary else secondary
                idle_ewma = (
                    self._idle_write_ewma.value()
                    if primary
                    else self._idle_ewma_total()
                )
            # Smoothed, not instantaneous (see _IdleEwma), and never less
            # than 1 -- the calling thread itself is always "available".
            idle_threads = max(1, round(idle_ewma))

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
                    self._idle_read_ewma.update(self._idle_read_threads)
                else:
                    self._idle_write_threads -= 1
                    self._idle_write_ewma.update(self._idle_write_threads)
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
                    self._idle_read_ewma.update(self._idle_read_threads)
                else:
                    self._idle_write_threads += 1
                    self._idle_write_ewma.update(self._idle_write_threads)
