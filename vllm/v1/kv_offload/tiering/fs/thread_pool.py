# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Thread pool:
    Two queues (load, store) and two sets of threads:
      - Load-priority threads: drain the load queue first, then the store queue.
      - Store-priority threads: drain the store queue first, then the load queue.
    Load jobs are enqueued to the load queue; store jobs to the store queue.

AtimeTouchWorker:
    A single background worker that accumulates file paths whose atime should
    be refreshed (via os.utime) and flushes them in batches.  Two triggers fire
    the flush: the pending-set size reaching `max_pending`, or `flush_interval_s`
    seconds elapsing — whichever comes first.
"""

import contextlib
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.base import JobId

logger = init_logger(__name__)


class JobState:
    """
    Thread-safe completion tracker for a set of per-block I/O tasks.

    Each task calls task_done(success) when it finishes.
    """

    __slots__ = ("_job_id", "_n_tasks", "_completed", "_success", "_lock")

    def __init__(self, job_id: JobId, n_tasks: int) -> None:
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._lock = threading.Lock()

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def task_done(self, success: bool) -> tuple[bool, bool]:
        """Returns if job completed and success flag"""
        with self._lock:
            self._completed += 1
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
        self._load_q: deque = deque()
        self._store_q: deque = deque()
        self._condition = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []
        self._finished_q: deque[tuple[JobId, bool]] = deque()

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
        state = JobState(job_id, n_tasks)
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
        state = JobState(job_id, n_tasks)
        with self._condition:
            for fn in tasks:
                self._store_q.append((fn, state))
            self._condition.notify(n_tasks)

    def get_finished(self) -> list[tuple[JobId, bool]]:
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
                task()
                job_finished, success = state.task_done(True)
            except Exception as exc:
                logger.error(
                    "Job %s block I/O failed: %s",
                    state.job_id,
                    exc,
                )
                job_finished, success = state.task_done(False)

            if job_finished:
                self._finished_q.append((state.job_id, success))


class AtimeTouchWorker:
    """
    Background worker that refreshes file atimes in batches.

    Paths are accumulated in a dedup dict mapping path → enqueue timestamp.
    The worker flushes — calling os.utime(path, (ts, ts)) for every collected
    entry — when either the size cap is reached or the timeout elapses,
    whichever comes first.  Duplicates within a window collapse to a single
    syscall; the later enqueue timestamp wins.

    Recording the timestamp at enqueue time (when the load is confirmed) rather
    than at flush time means the atime written to disk reflects when the block
    was actually read, not when the batch happened to run.
    """

    SUPPORTED_CONFIG_KEYS: frozenset[str] = frozenset(
        {"max_pending", "flush_interval_s"}
    )

    def __init__(self, config: dict) -> None:
        unknown = set(config) - self.SUPPORTED_CONFIG_KEYS
        assert not unknown, (
            f"Unrecognized atime_config keys: {sorted(unknown)}. "
            f"Supported keys: {sorted(self.SUPPORTED_CONFIG_KEYS)}"
        )
        self._max_pending: int = config.get("max_pending", 10_000)
        self._flush_interval_s: float = config.get("flush_interval_s", 1800.0)
        # path → POSIX timestamp recorded at enqueue (load-confirmed) time.
        self._pending: dict[str, float] = {}
        self._cond = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="vllm_kv_atime",
        )
        self._thread.start()

    def enqueue(self, path: str) -> None:
        """
        Record a file path for atime refresh.

        Called from the scheduler thread after get_finished_jobs() confirms a
        successful load.  The current time is stored as the intended atime so
        that the written timestamp reflects the actual read time, not the later
        batch-flush time.  Wakes the worker immediately when the size cap is
        reached.
        """
        ts = time.time()
        with self._cond:
            self._pending[path] = ts
            if len(self._pending) >= self._max_pending:
                self._cond.notify()

    def shutdown(self) -> None:
        """Flush remaining paths and stop the worker thread."""
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join()

    def _run(self) -> None:
        while True:
            with self._cond:
                timed_out = not self._cond.wait_for(
                    lambda: self._stop or len(self._pending) >= self._max_pending,
                    timeout=self._flush_interval_s,
                )
                # Flush on timeout, size cap, or stop.
                if timed_out or self._stop or self._pending:
                    pending, self._pending = self._pending, {}
                else:
                    pending = {}
                stop = self._stop

            if pending:
                trigger = "stop" if stop else "timeout" if timed_out else "size_cap"
                logger.debug(
                    "AtimeTouchWorker flushing %d path(s) [trigger=%s]",
                    len(pending),
                    trigger,
                )
            for p, ts in pending.items():
                with contextlib.suppress(OSError):
                    os.utime(p, (ts, ts))

            if stop:
                return
