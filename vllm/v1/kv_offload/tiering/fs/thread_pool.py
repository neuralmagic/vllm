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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.tiering.base import JobId
from vllm.v1.kv_offload.tiering.fs.thread_pool_deque import TPDeque

try:
    from vllm.fs_io_C import get_status_snapshot as get_status_snapshot_C

    _HAS_FSIO_C = True
except ImportError:
    _HAS_FSIO_C = False

logger = init_logger(__name__)


@dataclass
class Task:
    job_id: JobId
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
        "_failed_keys",
        "_lock",
    )

    def __init__(self, job_id: JobId, n_tasks: int) -> None:
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._transfer_time = 0.0
        self._failed_keys: list[OffloadKey] = []
        self._lock = threading.Lock()

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def task_done(
        self, task: Task, success: bool, transfer_time: float
    ) -> tuple[bool, bool, float]:
        """Returns if job completed and success flag"""
        with self._lock:
            self._completed += 1
            self._transfer_time += transfer_time
            if not success:
                self._failed_keys.append(task.key)
                self._success = False
            return self._completed == self._n_tasks, self._success, self._transfer_time


class BatchResultsTracker:
    BatchId: TypeAlias = int

    @dataclass
    class Tracker:
        tasks: list[Task]
        _results: np.ndarray
        processed: int = 0

        def update(self) -> np.ndarray | list[int] | None:
            if _HAS_FSIO_C:
                try:
                    return get_status_snapshot_C(self._results, self.processed)
                except Exception as exc:
                    logger.error(
                        "Failed to receive I/O results asynchronously %s",
                        exc,
                    )
                    return None
            else:
                return self._results[self.processed :]

    def __init__(self):
        self.batch_id = 0
        self._batch_results: dict[
            BatchResultsTracker.BatchId, BatchResultsTracker.Tracker
        ] = {}
        self._lock = threading.Lock()

    def new_tracker(self, tasks: list[Task]):
        results = np.full(len(tasks), -1, dtype=np.int8)
        with self._lock:
            self.batch_id += 1
            batch_id = self.batch_id
            self._batch_results[batch_id] = BatchResultsTracker.Tracker(
                tasks, _results=results
            )
        return batch_id, results

    def drain(self) -> Iterable[tuple[Task, bool]]:
        # Update all trackers
        with self._lock:
            batch_ids, trackers = (
                list(self._batch_results.keys()),
                list(self._batch_results.values()),
            )

        for bid, tracker in zip(batch_ids, trackers):
            # handle the error case
            results = tracker.update()
            assert results is not None  # TODO(varun) : Handle none case
            for r in results:
                if r == -1:
                    break
                task = tracker.tasks[tracker.processed]
                success = r == 0
                tracker.processed += 1
                yield task, success

            if tracker.processed == len(tracker.tasks):
                with self._lock:
                    self._batch_results.pop(bid)


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
        self._load_q: TPDeque = TPDeque(n_read_threads)
        self._store_q: TPDeque = TPDeque(n_write_threads)
        self._condition = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []
        self._inflight_jobs = 0  # guarded by _condition
        self._job_state: dict[JobId, JobState] = {}
        # Finished-but-not-yet-reported jobs, guarded by _condition. Populated
        # by _pump(), drained by get_finished().
        self._finished_q: list[tuple[JobId, bool, float, list[OffloadKey]]] = []

        self._tracker: BatchResultsTracker = BatchResultsTracker()

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

    def _enqueue(
        self,
        q: TPDeque,
        job_id: JobId,
        tasks: list[Task],
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
    ) -> None:
        if len(tasks) == 0:
            with self._condition:
                self._finished_q.append((job_id, True, 0.0, []))
                return

        state = JobState(job_id, len(tasks))
        with self._condition:
            self._inflight_jobs += 1
            self._job_state[job_id] = state
            q.submit(tasks, make_batch_fn)
            self._condition.notify(len(tasks))

    def enqueue_load(
        self,
        job_id: JobId,
        tasks: list[Task],
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
    ) -> None:
        # submit to the queue deque enmasse
        self._enqueue(self._load_q, job_id, tasks, make_batch_fn)

    def enqueue_store(
        self,
        job_id: JobId,
        tasks: list[Task],
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
    ) -> None:
        self._enqueue(self._store_q, job_id, tasks, make_batch_fn)

    def _pump(self) -> None:
        """Discover newly finished jobs and buffer their results in
        ``_finished_q`` for ``get_finished()`` to hand back later.

        Nothing else drives completion discovery on its own -- I/O results
        only become visible by actively polling ``self._tracker``. Safe to
        call from any thread, repeatedly; a call that finds nothing new is
        a cheap no-op.
        """
        newly_finished: list[tuple[JobId, bool, float, list[OffloadKey]]] = []
        for task, success in self._tracker.drain():
            state = self._job_state[task.job_id]
            finished, success, transfer_time = state.task_done(task, success, 0.0)
            if finished:
                self._job_state.pop(task.job_id)
                newly_finished.append(
                    (task.job_id, success, transfer_time, state._failed_keys)
                )
        if newly_finished:
            with self._condition:
                self._finished_q.extend(newly_finished)
                self._inflight_jobs -= len(newly_finished)
                self._condition.notify_all()

    def get_finished(self) -> Iterable[tuple[JobId, bool, float, list[OffloadKey]]]:
        self._pump()
        with self._condition:
            finished, self._finished_q = self._finished_q, []
        yield from finished

    def wait_idle(self, poll_interval: float = 0.001) -> None:
        """Block until there are no in-flight jobs.

        Actively pumps I/O-completion status itself -- nothing else does
        so on its own -- so this is always safe to call and will not hang.
        After this returns, every submitted job has had its last task
        finish, so no worker thread is still copying data. Note:
        completed jobs may still be sitting in ``_finished_q`` waiting
        for ``get_finished()`` to drain them.
        """
        while True:
            self._pump()
            with self._condition:
                if self._inflight_jobs == 0:
                    return
                self._condition.wait(timeout=poll_interval)

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._stop = True
            self._load_q.clear()
            self._store_q.clear()
            # Cancelled tasks will not decrement _inflight_jobs; reset it so a
            # subsequent wait_idle() returns instead of hanging.
            self._inflight_jobs = 0
            self._finished_q.clear()
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
                fn, tasks = primary.fetch() if primary else secondary.fetch()
            try:
                _, results = self._tracker.new_tracker(tasks)
                fn(results=results)
            except Exception as exc:
                logger.error(
                    "Block I/O failed: %s",
                    exc,
                )
            finally:
                # Wake any wait_idle() poller so it re-pumps promptly instead
                # of waiting out its poll_interval.
                with self._condition:
                    self._condition.notify_all()
