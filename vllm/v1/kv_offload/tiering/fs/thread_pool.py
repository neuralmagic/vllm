# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thread pool:
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
from vllm.v1.kv_offload.base import Locality, OffloadKey
from vllm.v1.kv_offload.tiering.base import JobId
from vllm.v1.kv_offload.tiering.fs.dispatch import (
    Job,
    JobSentinel,
    JobState,
    LoadJob,
    LoadQueue,
    StoreJob,
    StoreQueue,
    WorkDispatcher,
    make_batches,
)

logger = init_logger(__name__)


@dataclass
class Task:
    """I/O Task inputs"""

    key: OffloadKey
    path: str
    offset: int | None  # None for load jobs until alloc_fn resolves chunk_ids


class DualQueueThreadPool:
    """Thread pool with two task queues (load and store) and two thread groups.

    - Load-priority threads: drain the load queue first, steal stores when idle.
    - Store-priority threads: drain the store queue first, steal loads when idle.

    All groups share a single condition variable.
    """

    def __init__(
        self,
        n_read_threads: int,
        n_write_threads: int,
        block_size: int,
        locality: Locality,
        thread_name_prefix: str = "fs_secondary_tier",
    ) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._idle_condition = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []
        self._finished_q: deque[tuple[JobId, bool, float]] = deque()
        self._inflight_jobs = 0  # guarded by _condition

        assert n_read_threads + n_write_threads > 0, (
            "Threadpool needs atleast on 1 rw thread"
        )

        self._dispatcher = WorkDispatcher(
            locality=locality,
            load_job_q=LoadQueue(block_size),
            store_job_q=StoreQueue(block_size),
            n_read_threads=n_read_threads,
            n_write_threads=n_write_threads,
        )

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

    def _submit_job(self, job_id: JobId, job: Job, n_tasks: int, is_load: bool) -> None:
        """Register a job with the dispatcher and wake worker threads.

        If n_tasks is 0 the job completes immediately with success.
        """
        if n_tasks == 0:
            self._finished_q.append((job_id, True, 0.0))
            return
        with self._condition:
            self._inflight_jobs += 1
            n_wake = self._dispatcher.submit(job_id, job, n_tasks, is_load)
            self._condition.notify(n_wake)

    def enqueue_load(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Task],
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
        alloc_fn: Callable[[], tuple[list, list] | None],
    ) -> None:
        """Enqueue a load job (high-priority for load-priority threads).

        CPU slot allocation and task batching are deferred to the first worker
        thread that picks up the job (via alloc_fn). File paths are
        pre-computed by the caller on the calling thread.
        """
        task_lst = list(tasks)
        job = LoadJob(
            alloc_fn=alloc_fn,
            make_batch_fn=make_batch_fn,
            n_threads=self._dispatcher.n_batch_threads(is_load=True),
        )
        self._submit_job(job_id, job, len(task_lst), is_load=True)

    def enqueue_store(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Task],
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
    ) -> None:
        """Enqueue a store job (high-priority for store-priority threads).

        Tasks are pre-batched outside the lock before submission.
        """
        task_lst = list(tasks)
        assert len(task_lst) == n_tasks, "Unaccounted tasks"
        n_threads = self._dispatcher.n_batch_threads(is_load=False)
        state = JobState(job_id, n_tasks)
        # Build batches outside the lock — O(n_tasks) list-slicing and closure
        # construction must not block threads waiting on the condition variable.
        work_items = make_batches(state, task_lst, make_batch_fn, n_threads)
        job = StoreJob(work_items)
        self._submit_job(job_id, job, n_tasks, is_load=False)

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
        with self._idle_condition:
            self._idle_condition.wait_for(lambda: self._inflight_jobs == 0)

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._stop = True
            self._dispatcher.clear()
            # Cancelled tasks will not decrement _inflight_jobs; reset it so a
            # subsequent wait_idle() returns instead of hanging.
            self._inflight_jobs = 0
            self._condition.notify_all()
        with self._idle_condition:
            self._idle_condition.notify_all()
        if wait:
            for t in self._threads:
                t.join()

    def _worker(self, load_priority: bool) -> None:
        # Wait for tasks, drain primary queue first, steal from secondary when idle.
        while True:
            io_work: tuple | None = None
            sentinel_job_id: JobId | None = None
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop or self._dispatcher.has_work(load_priority)
                )
                if self._stop:
                    return
                work = self._dispatcher.fetch_work(load_priority)
                if work is None:
                    continue
                if isinstance(work, JobSentinel):
                    # Job resolved synchronously (OOM or all keys cached).
                    self._finished_q.append((work.job_id, work.success, 0.0))
                    self._inflight_jobs -= 1
                    sentinel_job_id = work.job_id
                else:
                    io_work = work

            if sentinel_job_id is not None:
                with self._idle_condition:
                    self._idle_condition.notify_all()
                continue

            assert io_work is not None
            fn, batch_size, state = io_work
            try:
                start_time = time.monotonic()
                fn()
                end_time = time.monotonic()
                job_finished, success, total_time = state.task_done(
                    batch_size, True, start_time, end_time
                )
            except Exception as exc:
                end_time = time.monotonic()
                logger.error(
                    "Job %s block I/O failed: %s",
                    state.job_id,
                    exc,
                )
                job_finished, success, total_time = state.task_done(
                    batch_size, False, start_time, end_time
                )

            if job_finished:
                with self._condition:
                    self._finished_q.append((state.job_id, success, total_time))
                    self._inflight_jobs -= 1
                with self._idle_condition:
                    self._idle_condition.notify_all()
