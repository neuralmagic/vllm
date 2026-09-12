# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import math
import queue
import threading
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterator
from enum import Enum
from typing import Any

from vllm.v1.kv_offload.tiering.base import JobId


class ThreadWaker:
    """
    Role-scoped wakeup primitive backed by a POSIX semaphore (SimpleQueue).

    wait() parks the calling thread at the C level (GIL fully released) until
    notify() or stop() is called by another thread.  Each notify() wakes
    exactly one waiting thread — no thundering herd.  Tokens accumulate in the
    semaphore counter when all threads are busy, so no wakeup is ever lost.

    The sentinel objects (_WORK / _STOP) are private implementation details;
    callers only see the three named operations.
    """

    _WORK = object()
    _STOP = object()

    def __init__(self) -> None:
        self._q: queue.SimpleQueue = queue.SimpleQueue()

    def notify(self, n: int = 1) -> None:
        """Wake up to n threads that are blocked in wait()."""
        for _ in range(n):
            self._q.put(ThreadWaker._WORK)

    def wait(self) -> bool:
        """Block until notified. Returns True → do work, False → stop."""
        return self._q.get() is ThreadWaker._WORK

    def stop(self) -> None:
        """Unblock exactly one wait() call and signal it to exit."""
        self._q.put(ThreadWaker._STOP)


class TPScheduler(ABC):
    def __init__(self, n_read_threads: int, n_write_threads: int, block_size: int):
        self._n_read_threads = n_read_threads
        self._n_write_threads = n_write_threads
        self._block_size = block_size

    @abstractmethod
    def make_wakers(self) -> dict[Any, ThreadWaker]:
        """
        Return a mapping of {thread-args → ThreadWaker} for each thread role.

        Each worker thread receives its role-specific ThreadWaker and blocks on
        waker.wait() exclusively, so notify() calls from submit() can only wake
        threads eligible to process the incoming job.

        The scheduler stores the returned wakers internally so that submit()
        can call notify() directly — the thread pool does not route tokens.
        """
        pass

    @abstractmethod
    def make_threads(
        self,
        worker_fn: Callable,
        thread_name_prefix: str,
        wakers: dict[Any, ThreadWaker],
    ) -> list[tuple[threading.Thread, ThreadWaker]]:
        """
        Spawn and start worker threads, returning a list of (thread, waker) pairs.

        Each thread is started with worker_fn(args, role_waker) so it knows both
        its scheduler role (for has_work / fetch_work) and its dedicated waker
        (for blocking).  The companion waker is returned alongside the thread so
        the pool can call waker.stop() for each thread during shutdown.
        """
        pass

    @abstractmethod
    def submit(
        self, state: Any, make_batch_fn: Callable, tasks: list[Any], is_load: bool
    ) -> None:
        """
        Accept a job and call notify() on the appropriate role wakers.
        All wakeup routing logic lives here; the thread pool sees no return value.
        """
        pass

    @abstractmethod
    def has_work(self, args: Any) -> bool:
        pass

    @abstractmethod
    def fetch_work(self, args: Any):
        pass

    @abstractmethod
    def clear(self):
        pass


class NoBatchTPScheduler(TPScheduler):
    def __init__(self, n_read_threads: int, n_write_threads: int, block_size: int):
        super().__init__(n_read_threads, n_write_threads, block_size)
        self._load_q: deque = deque()
        self._store_q: deque = deque()
        # Populated by make_wakers(); used in submit() to notify threads.
        self._waker: ThreadWaker | None = None

    def make_wakers(self) -> dict[Any, ThreadWaker]:
        # Both roles share one waker: priority is enforced by fetch_work(), not
        # by which thread is woken.  has_work(True) == has_work(False) so any
        # thread can handle any job; the shared waker is therefore correct.
        waker = ThreadWaker()
        self._waker = waker
        return {True: waker, False: waker}

    def make_threads(
        self,
        worker_fn: Callable,
        thread_name_prefix: str,
        wakers: dict[Any, ThreadWaker],
    ) -> list[tuple[threading.Thread, ThreadWaker]]:
        result: list[tuple[threading.Thread, ThreadWaker]] = []
        for i in range(self._n_read_threads):
            waker = wakers[True]
            t = threading.Thread(
                target=worker_fn,
                args=(True, waker),
                name=f"{thread_name_prefix}_l{i}",
                daemon=True,
            )
            t.start()
            result.append((t, waker))
        for i in range(self._n_write_threads):
            waker = wakers[False]
            t = threading.Thread(
                target=worker_fn,
                args=(False, waker),
                name=f"{thread_name_prefix}_s{i}",
                daemon=True,
            )
            t.start()
            result.append((t, waker))
        return result

    def submit(
        self, state: Any, make_batch_fn: Callable, tasks: list[Any], is_load: bool
    ) -> None:
        q = self._load_q if is_load else self._store_q
        q.append((make_batch_fn(tasks), len(tasks), state))
        assert self._waker is not None
        self._waker.notify()

    def has_work(self, load_priority: bool) -> bool:
        return bool(self._load_q) or bool(self._store_q)

    def fetch_work(self, load_priority: bool):
        primary = self._load_q if load_priority else self._store_q
        secondary = self._store_q if load_priority else self._load_q
        if primary:
            return primary.popleft()
        if secondary:
            return secondary.popleft()
        return None

    def clear(self):
        self._load_q.clear()
        self._store_q.clear()


class BatchTPScheduler(NoBatchTPScheduler):
    def __init__(self, n_read_threads: int, n_write_threads: int, block_size: int):
        super().__init__(n_read_threads, n_write_threads, block_size)

    @property
    def _total_threads(self):
        return self._n_read_threads + self._n_write_threads

    def _batch_tasks(
        self,
        tasks: list[Any],
        n_threads: int,
    ) -> Iterator[list[Any]]:
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

    def submit(
        self, state: Any, make_batch_fn: Callable, tasks: list[Any], is_load: bool
    ) -> None:
        q = self._load_q if is_load else self._store_q
        if is_load:
            n_threads = (
                self._n_read_threads if self._n_read_threads else self._total_threads
            )
        else:
            n_threads = (
                self._n_write_threads if self._n_write_threads else self._total_threads
            )
        assert self._waker is not None
        for b in self._batch_tasks(tasks, n_threads):
            q.append((make_batch_fn(b), len(b), state))
            self._waker.notify()


class SSDTPScheduler(TPScheduler):
    class Role(Enum):
        READ = 1
        WRITE = 2
        FAST = 3

    class FCFSQueue:
        def __init__(self):
            self.fcfs_q: deque[tuple[JobId, int]] = deque()

        def add(self, job_id: JobId, num_tasks: int):
            self.fcfs_q.append((job_id, num_tasks))

        def next(self) -> tuple[JobId, int]:
            return self.fcfs_q.popleft()

        def clear(self):
            self.fcfs_q.clear()

        def has_work(self):
            return bool(self.fcfs_q)

    class SJFQueue:
        MAX_BUCKETS = 32

        def __init__(self, block_size: int):
            self._block_size = block_size
            self.sjf_q: list[deque[tuple[JobId, int]]] = [
                deque() for _ in range(SSDTPScheduler.SJFQueue.MAX_BUCKETS)
            ]
            self.sjf_meta = ctypes.c_uint32(0x00000000)

        def _get_bucket_id(self, job_id: JobId, num_tasks: int):
            assert num_tasks != 0
            mbytes = max(1, num_tasks * self._block_size)
            return int(math.floor(math.log2(mbytes)))

        def _add(self, job_id: JobId, num_tasks: int):
            bucket_id = self._get_bucket_id(job_id, num_tasks)
            assert bucket_id < SSDTPScheduler.SJFQueue.MAX_BUCKETS
            self.sjf_q[bucket_id].append((job_id, num_tasks))
            self.sjf_meta.value |= 1 << bucket_id

        def _remove(self, job_id: JobId, num_tasks: int):
            bucket_id = self._get_bucket_id(job_id, num_tasks)
            assert bucket_id < SSDTPScheduler.SJFQueue.MAX_BUCKETS

            q = self.sjf_q[bucket_id]
            assert q
            q_job_id, _ = q[0]
            assert job_id == q_job_id, f"Inconsistent q state {q}"
            q.popleft()
            if not q:
                self.sjf_meta.value &= ~(1 << bucket_id)

        def _get_sjf_bucket(self):
            x = self.sjf_meta.value
            assert x != 0
            # ~v inverts the bits.
            # + 1 completes the two's complement.
            # & 0xFFFFFFFF forces it to stay within 32-bit unsigned bounds.
            pow2 = x & ((~x + 1) & 0xFFFFFFFF)
            return pow2.bit_length() - 1

        def has_short_job(self) -> bool:
            # return true if we detect multimodal distribution
            x = self.sjf_meta.value
            if x == 0:
                return False

            b = self._get_sjf_bucket()
            # Shift the mask so the minimum bucket aligns to the 0th bit
            m = x >> b

            # 3. Apply the bitwise rules
            # Condition A:
            #  Is this a wide distribution
            # Condition B:
            #  Is there a valley in distribution
            return ((m & 7) == 7) or (((m + 1) & m) != 0)

        def add(self, job_id: JobId, num_tasks: int):
            self._add(job_id, num_tasks)

        def next(self) -> tuple[JobId, int]:
            bucket = self._get_sjf_bucket()
            assert self.sjf_q[bucket]
            job_id, num_tasks = self.sjf_q[bucket][0]
            self._remove(job_id, num_tasks)
            return job_id, num_tasks

        def clear(self):
            for q in self.sjf_q:
                q.clear()
            self.sjf_meta = ctypes.c_uint32(0x00000000)

        def has_work(self):
            return self.sjf_meta.value != 0

    class LoadQueue:
        def __init__(self, n_read_threads: int, n_fast_threads: int, block_size: int):
            self.n_read_threads = n_read_threads
            self.n_fast_threads = n_fast_threads
            self.jobs: dict[JobId, Any] = {}
            self.fcfs_q = SSDTPScheduler.FCFSQueue()
            self.sjf_q = SSDTPScheduler.SJFQueue(block_size)

        def add(self, item: Any, job_id: JobId, num_tasks: int):
            self.jobs[job_id] = item
            self.fcfs_q.add(job_id, num_tasks)
            self.sjf_q.add(job_id, num_tasks)

        def _get_job(self) -> tuple[Any, int | None]:
            # fetch from fcfs; job fetched from here is completely
            # removed from system
            job_id = None
            while self.jobs and job_id is None:
                job_id, num_tasks = self.fcfs_q.next()
                if job_id not in self.jobs:
                    # already grabbed by the fast threads
                    job_id = None
                    continue
                self.sjf_q._remove(job_id, num_tasks)

            if job_id is None:
                return None, None
            else:
                return self.jobs.pop(job_id), self.n_read_threads

        def _get_sjf_job(self, role: "SSDTPScheduler.Role") -> tuple[Any, int | None]:
            # jobs fetched here has leftover fcfs
            assert self.sjf_q.has_short_job()
            job_id, _ = self.sjf_q.next()
            n_threads = 1 if role == SSDTPScheduler.Role.WRITE else self.n_fast_threads
            return self.jobs.pop(job_id), n_threads

        def clear(self):
            self.fcfs_q.clear()
            self.sjf_q.clear()
            self.jobs.clear()

        def has_work(self):
            return len(self.jobs) != 0

        def has_short_job(self):
            return self.sjf_q.has_short_job()

        def fetch_work(self, role: "SSDTPScheduler.Role") -> tuple[Any, int | None]:
            if role == SSDTPScheduler.Role.READ:
                return self._get_job()
            else:
                return self._get_sjf_job(role)

    class StoreQueue:
        def __init__(self, block_size: int):
            self.jobs: dict[JobId, Any] = {}
            self.sjf_q = SSDTPScheduler.SJFQueue(block_size)

        def add(self, item: Any, job_id: JobId, num_tasks: int):
            self.sjf_q.add(job_id, num_tasks)
            self.jobs[job_id] = item

        def _get_sjf_job(self) -> tuple[Any, int | None]:
            job_id, _ = self.sjf_q.next()
            return self.jobs.pop(job_id), 1

        def clear(self):
            self.sjf_q.clear()
            self.jobs.clear()

        def has_work(self):
            return len(self.jobs) != 0

        def fetch_work(self, role: "SSDTPScheduler.Role") -> tuple[Any, int | None]:
            return self._get_sjf_job()

    ## Policy:
    #  * The read threads always batch
    #  * fast threads drains the small jobs from the load_q and
    #    executes it with batch_size 1
    #  * store threads always execute with batch_size 1 so the
    #    SSD dies dont lock up and prevent read/write contention
    #  * store threads join the fast threads in draining the small
    #    jobs.
    #  * fast threads and store threads, if they dont have work,
    #    they simply wait for incoming work.
    def __init__(self, n_read_threads: int, n_write_threads: int, block_size: int):
        # allocate 25% of the threads for the load fast queue
        self._n_fast_threads = n_read_threads // 4
        super().__init__(
            n_read_threads - self._n_fast_threads, n_write_threads, block_size
        )
        self._load_q: SSDTPScheduler.LoadQueue = SSDTPScheduler.LoadQueue(
            self._n_read_threads, self._n_fast_threads, self._block_size
        )
        self._store_q: SSDTPScheduler.StoreQueue = SSDTPScheduler.StoreQueue(
            self._block_size
        )
        print(
            f"SSDTPScheduler : \n"
            f"  - {self._n_read_threads=} \n"
            f"  - {self._n_write_threads=} \n"
            f"  - {self._n_fast_threads=} \n"
        )

        self._load_deque: deque[Any] = deque()
        self._fast_load_deque: deque[Any] = deque()
        # Role wakers: populated by make_wakers(), used directly in submit().
        self._role_wakers: dict[SSDTPScheduler.Role, ThreadWaker] = {}

    @property
    def total_threads(self) -> int:
        return self._n_read_threads + self._n_write_threads + self._n_fast_threads

    def make_wakers(self) -> dict[Any, ThreadWaker]:
        # One dedicated ThreadWaker per role.  notify() on Role.READ can only
        # wake READ threads, etc.  This eliminates spurious wakeups and the
        # starvation risk of a single shared waker.
        # Stored on self so submit() can call notify() directly.
        self._role_wakers = {
            SSDTPScheduler.Role.READ: ThreadWaker(),
            SSDTPScheduler.Role.FAST: ThreadWaker(),
            SSDTPScheduler.Role.WRITE: ThreadWaker(),
        }
        return self._role_wakers

    def make_threads(
        self,
        worker_fn: Callable,
        thread_name_prefix: str,
        wakers: dict[Any, ThreadWaker],
    ) -> list[tuple[threading.Thread, ThreadWaker]]:
        result: list[tuple[threading.Thread, ThreadWaker]] = []
        for i in range(self._n_read_threads):
            waker = wakers[SSDTPScheduler.Role.READ]
            t = threading.Thread(
                target=worker_fn,
                args=(SSDTPScheduler.Role.READ, waker),
                name=f"{thread_name_prefix}_l{i}",
                daemon=True,
            )
            t.start()
            result.append((t, waker))

        for i in range(self._n_write_threads):
            waker = wakers[SSDTPScheduler.Role.WRITE]
            t = threading.Thread(
                target=worker_fn,
                args=(SSDTPScheduler.Role.WRITE, waker),
                name=f"{thread_name_prefix}_s{i}",
                daemon=True,
            )
            t.start()
            result.append((t, waker))

        for i in range(self._n_fast_threads):
            waker = wakers[SSDTPScheduler.Role.FAST]
            t = threading.Thread(
                target=worker_fn,
                args=(SSDTPScheduler.Role.FAST, waker),
                name=f"{thread_name_prefix}_f{i}",
                daemon=True,
            )
            t.start()
            result.append((t, waker))

        return result

    def _batch_tasks(
        self,
        tasks: list[Any],
        n_threads: int,
    ) -> Iterator[list[Any]]:
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

    def submit(
        self, state: Any, make_batch_fn: Callable, tasks: list[Any], is_load: bool
    ) -> None:
        rw = self._role_wakers
        if is_load:
            self._load_q.add((make_batch_fn, tasks, state), state.job_id, len(tasks))
            # Always wake read threads — they batch and execute the job.
            rw[SSDTPScheduler.Role.READ].notify(self._n_read_threads)
            # If this submission created a multimodal size distribution, also
            # wake fast and write threads so they can drain the short job.
            if self._load_q.has_short_job():
                rw[SSDTPScheduler.Role.FAST].notify(self._n_fast_threads)
        else:
            self._store_q.add((make_batch_fn, tasks, state), state.job_id, len(tasks))
            # One store job is processed one-at-a-time (batch_size=1 policy).
            # Wake one write thread.  Also wake one fast thread because
            # has_work(FAST) includes store_q.has_work(), so fast threads can
            # drain stores when no short load job exists.
            rw[SSDTPScheduler.Role.WRITE].notify()
            rw[SSDTPScheduler.Role.FAST].notify()

    def has_work(self, role: "SSDTPScheduler.Role") -> bool:
        if role == SSDTPScheduler.Role.READ:
            return (
                bool(self._load_deque)
                or self._load_q.has_work()
                or self._store_q.has_work()
            )
        elif role == SSDTPScheduler.Role.FAST:
            return (
                bool(self._fast_load_deque)
                or self._load_q.has_short_job()
                or self._store_q.has_work()
            )
        else:
            return self._store_q.has_work()

    def fetch_work(self, role: "SSDTPScheduler.Role"):
        if role == SSDTPScheduler.Role.READ:
            if self._load_deque:
                return self._load_deque.popleft()

            q = self._load_q if self._load_q.has_work() else self._store_q
            work, n_batch = q.fetch_work(role)
            if work is None:
                return None
            assert n_batch is not None

            make_batch_fn, tasks, state = work
            if n_batch == 1:
                return (make_batch_fn(tasks), len(tasks), state)

            for b in self._batch_tasks(tasks, n_batch):
                self._load_deque.append((make_batch_fn(b), len(b), state))
            return self._load_deque.popleft()
        elif role == SSDTPScheduler.Role.FAST:
            if self._fast_load_deque:
                return self._fast_load_deque.popleft()

            q = self._load_q if self._load_q.has_short_job() else self._store_q
            work, n_batch = q.fetch_work(role)
            if work is None:
                return None
            assert n_batch is not None
            make_batch_fn, tasks, state = work
            if n_batch == 1:
                return (make_batch_fn(tasks), len(tasks), state)

            for b in self._batch_tasks(tasks, n_batch):
                self._fast_load_deque.append((make_batch_fn(b), len(b), state))
            return self._fast_load_deque.popleft()
        else:
            q = self._store_q
            if not q.has_work():
                return None

            assert role in [SSDTPScheduler.Role.WRITE]
            work, n_batch = q.fetch_work(role)
            if work is None:
                return None
            make_batch_fn, tasks, state = work
            assert n_batch == 1
            return (make_batch_fn(tasks), len(tasks), state)

    def clear(self):
        self._load_q.clear()
        self._store_q.clear()
        self._load_deque.clear()
        self._fast_load_deque.clear()


SCHEDULER_CLASS_MAPPING = {
    "NoBatchTPScheduler": NoBatchTPScheduler,
    "BatchTPScheduler": BatchTPScheduler,
    "SSDTPScheduler": SSDTPScheduler,
}
