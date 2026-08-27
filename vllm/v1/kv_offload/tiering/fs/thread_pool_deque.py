# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dequeue definitions for thread_pool work storing
"""

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any


class _TPDeque(ABC):
    def __init__(self, n_threads: int):
        self._n_threads = n_threads

    @abstractmethod
    def submit(
        self, tasks: list[Any], make_batch_fn: Callable[[list[Any]], Callable[[], None]]
    ):
        pass

    @abstractmethod
    def fetch(self):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class TPDeque(_TPDeque):
    def __init__(self, n_threads: int):
        super().__init__(n_threads)
        self.q: deque = deque()

    def submit(
        self, tasks: list[Any], make_batch_fn: Callable[[list[Any]], Callable[[], None]]
    ) -> int:
        self.q.append((make_batch_fn(tasks), tasks))
        return 1

    def fetch(self):
        return self.q.popleft()

    def clear(self):
        self.q.clear()

    def __len__(self) -> int:
        return len(self.q)


class TPDequeBalancedBatch(_TPDeque):
    def __init__(self, n_threads: int):
        super().__init__(n_threads)
        self._qs: list[list[Any]] = [[] for _ in range(self._n_threads)]
        self._fetch_cursor: int = 0
        # admits only one type of work. The make_batch_fn delivered via
        # submit must always be the same.
        self._make_batch_fn: None | Callable[[list[Any]], Callable[[], None]] = None

        # running counter to make __len__ inexpensive
        self._len = 0

    def _pop(self) -> list[Any]:
        q = self._qs[self._fetch_cursor]
        self._qs[self._fetch_cursor] = []

        # update cursor
        self._fetch_cursor += 1
        self._fetch_cursor = self._fetch_cursor % self._n_threads
        # update len
        self._len -= len(q)

        return q

    def _batch_tasks(
        self,
        tasks: list[Any],
    ) -> Iterator[list[Any]]:
        """
        Batch tasks so that the request's tasks are split evenly across the
        n_threads.
        """
        assert self._n_threads > 0

        n_tasks = len(tasks)
        q, r = divmod(n_tasks, self._n_threads)
        batch_sizes = [q + 1 if i < r else q for i in range(self._n_threads)]
        assert sum(batch_sizes) == n_tasks
        start = 0
        for bs in batch_sizes[: min(n_tasks, self._n_threads)]:
            yield tasks[start : start + bs]
            start += bs

    def submit(
        self, tasks: list[Any], make_batch_fn: Callable[[list[Any]], Callable[[], None]]
    ) -> int:
        if self._make_batch_fn is None:
            self._make_batch_fn = make_batch_fn
        # Balanced batching requires all make_batch_fn be the same as
        # jobs across submit calls may be batched together
        assert self._make_batch_fn == make_batch_fn

        # Split batch equally amongst all threads.
        n_batches = 0
        for idx, ts in enumerate(self._batch_tasks(tasks)):
            self._qs[idx].extend(ts)
            n_batches += 1

        # update len
        self._len += len(tasks)

        return n_batches

    def fetch(self):
        # make batch out of _fetch_cursor and return
        # pop a queue
        tasks = self._pop()
        assert self._make_batch_fn is not None
        return (self._make_batch_fn(tasks), tasks)

    def clear(self):
        for i in range(self._n_threads):
            self._qs[i] = []
        self._fetch_cursor = 0
        self._make_batch_fn = None
        self._len = 0

    def __len__(self) -> int:
        return self._len
