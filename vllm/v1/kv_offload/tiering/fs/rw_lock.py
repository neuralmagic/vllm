# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class RWLock:
    """Writer-preferring readers-writer lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._active_readers = 0
        self._active_writer = False
        self._waiting_writers = 0

    @contextmanager
    def read_locked(self) -> Iterator[None]:
        with self._condition:
            self._condition.wait_for(
                lambda: not self._active_writer and self._waiting_writers == 0
            )
            self._active_readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._active_readers -= 1
                if self._active_readers == 0:
                    self._condition.notify_all()

    @contextmanager
    def write_locked(self) -> Iterator[None]:
        with self._condition:
            self._waiting_writers += 1
            self._condition.wait_for(
                lambda: not self._active_writer and self._active_readers == 0
            )
            self._waiting_writers -= 1
            self._active_writer = True
        try:
            yield
        finally:
            with self._condition:
                self._active_writer = False
                self._condition.notify_all()
