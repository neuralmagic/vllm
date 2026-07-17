# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
import random
import resource
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# O_DIRECT is Linux-specific and not available on macOS
O_DIRECT = getattr(os, "O_DIRECT", 0)

# RUSAGE_THREAD is Linux-specific and not available on macOS
_HAS_RUSAGE_THREAD = hasattr(resource, "RUSAGE_THREAD")

# How often (in seconds) each thread flushes its buffered FSIO_TIMING lines.
# Raw (unaggregated) lines are buffered rather than logged per-task so the
# series can be parsed later to plot wall/cpu time and context-switch counts
# over time, without paying a logging call per I/O task.
_TIMING_FLUSH_INTERVAL_S = 10.0

# Thread-local storage for unique temporary file suffixes and timing state
_thread_local = threading.local()


_RusageSnapshot = tuple[float, "resource.struct_rusage"]


def _timing_start() -> _RusageSnapshot | None:
    if not _HAS_RUSAGE_THREAD:
        return None
    return time.perf_counter(), resource.getrusage(resource.RUSAGE_THREAD)


def _timing_end(kind: str, start: _RusageSnapshot | None) -> None:
    """Record one FSIO_TIMING line for a completed store/load callback.

    ``wall`` is the callback's wall-clock duration. ``cpu`` is the CPU time
    this thread actually spent executing over that span
    (``ru_utime + ru_stime``); any remaining wall time was spent off-CPU,
    either blocked on disk I/O or waiting to (re)acquire the GIL —
    indistinguishable from each other using these numbers alone.

    ``nvcsw``/``nivcsw`` are voluntary/involuntary context-switch deltas:
    ``nvcsw`` counts times this thread went to sleep (disk I/O wait or
    GIL-reacquire wait, also indistinguishable from each other), while
    ``nivcsw`` counts times the OS preempted it while still runnable
    (CPU/scheduler oversubscription, unrelated to the GIL).

    Lines are buffered per-thread and flushed periodically rather than
    logged immediately, to avoid a logging call per I/O task.
    """
    if start is None:
        return
    t0, r0 = start
    t1 = time.perf_counter()
    r1 = resource.getrusage(resource.RUSAGE_THREAD)

    line = (
        "FSIO_TIMING ts=%.6f thread=%s kind=%s wall_ms=%.3f cpu_ms=%.3f "
        "nvcsw=%d nivcsw=%d"
        % (
            time.time(),
            threading.current_thread().name,
            kind,
            (t1 - t0) * 1e3,
            ((r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime)) * 1e3,
            r1.ru_nvcsw - r0.ru_nvcsw,
            r1.ru_nivcsw - r0.ru_nivcsw,
        )
    )

    try:
        buf: list[str] = _thread_local.timing_buf
    except AttributeError:
        buf = _thread_local.timing_buf = []
        _thread_local.timing_last_flush = t1
    buf.append(line)

    if t1 - _thread_local.timing_last_flush >= _TIMING_FLUSH_INTERVAL_S:
        logger.debug("\n".join(buf))
        buf.clear()
        _thread_local.timing_last_flush = t1


@contextmanager
def _timed(kind: str) -> Iterator[None]:
    """Time a single sub-operation (e.g. one os.* call) and record it.

    Uses try/finally so a sub-operation that raises (e.g. a short write)
    still gets its timing recorded before the exception propagates.
    """
    start = _timing_start()
    try:
        yield
    finally:
        _timing_end(kind, start)


def _get_tmp_suffix() -> str:
    """Generate a thread-local unique suffix for temporary files."""
    try:
        return _thread_local.tmp_suffix
    except AttributeError:
        _thread_local.tmp_suffix = f"_{random.randint(0, 2**63 - 1)}.tmp"
        return _thread_local.tmp_suffix


def _ensure_dirs(path: str) -> None:
    """Create parent directories of *path* if they don't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def store_block(
    dest_path: str,
    buffer: memoryview,
    offset: int,
    block_size: int,
) -> None:
    """
    Store callback: Writes to a temp file then atomically replaces the destination.
    """
    # Check if block already exists to avoid redundant writes
    if os.path.exists(dest_path):
        return None

    _t_total = _timing_start()

    tmp_path = dest_path + _get_tmp_suffix()
    # Ensure parent directories exist
    with _timed("store.ensure_dirs"):
        _ensure_dirs(dest_path)

    # Write block atomically. Cast to a flat byte view so the slice uses byte
    # indices; the raw memoryview may be multi-dimensional with itemsize > 1.
    view_slice = buffer.cast("B")[offset : offset + block_size]
    try:
        with _timed("store.open"):
            fd = os.open(
                tmp_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC | O_DIRECT,
                0o644,
            )
        try:
            with _timed("store.write"):
                written = os.write(fd, view_slice)
            if written < len(view_slice):
                raise OSError(
                    f"Short write: expected {len(view_slice)} bytes, wrote {written}"
                )
        finally:
            with _timed("store.close"):
                os.close(fd)
        with _timed("store.replace"):
            os.replace(tmp_path, dest_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError as cleanup_exc:
            logger.warning("Failed to remove temp file %s: %s", tmp_path, cleanup_exc)
        raise

    _timing_end("store.total", _t_total)


def load_block(
    source_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
) -> None:
    """
    Load callback: read one KV block from disk. Remove the file on failure.
    """
    _t_total = _timing_start()

    fd: int | None = None
    view_slice = view.cast("B")[offset : offset + block_size]
    try:
        with _timed("load.open"):
            fd = os.open(source_path, os.O_RDONLY | O_DIRECT)
        with _timed("load.read"):
            bytes_read = os.readv(fd, [view_slice])
        if bytes_read < block_size:
            raise OSError(f"Short read: expected {block_size} bytes, read {bytes_read}")
    except Exception:
        try:
            os.remove(source_path)
        except OSError as cleanup_exc:
            logger.warning(
                "Failed to remove unreadable file %s: %s", source_path, cleanup_exc
            )
        raise
    finally:
        if fd is not None:
            with _timed("load.close"):
                os.close(fd)

    _timing_end("load.total", _t_total)
