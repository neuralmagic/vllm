# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Function timing for cache policy debugging.

Wall times of decorated functions are recorded by default. Timings
are:

- summarized (count, p50/p95/p99, max) at process exit in a single
  atomic write (SIGTERM included)
- written raw to a CSV (one row per call:
  ``elapsed_s,function,duration_ns``), flushed by a background thread
  every 450s and at process exit

The wrapper's hot path is two perf_counter calls and two appends; all
file writing happens on the background thread.

Optional env vars:
- ``VLLM_CACHE_POLICY_TIMING_OUT``: summary checkpoint path
  (default ``cache_policy_timing_<pid>.txt`` in the working directory)
- ``VLLM_CACHE_POLICY_TIMING_RAW``: overrides the raw CSV path (it is
  always written; default ``cache_policy_timing_<pid>.csv`` in the
  working directory)
- ``VLLM_CACHE_POLICY_TIMING_INTERVAL``: flush/checkpoint seconds
  (default 450)

Set ``VLLM_CACHE_POLICY_TIMING=0`` to disable; then the decorator is
a no-op pass-through.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import os
import signal
import statistics
import threading
import time
from array import array
from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")

_ENABLED = os.environ.get("VLLM_CACHE_POLICY_TIMING", "1") != "0"
_SAMPLES: dict[str, array] = {}
_RAW: list[tuple[float, str, float]] = []
_RAW_LOCK = threading.Lock()
_T0 = time.perf_counter()
_SUMMARY_PATH = os.environ.get("VLLM_CACHE_POLICY_TIMING_OUT") or (
    f"cache_policy_timing_{os.getpid()}.txt"
)
_RAW_PATH = os.environ.get("VLLM_CACHE_POLICY_TIMING_RAW") or (
    f"cache_policy_timing_{os.getpid()}.csv"
)
_INTERVAL = float(os.environ.get("VLLM_CACHE_POLICY_TIMING_INTERVAL", "450"))


def timed(fn: Callable[..., _T]) -> Callable[..., _T]:
    """Record per-call wall time of ``fn`` when timing is enabled."""
    if not _ENABLED:
        return fn
    name = fn.__qualname__
    samples = array("d")
    _SAMPLES[name] = samples

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # hot path: two perf_counter calls and two appends; everything
        # else (formatting, file writes) happens on the background thread
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            samples.append(elapsed)
            _RAW.append((t0 - _T0, name, elapsed))

    return wrapper


def _flush_raw() -> None:
    """Append buffered raw records to the CSV."""
    with _RAW_LOCK:
        if not _RAW:
            return
        records, _RAW[:] = _RAW[:], []
    try:
        with open(_RAW_PATH, "a") as f:
            f.write("elapsed_s,function,duration_ns\n")
            f.writelines(
                f"{t:.6f},{name},{dur * 1e9:.0f}\n" for t, name, dur in records
            )
    except OSError:
        pass


def _percentiles(samples: array) -> tuple[float, float, float]:
    if len(samples) < 20:
        med = statistics.median(samples)
        return (med, med, med)
    qs = statistics.quantiles(samples, n=100)
    return (statistics.median(samples), qs[94], qs[98])


def _table() -> str:
    header = (
        f"{'function':<40}{'calls':>10}{'p50':>12}{'p95':>12}{'p99':>12}{'max':>12}"
    )
    lines = [
        f"cache policy timing summary (pid={os.getpid()})",
        header,
        "-" * len(header),
    ]
    for name in sorted(_SAMPLES):
        samples = _SAMPLES[name]
        p50, p95, p99 = _percentiles(samples)
        lines.append(
            f"{name:<40}{len(samples):>10}"
            f"{p50 * 1e6:>10.1f}us{p95 * 1e6:>10.1f}us"
            f"{p99 * 1e6:>10.1f}us{max(samples) * 1e6:>10.1f}us"
        )
    return "\n".join(lines) + "\n"


def _checkpoint() -> None:
    """Write the summary table to its checkpoint file, rename-atomic."""
    tmp = _SUMMARY_PATH + ".tmp"
    try:
        with open(tmp, "w") as f:
            f.write(f"# updated {time.strftime('%H:%M:%S')}\n" + _table())
        os.replace(tmp, _SUMMARY_PATH)
    except OSError:
        pass


def _report() -> None:
    if not _SAMPLES:
        if _ENABLED:
            print(
                "\ncache policy timing enabled but no samples recorded "
                "(no decorated function ran in this process)",
                flush=True,
            )
        return
    # one write so a SIGKILL cannot truncate the table mid-print
    # (stays under PIPE_BUF, keeping the pipe write atomic)
    os.write(1, ("\n" + _table()).encode())
    _checkpoint()
    _flush_raw()


def _flush_loop() -> None:
    while True:
        time.sleep(_INTERVAL)
        _flush_raw()
        if _SAMPLES:
            _checkpoint()


if _ENABLED:
    _flush_raw()  # emit the CSV header immediately
    atexit.register(_report)

    # child processes are torn down with SIGTERM (proc.terminate()), which
    # skips atexit; report there too, then exit with the conventional code
    def _report_on_sigterm(signum, frame):
        _report()
        os._exit(128 + signum)

    # not the main thread of the main interpreter
    with contextlib.suppress(ValueError):
        signal.signal(signal.SIGTERM, _report_on_sigterm)

    threading.Thread(target=_flush_loop, daemon=True).start()
