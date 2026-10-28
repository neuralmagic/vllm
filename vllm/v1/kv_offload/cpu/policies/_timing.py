# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated function timing for cache policy debugging.

Set ``VLLM_CACHE_POLICY_TIMING=1`` to record wall times of decorated
functions; a per-function summary (count, p50/p95/p99, max) prints at
process exit. Without the env var the decorator is a no-op pass-through.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import os
import signal
import statistics
import time
from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")

_ENABLED = bool(os.environ.get("VLLM_CACHE_POLICY_TIMING"))
_SAMPLES: dict[str, list[float]] = {}


def timed(fn: Callable[..., _T]) -> Callable[..., _T]:
    """Record per-call wall time of ``fn`` when timing is enabled."""
    if not _ENABLED:
        return fn
    name = fn.__qualname__

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _SAMPLES.setdefault(name, []).append(time.perf_counter() - t0)

    return wrapper


def _percentiles(samples: list[float]) -> tuple[float, float, float]:
    if len(samples) < 20:
        med = statistics.median(samples)
        return (med, med, med)
    qs = statistics.quantiles(samples, n=100)
    return (statistics.median(samples), qs[94], qs[98])


def _report() -> None:
    if not _SAMPLES:
        if _ENABLED:
            print(
                "\ncache policy timing enabled but no samples recorded "
                "(no decorated function ran in this process)",
                flush=True,
            )
        return
    print("\ncache policy timing summary", flush=True)
    header = (
        f"{'function':<40}{'calls':>10}{'p50':>12}{'p95':>12}{'p99':>12}{'max':>12}"
    )
    print(header)
    print("-" * len(header))
    for name in sorted(_SAMPLES):
        samples = _SAMPLES[name]
        p50, p95, p99 = _percentiles(samples)
        print(
            f"{name:<40}{len(samples):>10}"
            f"{p50 * 1e6:>10.1f}us{p95 * 1e6:>10.1f}us"
            f"{p99 * 1e6:>10.1f}us{max(samples) * 1e6:>10.1f}us",
            flush=True,
        )


if _ENABLED:
    atexit.register(_report)

    # child processes are torn down with SIGTERM (proc.terminate()), which
    # skips atexit; report there too, then restore the default disposition
    def _report_on_sigterm(signum, frame):
        _report()
        os._exit(128 + signum)

    # not the main thread of the main interpreter
    with contextlib.suppress(ValueError):
        signal.signal(signal.SIGTERM, _report_on_sigterm)
