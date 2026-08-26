# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in GPU memory attribution (VLLM_MEM_TRACE=1): logs device-used /
torch-reserved / non-torch memory at tagged points so growth that the memory
profiler only reports as a lump "non-torch" can be attributed to an owner."""

import functools
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED = os.environ.get("VLLM_MEM_TRACE", "0") == "1"
_SNAPSHOT = os.environ.get("VLLM_MEM_TRACE_SNAPSHOT", "0") == "1"
_MIN_MIB = int(os.environ.get("VLLM_MEM_TRACE_MIN_MIB", "32"))
_seen: set[str] = set()
_last: dict[str, float] = {}


def mem_trace(tag: str, once: bool = False) -> None:
    if not _ENABLED or not torch.cuda.is_available():
        return
    if once:
        if tag in _seen:
            return
        _seen.add(tag)
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    used = total - free
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    non_torch = used - reserved
    prev = _last.get("non_torch")
    delta = "" if prev is None else f" d_non_torch={(non_torch - prev) / 2**30:+.2f}"
    _last["non_torch"] = non_torch
    logger.info(
        "MEMTRACE %-34s used=%6.2f reserved=%6.2f allocated=%6.2f non_torch=%6.2f GiB%s",
        tag, used / 2**30, reserved / 2**30, allocated / 2**30, non_torch / 2**30, delta,
    )


def mem_record_start() -> None:
    """Start recording allocator history (VLLM_MEM_TRACE_SNAPSHOT=1)."""
    if _ENABLED and _SNAPSHOT and torch.cuda.is_available():
        torch.cuda.memory._record_memory_history(max_entries=300000)
        logger.info("MEMTRACE recording allocator history")


def mem_snapshot_report(tag: str, top: int = 40) -> None:
    """Log live torch allocations >= VLLM_MEM_TRACE_MIN_MIB grouped by the
    innermost vllm frames of their allocating stack."""
    if not (_ENABLED and _SNAPSHOT and torch.cuda.is_available()):
        return
    torch.cuda.synchronize()
    snap = torch.cuda.memory._snapshot()
    groups: dict[str, list[int]] = {}
    total = 0
    for seg in snap.get("segments", []):
        for blk in seg.get("blocks", []):
            if blk.get("state") != "active_allocated":
                continue
            size = blk.get("size", 0)
            total += size
            if size < _MIN_MIB * 2**20:
                continue
            frames = blk.get("frames") or []
            vf = [
                f"{os.path.basename(os.path.dirname(f['filename']))}/"
                f"{os.path.basename(f['filename'])}:{f['line']}:{f['name']}"
                for f in frames
                if "/vllm/" in f.get("filename", "")
                and "mem_trace" not in f.get("filename", "")
            ]
            key = " <- ".join(vf[:4]) if vf else "(no vllm frame)"
            g = groups.setdefault(key, [0, 0])
            g[0] += size
            g[1] += 1
    logger.info(
        "MEMSNAP %s live torch allocations: %.2f GiB total; top %d groups (>= %d MiB blocks):",
        tag, total / 2**30, top, _MIN_MIB,
    )
    for key, (size, n) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:top]:
        logger.info("MEMSNAP   %7.2f GiB  x%-3d %s", size / 2**30, n, key)


def mem_trace_once(tag: str):
    """Decorator: trace before/after the first call of the wrapped function."""

    def deco(fn):
        if not _ENABLED:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            first = f"{tag}:before" not in _seen
            if first:
                mem_trace(f"{tag}:before", once=True)
            out = fn(*args, **kwargs)
            if first:
                mem_trace(f"{tag}:after", once=True)
            return out

        return wrapper

    return deco
