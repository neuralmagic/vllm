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
    """No-op kept for call-site compatibility (allocator history recording was
    far too slow on a 100+ GiB model; the report below walks live tensors)."""
    return


def mem_snapshot_report(tag: str, top: int = 30) -> None:
    """Log live CUDA tensors grouped by shape/dtype (VLLM_MEM_TRACE_SNAPSHOT=1).

    Run at the end of the profile run, after activations are freed, so what is
    left is the persistent buffers: the ones sized from
    max_num_batched_tokens are visible directly by their leading dimension.
    """
    if not (_ENABLED and _SNAPSHOT and torch.cuda.is_available()):
        return
    import gc

    torch.cuda.synchronize()
    gc.collect()
    groups: dict[tuple, list] = {}
    seen: set[int] = set()
    total = 0
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor) or not obj.is_cuda:
                continue
            storage = obj.untyped_storage()
            ptr = storage.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            nbytes = storage.nbytes()
        except Exception:
            continue
        if nbytes < _MIN_MIB * 2**20:
            total += nbytes
            continue
        total += nbytes
        key = (tuple(obj.shape), str(obj.dtype), isinstance(obj, torch.nn.Parameter))
        entry = groups.setdefault(key, [0, 0])
        entry[0] += nbytes
        entry[1] += 1
    ranked = sorted(groups.items(), key=lambda kv: -kv[1][0])[:top]
    logger.info(
        "MEMSNAP %s live CUDA tensors: %.2f GiB total, %d storages; "
        "top %d groups of blocks >= %d MiB:",
        tag, total / 2**30, len(seen), len(ranked), _MIN_MIB,
    )
    for (shape, dtype, is_param), (size, count) in ranked:
        logger.info(
            "MEMSNAP   %7.2f GiB  x%-4d %-8s %s %s",
            size / 2**30, count, "param" if is_param else "buffer", dtype, shape,
        )


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
