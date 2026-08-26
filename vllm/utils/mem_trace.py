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
