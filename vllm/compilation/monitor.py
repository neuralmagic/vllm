# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from contextlib import contextmanager

import torch  # type: ignore
from typing import Any, Dict, List, Tuple, cast

from vllm.config import CompilationConfig, CompilationLevel, VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

context_manager = None
torch_compile_start_time: float = 0.0


def start_monitoring_torch_compile(vllm_config: VllmConfig):
    global torch_compile_start_time
    torch_compile_start_time = time.time()

    compilation_config: CompilationConfig = vllm_config.compilation_config
    if compilation_config.level == CompilationLevel.PIECEWISE and \
        compilation_config.debug_dump_path:
        import depyf
        path = os.path.join(compilation_config.debug_dump_path,
                            f"rank_{vllm_config.parallel_config.rank}")
        global context_manager
        context_manager = depyf.prepare_debug(path)
        context_manager.__enter__()


def end_monitoring_torch_compile(vllm_config: VllmConfig):
    compilation_config: CompilationConfig = vllm_config.compilation_config
    if compilation_config.level == CompilationLevel.PIECEWISE:
        logger.info("torch.compile takes %.2f s in total",
                    compilation_config.compilation_time)
        global context_manager
        if context_manager is not None:
            context_manager.__exit__(None, None, None)
            context_manager = None


cudagraph_capturing_enabled: bool = True


def validate_cudagraph_capturing_enabled():
    # used to monitor whether an cudagraph capturing is legal at runtime.
    # should be called before any cudagraph capturing.
    # if an illegal cudagraph capturing happens, raise an error.
    global cudagraph_capturing_enabled
    if not cudagraph_capturing_enabled:
        raise RuntimeError("CUDA graph capturing detected at an inappropriate "
                           "time. This operation is currently disabled.")


def set_cudagraph_capturing_enabled(enabled: bool):
    global cudagraph_capturing_enabled
    cudagraph_capturing_enabled = enabled


# -------------------------------
# CUDA allocator history helpers
# -------------------------------

def enable_alloc_history(max_entries: int = 400_000) -> None:
    """Enable CUDA allocator history if available.

    This allows attributing allocations (including graph/private pool) to
    python callsites via torch.cuda.memory_snapshot(). Safe to call multiple
    times.
    """
    # Set a reasonable default allocator config (no record_history here, some
    # Torch builds don't recognize it via env).
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:4096",
    )
    # Optionally, try to enable runtime history if explicitly requested.
    # This can populate frames in _snapshot for inactive blocks.
    if os.environ.get("VLLM_DEBUG_ENABLE_ALLOC_HISTORY", "0") == "1":
        try:
            # Accept common spellings across torch versions
            rec: Any = torch.cuda.memory._record_memory_history  # type: ignore[attr-defined]
            try:
                rec(enabled="state", max_entries=max_entries)
            except TypeError:
                rec(enabled=True, max_entries=max_entries)  # type: ignore[arg-type]
        except Exception:
            pass
    # Rely on env var; skip direct runtime call to avoid cross-version typing issues.
    # Users can still opt-in via PYTORCH_CUDA_ALLOC_CONF record_history=1.


def _frames_to_key(frames) -> str:
    # Prefer python frames. Fall back to the first frame (often C++ symbolized)
    for fr in frames or []:
        filename = fr.get("filename", "")
        if filename.endswith(".py"):
            return f"{filename}:{fr.get('line','?')}:{fr.get('name','?')}"
    if frames:
        fr = frames[0]
        return f"{fr.get('filename','<c++>')}:{fr.get('line','?')}:{fr.get('name','?')}"
    return "<no-python-frame>"


def graph_pool_usage_by_callsite(segment_types=("graph", "private")) -> dict:
    """Summarize graph/private pool usage by callsite and scope labels.

    Returns a dict keyed by device id with totals, top callsites, and scopes.
    """
    snap = torch.cuda.memory_snapshot()
    per_dev_total: Dict[int, int] = {}
    per_dev_callsite: Dict[int, Dict[str, int]] = {}
    per_dev_scopes: Dict[int, Dict[str, int]] = {}

    def _accumulate_for_segment_types(candidates: tuple[str, ...]) -> bool:
        matched = False
        for seg in snap:
            seg_type = seg.get("segment_type") or seg.get("segment_kind") or "unknown"
            seg_type_l = str(seg_type).lower()
            # Fuzzy match: exact or substring match against provided candidates
            if not any((ct in seg_type_l) or (ct == seg_type) for ct in candidates):
                continue
            matched = True
            dev = seg.get("device")
            for b in seg.get("blocks", []):
                size = int(b.get("size", 0))
                state = b.get("state", "")
                # Accept common spellings across torch versions and include inactive blocks
                # so we can see reserved pool capacity immediately after capture.
                valid_states = {"active", "active_allocated", "inactive_allocated", "inactive"}
                if size <= 0 or state not in valid_states:
                    continue
                frames = b.get("frames") or (b.get("history", {}) or {}).get("frames")
                key = _frames_to_key(frames)
                per_dev_total[dev] = per_dev_total.get(dev, 0) + size
                callsite_map = per_dev_callsite.setdefault(dev, {})
                callsite_map[key] = callsite_map.get(key, 0) + size
                for sc in (
                    b.get("scopes")
                    or (b.get("history", {}) or {}).get("scopes")
                    or []
                ):
                    scope_map = per_dev_scopes.setdefault(dev, {})
                    scope_map[sc] = scope_map.get(sc, 0) + size
        return matched

    # First attempt: user-specified segment types
    if not _accumulate_for_segment_types(tuple(segment_types)):
        # Fallback: fuzzy-match common graph/private segment names
        _accumulate_for_segment_types(("graph", "cudagraph", "priv"))

    report: dict = {}
    for dev in per_dev_total:
        total = per_dev_total[dev] / 1024 ** 2
        calls = sorted(per_dev_callsite[dev].items(), key=lambda kv: kv[1], reverse=True)
        scopes = sorted(per_dev_scopes[dev].items(), key=lambda kv: kv[1], reverse=True)
        report[dev] = {
            "total_graph_like_MiB": total,
            "top_callsites": [(k, v / 1024 ** 2) for k, v in calls[:30]],
            "top_scopes": [(k, v / 1024 ** 2) for k, v in scopes[:30]],
        }
    return report


def snapshot_segment_summary() -> dict:
    """Return a minimal summary of allocator segments for debugging.

    Structure:
      {
        'segment_types': {type_name: count, ...}
      }
    """
    types: Dict[str, int] = {}
    for seg in torch.cuda.memory_snapshot():
        t = str(seg.get("segment_type") or seg.get("segment_kind") or "unknown").lower()
        types[t] = types.get(t, 0) + 1
    return {"segment_types": types}


def _frame_key_from_list(frames: Any) -> str:
    if isinstance(frames, list) and frames:
        for fr in frames:
            filename = (fr.get("filename") if isinstance(fr, dict) else None) or ""
            if filename.endswith(".py"):
                line = fr.get("line", "?") if isinstance(fr, dict) else "?"
                name = fr.get("name", "?") if isinstance(fr, dict) else "?"
                return f"{filename}:{line}:{name}"
        fr0 = frames[0]
        if isinstance(fr0, dict):
            return f"{fr0.get('filename','<c++>')}:{fr0.get('line','?')}:{fr0.get('name','?')}"
    return "<no-python-frame>"


def inactive_usage_by_callsite_via_snapshot(topn: int = 30) -> Dict[int, List[Tuple[str, float]]]:
    """Attribute inactive blocks to callsites using torch.cuda.memory._snapshot().

    Useful on builds where memory_snapshot does not expose 'graph'/'private'.
    Returns per-device list of (callsite, MiB) sorted by size.
    """
    try:
        snap = torch.cuda.memory._snapshot()  # type: ignore[attr-defined]
    except Exception:
        return {}

    per_dev: Dict[int, Dict[str, int]] = {}
    for seg in snap.get("segments", []):
        dev = seg.get("device") if isinstance(seg, dict) else None
        if not isinstance(dev, int):
            # Some torch versions do not include device here; fall back to 0
            dev = 0
        blocks = seg.get("blocks", []) if isinstance(seg, dict) else []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("state") != "inactive":
                continue
            size = int(b.get("size", 0))
            if size <= 0:
                continue
            key = _frame_key_from_list(b.get("frames"))
            dev_map = per_dev.setdefault(dev, {})
            dev_map[key] = dev_map.get(key, 0) + size

    result: Dict[int, List[Tuple[str, float]]] = {}
    for dev, m in per_dev.items():
        items = sorted(m.items(), key=lambda kv: kv[1], reverse=True)[:topn]
        result[dev] = [(k, v / 1024 ** 2) for k, v in items]
    return result


def top_inactive_blocks_via_snapshot(topn: int = 10) -> Dict[int, List[Dict[str, Any]]]:
    """Return top-N inactive blocks per device with size/requested/address.

    Helps debug who is holding large free chunks in the allocator when frames
    are unavailable.
    """
    try:
        snap = torch.cuda.memory._snapshot()  # type: ignore[attr-defined]
    except Exception:
        return {}
    per_dev: Dict[int, List[Dict[str, Any]]] = {}
    for seg in snap.get("segments", []):
        dev = seg.get("device") if isinstance(seg, dict) else None
        if not isinstance(dev, int):
            dev = 0
        for b in seg.get("blocks", []) if isinstance(seg, dict) else []:
            if not isinstance(b, dict):
                continue
            if b.get("state") != "inactive":
                continue
            entry = {
                "size_MiB": float(b.get("size", 0)) / 1024 ** 2,
                "requested_MiB": float(b.get("requested_size", 0)) / 1024 ** 2,
                "address": hex(int(b.get("address", 0))) if b.get("address") is not None else None,
            }
            per_dev.setdefault(dev, []).append(entry)
    for dev, rows in per_dev.items():
        rows.sort(key=lambda r: r.get("size_MiB", 0.0), reverse=True)
        per_dev[dev] = rows[:topn]
    return per_dev


def external_usage(device: int | None = None) -> Dict[str, float]:
    """Driver vs torch allocator usage for a device.

    external_MiB ~= driver_used - torch_reserved (memory not tracked by torch allocator).
    """
    if device is None:
        device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    driver_used = total - free
    external = max(0, driver_used - reserved)
    mib = 1024.0 ** 2
    return {
        "device": float(device),
        "driver_total_MiB": total / mib,
        "driver_used_MiB": driver_used / mib,
        "torch_reserved_MiB": reserved / mib,
        "torch_allocated_MiB": allocated / mib,
        "external_MiB": external / mib,
    }


@contextmanager
def alloc_tag(label: str):
    """Tag subsequent allocations with a debug label if supported.

    Also emits a profiler record_function scope so the label appears in traces.
    """
    try:
        torch.cuda.memory._set_allocator_debug_context(label)  # type: ignore[attr-defined]
    except Exception:
        pass
    with torch.autograd.profiler.record_function(label):  # type: ignore[attr-defined]
        try:
            yield
        finally:
            try:
                torch.cuda.memory._set_allocator_debug_context("")  # type: ignore[attr-defined]
            except Exception:
                pass
