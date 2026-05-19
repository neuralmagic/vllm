# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Per-step per-layer timing collector for DeepEP dispatch/combine and MoE
expert compute.

Uses CUDA driver API events with CU_EVENT_RECORD_EXTERNAL to produce valid
timestamps even inside CUDA graph replays. Gated by VLLM_MOE_TIMING_ENABLED.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.metrics.stats import DeepEPStats

try:
    from cuda.bindings import driver as cuda_drv

    _CUDA_BINDINGS_AVAILABLE = True
except ImportError:
    cuda_drv = None  # type: ignore[assignment]
    _CUDA_BINDINGS_AVAILABLE = False

CU_EVENT_RECORD_EXTERNAL = 0x1


class _DriverEvent:
    """CUDA driver event with automatic graph-compatible recording."""

    __slots__ = ("_event",)

    def __init__(self):
        err, self._event = cuda_drv.cuEventCreate(0)  # CU_EVENT_DEFAULT

    def record(self):
        stream = torch.cuda.current_stream().cuda_stream
        if torch.cuda.is_current_stream_capturing():
            cuda_drv.cuEventRecordWithFlags(
                self._event, stream, CU_EVENT_RECORD_EXTERNAL
            )
        else:
            cuda_drv.cuEventRecord(self._event, stream)

    def elapsed_time(self, end: _DriverEvent) -> float | None:
        """Returns milliseconds, or None if events are not yet complete."""
        err, ms = cuda_drv.cuEventElapsedTime(self._event, end._event)
        if err != cuda_drv.CUresult.CUDA_SUCCESS:
            return None
        return ms


@dataclass(slots=True)
class _EventRecord:
    start: _DriverEvent
    end: _DriverEvent
    phase: str  # "dispatch", "combine", or "expert_compute"
    layer_idx: int


class DeepEPTimingCollector:
    """Collects per-layer CUDA event pairs and computes elapsed times.

    Usage:
        1. In layer __init__: call enable() and register_layer()
        2. In forward: call get_event_pair(), record start/end, then record_*()
        3. After forward: call finish_step() to get per-layer stats
    """

    def __init__(self):
        self._pending_events: list[_EventRecord] = []
        self._graph_events: list[_EventRecord] | None = None
        self._enabled = False
        self._mode = ""
        self._next_layer_idx = 0

    def enable(self, mode: str = ""):
        if not _CUDA_BINDINGS_AVAILABLE:
            from vllm.logger import init_logger

            logger = init_logger(__name__)
            logger.warning(
                "cuda.bindings not available; MoE timing disabled"
            )
            return
        self._enabled = True
        self._mode = mode

    @property
    def enabled(self) -> bool:
        return self._enabled

    def register_layer(self) -> int:
        """Assign sequential layer index. Called once per MoE layer."""
        idx = self._next_layer_idx
        self._next_layer_idx += 1
        return idx

    def get_event_pair(self) -> tuple[_DriverEvent, _DriverEvent]:
        """Create a start/end event pair for timing a phase."""
        return _DriverEvent(), _DriverEvent()

    def record_dispatch(
        self, start: _DriverEvent, end: _DriverEvent, layer_idx: int
    ):
        if self._enabled:
            self._pending_events.append(
                _EventRecord(
                    start=start, end=end, phase="dispatch", layer_idx=layer_idx
                )
            )

    def record_combine(
        self, start: _DriverEvent, end: _DriverEvent, layer_idx: int
    ):
        if self._enabled:
            self._pending_events.append(
                _EventRecord(
                    start=start, end=end, phase="combine", layer_idx=layer_idx
                )
            )

    def record_expert_compute(
        self, start: _DriverEvent, end: _DriverEvent, layer_idx: int
    ):
        if self._enabled:
            self._pending_events.append(
                _EventRecord(
                    start=start,
                    end=end,
                    phase="expert_compute",
                    layer_idx=layer_idx,
                )
            )

    def finish_step(self) -> DeepEPStats | None:
        """Read event timings and return per-layer stats.

        Returns None if disabled, no events exist, or GPU has not yet
        completed the events (CUDA_ERROR_NOT_READY).
        """
        if not self._enabled:
            return None

        if self._pending_events:
            events_to_read = self._pending_events
            self._graph_events = list(self._pending_events)
            self._pending_events = []
        elif self._graph_events:
            events_to_read = self._graph_events
        else:
            return None

        return self._read_events(events_to_read)

    def _read_events(self, events: list[_EventRecord]) -> DeepEPStats | None:
        dispatch_times: dict[int, float] = {}
        combine_times: dict[int, float] = {}
        expert_times: dict[int, float] = {}

        for rec in events:
            ms = rec.start.elapsed_time(rec.end)
            if ms is None:
                return None
            elapsed_s = ms / 1000.0
            if rec.phase == "dispatch":
                dispatch_times[rec.layer_idx] = (
                    dispatch_times.get(rec.layer_idx, 0.0) + elapsed_s
                )
            elif rec.phase == "combine":
                combine_times[rec.layer_idx] = (
                    combine_times.get(rec.layer_idx, 0.0) + elapsed_s
                )
            elif rec.phase == "expert_compute":
                expert_times[rec.layer_idx] = (
                    expert_times.get(rec.layer_idx, 0.0) + elapsed_s
                )

        return DeepEPStats(
            dispatch_times_s=dispatch_times,
            combine_times_s=combine_times,
            expert_compute_times_s=expert_times,
            mode=self._mode,
        )


_collector: DeepEPTimingCollector | None = None


def get_deepep_timing_collector() -> DeepEPTimingCollector:
    """Get or create the module-level timing collector singleton."""
    global _collector
    if _collector is None:
        _collector = DeepEPTimingCollector()
    return _collector
