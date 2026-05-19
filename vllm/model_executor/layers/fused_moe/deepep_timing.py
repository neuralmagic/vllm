# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Per-step per-layer timing collector for DeepEP dispatch/combine and MoE
expert compute.

Uses torch.cuda.Event for GPU timing. Only produces valid measurements in
eager mode — CUDA graph replays do not re-execute the Python instrumentation
so metrics will read as zero when graphs are active.

Gated by VLLM_MOE_TIMING_ENABLED.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.v1.metrics.stats import DeepEPStats

logger = init_logger(__name__)


@dataclass(slots=True)
class _EventRecord:
    start: torch.cuda.Event
    end: torch.cuda.Event
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
        self._enabled = False
        self._mode = ""
        self._next_layer_idx = 0

    def enable(self, mode: str = ""):
        self._enabled = True
        self._mode = mode
        logger.info(
            "MoE timing enabled (mode=%s). Metrics are only produced when "
            "CUDA graphs are disabled (--enforce-eager or "
            "-cc.cudagraph_mode=none).",
            mode,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def register_layer(self) -> int:
        """Assign sequential layer index. Called once per MoE layer."""
        idx = self._next_layer_idx
        self._next_layer_idx += 1
        return idx

    def get_event_pair(self) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        """Create a start/end event pair for timing a phase."""
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    def record_dispatch(
        self, start: torch.cuda.Event, end: torch.cuda.Event, layer_idx: int
    ):
        if self._enabled:
            self._pending_events.append(
                _EventRecord(
                    start=start, end=end, phase="dispatch", layer_idx=layer_idx
                )
            )

    def record_combine(
        self, start: torch.cuda.Event, end: torch.cuda.Event, layer_idx: int
    ):
        if self._enabled:
            self._pending_events.append(
                _EventRecord(start=start, end=end, phase="combine", layer_idx=layer_idx)
            )

    def record_expert_compute(
        self, start: torch.cuda.Event, end: torch.cuda.Event, layer_idx: int
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

        Returns None if disabled or no events were recorded this step
        (e.g. during CUDA graph replay).
        """
        if not self._enabled:
            return None

        if not self._pending_events:
            return None

        events = self._pending_events
        self._pending_events = []
        return self._read_events(events)

    def _read_events(self, events: list[_EventRecord]) -> DeepEPStats | None:
        dispatch_times: dict[int, float] = {}
        combine_times: dict[int, float] = {}
        expert_times: dict[int, float] = {}

        for rec in events:
            try:
                ms = rec.start.elapsed_time(rec.end)
            except RuntimeError:
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
