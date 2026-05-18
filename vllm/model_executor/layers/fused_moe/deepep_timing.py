# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Per-step timing collector for DeepEP dispatch/combine and MoE expert compute.

CUDA events are recorded on the appropriate stream during each forward step.
Elapsed times are computed from the *previous* step's events (guaranteed
GPU-complete) to avoid introducing synchronization points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vllm.v1.metrics.stats import DeepEPStats


@dataclass
class _EventRecord:
    start: torch.cuda.Event
    end: torch.cuda.Event
    phase: str  # "dispatch", "combine", or "expert_compute"
    num_tokens: int


@dataclass
class _StepEvents:
    events: list[_EventRecord] = field(default_factory=list)
    mode: str = ""


class DeepEPTimingCollector:
    """Collects CUDA event pairs per forward step and computes elapsed times.

    Usage:
        1. During forward, call record_dispatch/record_combine/record_expert_compute
        2. After forward, call finish_step() to get the *previous* step's stats
           (current step's events are stored for the next call)
    """

    def __init__(self):
        self._current_step = _StepEvents()
        self._prev_step: _StepEvents | None = None
        self._enabled = False

    def enable(self, mode: str = ""):
        self._enabled = True
        self._current_step.mode = mode

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record_dispatch(
        self,
        start: torch.cuda.Event,
        end: torch.cuda.Event,
        num_tokens: int,
    ):
        if not self._enabled:
            return
        self._current_step.events.append(
            _EventRecord(start=start, end=end, phase="dispatch",
                         num_tokens=num_tokens)
        )

    def record_combine(
        self,
        start: torch.cuda.Event,
        end: torch.cuda.Event,
        num_tokens: int,
    ):
        if not self._enabled:
            return
        self._current_step.events.append(
            _EventRecord(start=start, end=end, phase="combine",
                         num_tokens=num_tokens)
        )

    def record_expert_compute(
        self,
        start: torch.cuda.Event,
        end: torch.cuda.Event,
        num_tokens: int,
    ):
        if not self._enabled:
            return
        self._current_step.events.append(
            _EventRecord(start=start, end=end, phase="expert_compute",
                         num_tokens=num_tokens)
        )

    def finish_step(self) -> DeepEPStats | None:
        """Compute stats from the previous step's events and rotate buffers.

        Returns None if no previous step data is available yet (first step).
        """
        if not self._enabled:
            return None

        result: DeepEPStats | None = None

        if self._prev_step is not None and self._prev_step.events:
            dispatch_time = 0.0
            combine_time = 0.0
            expert_time = 0.0
            dispatch_tokens = 0
            combine_tokens = 0

            for rec in self._prev_step.events:
                elapsed_ms = rec.start.elapsed_time(rec.end)
                elapsed_s = elapsed_ms / 1000.0
                if rec.phase == "dispatch":
                    dispatch_time += elapsed_s
                    dispatch_tokens += rec.num_tokens
                elif rec.phase == "combine":
                    combine_time += elapsed_s
                    combine_tokens += rec.num_tokens
                elif rec.phase == "expert_compute":
                    expert_time += elapsed_s

            result = DeepEPStats(
                dispatch_time_s=dispatch_time,
                combine_time_s=combine_time,
                expert_compute_time_s=expert_time,
                dispatch_tokens=dispatch_tokens,
                combine_tokens=combine_tokens,
                mode=self._prev_step.mode,
            )

        # Rotate: current becomes previous, start fresh current
        self._prev_step = self._current_step
        self._current_step = _StepEvents(mode=self._prev_step.mode)

        return result


_collector: DeepEPTimingCollector | None = None


def get_deepep_timing_collector() -> DeepEPTimingCollector:
    """Get or create the module-level timing collector singleton."""
    global _collector
    if _collector is None:
        _collector = DeepEPTimingCollector()
    return _collector
