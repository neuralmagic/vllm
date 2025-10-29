# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import prometheus_client

from vllm.config import EPLBConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def make_per_engine(
    counter: prometheus_client.Counter, per_engine_labelvalues: dict[int, list[str]]
):
    """Create a counter for each label value."""
    return {
        idx: counter.labels(*labelvalues)
        for idx, labelvalues in per_engine_labelvalues.items()
    }


@dataclass
class EPLBStats:
    counter_a: int = 0


class EPLBProm:
    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        eplb_config: EPLBConfig | None,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[str]],
    ):
        # self.eplb_enabled = eplb_config is not None
        self.eplb_enabled = True
        if not self.eplb_enabled:
            return

        counter_a = self._counter_cls(
            name="vllm:eplb_counter_a",
            documentation="The first EPLB counter",
            labelnames=labelnames,
        )
        self.counter_a = make_per_engine(counter_a, per_engine_labelvalues)

    def observe(self, eplb_stats: EPLBStats, engine_idx: int = 0):
        if not self.eplb_enabled:
            return
        self.counter_a[engine_idx].inc(eplb_stats.counter_a)
