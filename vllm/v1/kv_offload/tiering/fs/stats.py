# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FS tier Prometheus metric definitions, stats accumulation, and stat collection.
"""

from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingMetricMetadata,
)
from vllm.v1.kv_offload.tiering.async_lookup import LookupStats
from vllm.v1.kv_offload.tiering.fs.thread_pool import JobStats


def _fs_metric_names(tier_name: str) -> dict[str, str]:
    """Return Prometheus metric names for this FS tier instance.

    ``tier_name`` is always non-empty (``SecondaryTierFactory`` falls back to
    the registered tier type, e.g. ``"fs"``).  A single unnamed FS tier
    produces ``vllm:kv_offload_fs_store_jobs``; a tier named ``"nvme"``
    produces ``vllm:kv_offload_nvme_store_jobs``.
    """
    p = f"vllm:kv_offload_{tier_name}"
    return {
        "store_jobs": f"{p}_store_jobs",
        "load_jobs": f"{p}_load_jobs",
        "active_store_jobs": f"{p}_active_store_jobs",
        "active_load_jobs": f"{p}_active_load_jobs",
        "store_job_max_latency": f"{p}_store_job_max_latency",
        "load_job_max_latency": f"{p}_load_job_max_latency",
        "lookup_total": f"{p}_lookup_total",
        "lookup_resolved": f"{p}_lookup_resolved",
        "lookup_max_latency": f"{p}_lookup_max_latency",
        "store_bytes": f"{p}_store_bytes",
        "load_bytes": f"{p}_load_bytes",
        "store_time": f"{p}_store_time",
        "load_time": f"{p}_load_time",
    }


def get_fs_metric_definitions(tier_name: str) -> dict[str, OffloadingMetricMetadata]:
    m = _fs_metric_names(tier_name)
    return {
        # Counters
        m["store_jobs"]: OffloadingCounterMetadata(
            documentation="Number of FS store jobs completed.",
        ),
        m["load_jobs"]: OffloadingCounterMetadata(
            documentation="Number of FS load jobs completed.",
        ),
        # Gauges
        m["active_store_jobs"]: OffloadingGaugeMetadata(
            documentation="Number of FS store jobs currently queued or executing.",
        ),
        m["active_load_jobs"]: OffloadingGaugeMetadata(
            documentation="Number of FS load jobs currently queued or executing.",
        ),
        m["store_job_max_latency"]: OffloadingGaugeMetadata(
            documentation="Max wall-clock latency of FS store jobs completed this "
            "step, in seconds.",
        ),
        m["load_job_max_latency"]: OffloadingGaugeMetadata(
            documentation="Max wall-clock latency of FS load jobs completed this "
            "step, in seconds.",
        ),
        m["lookup_total"]: OffloadingGaugeMetadata(
            documentation="Number of FS lookup keys currently tracked.",
        ),
        m["lookup_resolved"]: OffloadingGaugeMetadata(
            documentation="Number of tracked FS lookup keys that have a result.",
        ),
        m["lookup_max_latency"]: OffloadingGaugeMetadata(
            documentation="Max end-to-end FS lookup latency this step, in seconds "
            "(from first lookup() call to result available or request cleanup).",
        ),
        m["store_bytes"]: OffloadingCounterMetadata(
            documentation="Cumulative bytes written to disk (excludes skipped blocks "
            "where the file already existed).",
        ),
        m["load_bytes"]: OffloadingCounterMetadata(
            documentation="Cumulative bytes read from disk.",
        ),
        m["store_time"]: OffloadingGaugeMetadata(
            documentation="Total I/O time (seconds) spent on store tasks this step, "
            "summed across all worker threads (excludes queue wait time).",
        ),
        m["load_time"]: OffloadingGaugeMetadata(
            documentation="Total I/O time (seconds) spent on load tasks this step, "
            "summed across all worker threads (excludes queue wait time).",
        ),
    }


@dataclass
class FSStats:
    """Tracks active job counts and per-step completed job stats."""

    # Active job counts — persistent across steps.
    active_store_jobs: int = 0
    active_load_jobs: int = 0

    # Per-step completed job stats — reset each step via reset_step().
    store_jobs: int = 0
    load_jobs: int = 0
    store_job_max_latency: float = 0.0
    load_job_max_latency: float = 0.0
    store_bytes: int = 0
    load_bytes: int = 0
    store_time: float = 0.0
    load_time: float = 0.0

    def on_submit_store(self) -> None:
        self.active_store_jobs += 1

    def on_submit_load(self) -> None:
        self.active_load_jobs += 1

    def on_finished(self, job: JobStats) -> None:
        if job.is_store:
            self.active_store_jobs -= 1
            self.store_jobs += 1
            self.store_job_max_latency = max(
                self.store_job_max_latency, job.elapsed_ms / 1e3
            )
            self.store_bytes += job.bytes_transferred
            self.store_time += job.io_elapsed_s
        else:
            self.active_load_jobs -= 1
            self.load_jobs += 1
            self.load_job_max_latency = max(
                self.load_job_max_latency, job.elapsed_ms / 1e3
            )
            self.load_bytes += job.bytes_transferred
            self.load_time += job.io_elapsed_s

    def reset_step(self) -> None:
        self.store_jobs = 0
        self.load_jobs = 0
        self.store_job_max_latency = 0.0
        self.load_job_max_latency = 0.0
        self.store_bytes = 0
        self.load_bytes = 0
        self.store_time = 0.0
        self.load_time = 0.0

    def is_step_empty(self) -> bool:
        return not self.store_jobs and not self.load_jobs


def collect_fs_stats(
    fs: FSStats,
    ls: LookupStats,
    tier_name: str,
) -> OffloadingConnectorStats | None:
    """Build an OffloadingConnectorStats from the current FS and lookup stats.

    Resets per-step state on both ``fs`` and ``ls`` before returning.
    Returns None when there is nothing to report.

    Args:
        fs: Per-step FS job stats accumulator.
        ls: Per-step async lookup stats accumulator.
        tier_name: Resolved tier name from :func:`_resolve_tier_name`
            (e.g. ``"fs"`` or ``"nvme"``).  Determines the Prometheus metric
            prefix ``vllm:kv_offload_{tier_name}_*``.  Must match the name
            used at registration time in ``build_metric_definitions``.
    """
    lookup_max_ms = ls.max_lookup_latency_ms
    if (
        fs.is_step_empty()
        and not lookup_max_ms
        and not ls.total
        and not fs.active_store_jobs
        and not fs.active_load_jobs
    ):
        return None

    m = _fs_metric_names(tier_name)
    connector_stats = OffloadingConnectorStats()

    # Active job gauges — always emitted when returning stats.
    connector_stats.set_gauge(m["active_store_jobs"], fs.active_store_jobs)
    connector_stats.set_gauge(m["active_load_jobs"], fs.active_load_jobs)

    if fs.store_jobs:
        connector_stats.increase_counter(m["store_jobs"], fs.store_jobs)
        connector_stats.set_gauge(m["store_job_max_latency"], fs.store_job_max_latency)
        connector_stats.increase_counter(m["store_bytes"], fs.store_bytes)
        connector_stats.set_gauge(m["store_time"], fs.store_time)
    if fs.load_jobs:
        connector_stats.increase_counter(m["load_jobs"], fs.load_jobs)
        connector_stats.set_gauge(m["load_job_max_latency"], fs.load_job_max_latency)
        connector_stats.increase_counter(m["load_bytes"], fs.load_bytes)
        connector_stats.set_gauge(m["load_time"], fs.load_time)

    connector_stats.set_gauge(m["lookup_total"], ls.total)
    connector_stats.set_gauge(m["lookup_resolved"], ls.resolved)
    if lookup_max_ms:
        connector_stats.set_gauge(m["lookup_max_latency"], lookup_max_ms / 1e3)

    ls.reset()
    fs.reset_step()
    return connector_stats
