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

# Counters — cumulative totals, use rate() for throughput.
METRIC_FS_STORE_JOBS = "vllm:kv_offload_fs_store_jobs"
METRIC_FS_LOAD_JOBS = "vllm:kv_offload_fs_load_jobs"

# Gauges — per-step snapshots.
METRIC_FS_ACTIVE_STORE_JOBS = "vllm:kv_offload_fs_active_store_jobs"
METRIC_FS_ACTIVE_LOAD_JOBS = "vllm:kv_offload_fs_active_load_jobs"
METRIC_FS_STORE_JOB_MAX_TIME_MS = "vllm:kv_offload_fs_store_job_max_time_ms"
METRIC_FS_LOAD_JOB_MAX_TIME_MS = "vllm:kv_offload_fs_load_job_max_time_ms"
METRIC_FS_LOOKUP_TOTAL = "vllm:kv_offload_fs_lookup_total"
METRIC_FS_LOOKUP_RESOLVED = "vllm:kv_offload_fs_lookup_resolved"
METRIC_FS_LOOKUP_MAX_TIME_MS = "vllm:kv_offload_fs_lookup_max_time_ms"


def get_fs_metric_definitions() -> dict[str, OffloadingMetricMetadata]:
    return {
        # Counters
        METRIC_FS_STORE_JOBS: OffloadingCounterMetadata(
            documentation="Number of FS store jobs completed.",
        ),
        METRIC_FS_LOAD_JOBS: OffloadingCounterMetadata(
            documentation="Number of FS load jobs completed.",
        ),
        # Gauges
        METRIC_FS_ACTIVE_STORE_JOBS: OffloadingGaugeMetadata(
            documentation="Number of FS store jobs currently queued or executing.",
        ),
        METRIC_FS_ACTIVE_LOAD_JOBS: OffloadingGaugeMetadata(
            documentation="Number of FS load jobs currently queued or executing.",
        ),
        METRIC_FS_STORE_JOB_MAX_TIME_MS: OffloadingGaugeMetadata(
            documentation="Max wall-clock time of FS store jobs completed this "
            "step, in milliseconds.",
        ),
        METRIC_FS_LOAD_JOB_MAX_TIME_MS: OffloadingGaugeMetadata(
            documentation="Max wall-clock time of FS load jobs completed this "
            "step, in milliseconds.",
        ),
        METRIC_FS_LOOKUP_TOTAL: OffloadingGaugeMetadata(
            documentation="Number of FS lookup keys currently tracked.",
        ),
        METRIC_FS_LOOKUP_RESOLVED: OffloadingGaugeMetadata(
            documentation="Number of tracked FS lookup keys that have a result.",
        ),
        METRIC_FS_LOOKUP_MAX_TIME_MS: OffloadingGaugeMetadata(
            documentation="Max end-to-end FS lookup latency this step, in milliseconds "
            "(from first lookup() call to result available or request cleanup).",
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
    store_max_ms: float = 0.0
    load_max_ms: float = 0.0

    def on_submit_store(self) -> None:
        self.active_store_jobs += 1

    def on_submit_load(self) -> None:
        self.active_load_jobs += 1

    def on_finished(self, job: JobStats) -> None:
        if job.is_store:
            self.active_store_jobs -= 1
            self.store_jobs += 1
            self.store_max_ms = max(self.store_max_ms, job.elapsed_ms)
        else:
            self.active_load_jobs -= 1
            self.load_jobs += 1
            self.load_max_ms = max(self.load_max_ms, job.elapsed_ms)

    def reset_step(self) -> None:
        self.store_jobs = 0
        self.load_jobs = 0
        self.store_max_ms = 0.0
        self.load_max_ms = 0.0

    def is_step_empty(self) -> bool:
        return not self.store_jobs and not self.load_jobs


def collect_fs_stats(
    fs: FSStats,
    ls: LookupStats,
) -> OffloadingConnectorStats | None:
    """Build an OffloadingConnectorStats from the current FS and lookup stats.

    Resets per-step state on both ``fs`` and ``ls`` before returning.
    Returns None when there is nothing to report.
    """
    lookup_max_ms = ls.max_lookup_ms
    if (
        fs.is_step_empty()
        and not lookup_max_ms
        and not ls.total
        and not fs.active_store_jobs
        and not fs.active_load_jobs
    ):
        return None

    connector_stats = OffloadingConnectorStats()

    # Active job gauges — always emitted when returning stats.
    connector_stats.set_gauge(METRIC_FS_ACTIVE_STORE_JOBS, fs.active_store_jobs)
    connector_stats.set_gauge(METRIC_FS_ACTIVE_LOAD_JOBS, fs.active_load_jobs)

    if fs.store_jobs:
        connector_stats.increase_counter(METRIC_FS_STORE_JOBS, fs.store_jobs)
        connector_stats.set_gauge(METRIC_FS_STORE_JOB_MAX_TIME_MS, fs.store_max_ms)
    if fs.load_jobs:
        connector_stats.increase_counter(METRIC_FS_LOAD_JOBS, fs.load_jobs)
        connector_stats.set_gauge(METRIC_FS_LOAD_JOB_MAX_TIME_MS, fs.load_max_ms)

    connector_stats.set_gauge(METRIC_FS_LOOKUP_TOTAL, ls.total)
    connector_stats.set_gauge(METRIC_FS_LOOKUP_RESOLVED, ls.resolved)
    if lookup_max_ms:
        connector_stats.set_gauge(METRIC_FS_LOOKUP_MAX_TIME_MS, lookup_max_ms)

    ls.reset()
    fs.reset_step()
    return connector_stats
