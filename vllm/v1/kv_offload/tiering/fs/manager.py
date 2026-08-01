# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import functools
import json
import os
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, ClassVar

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.io import (
    batch_load_block,
    batch_store_block,
    probe_o_direct,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool, Task

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class FsThreadPoolMetrics:
    """Metric names for FileSystemTierManager's thread pool."""

    JOB_DURATION_READ = "vllm:kv_offload_fs_job_duration_read_seconds"
    JOB_DURATION_WRITE = "vllm:kv_offload_fs_job_duration_write_seconds"
    JOB_QUEUEING_DELAY = "vllm:kv_offload_fs_job_queueing_delay_seconds"
    JOB_EXECUTION_TIME_READ = "vllm:kv_offload_fs_job_execution_time_read_seconds"
    JOB_EXECUTION_TIME_WRITE = "vllm:kv_offload_fs_job_execution_time_write_seconds"
    JOBS_IN_FLIGHT_READ = "vllm:kv_offload_fs_jobs_in_flight_read"
    JOBS_IN_FLIGHT_WRITE = "vllm:kv_offload_fs_jobs_in_flight_write"
    ACTIVE_READ_THREADS = "vllm:kv_offload_fs_active_read_threads"
    ACTIVE_WRITE_THREADS = "vllm:kv_offload_fs_active_write_threads"
    ACTIVE_READ_JOBS = "vllm:kv_offload_fs_active_read_jobs"
    ACTIVE_WRITE_JOBS = "vllm:kv_offload_fs_active_write_jobs"
    READ_BANDWIDTH_BYTES_PER_SEC = "vllm:kv_offload_fs_read_bandwidth_bytes_per_sec"
    WRITE_BANDWIDTH_BYTES_PER_SEC = "vllm:kv_offload_fs_write_bandwidth_bytes_per_sec"


class FsAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for FileSystemTierManager."""

    def __init__(
        self,
        tier: "FileSystemTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        paths = [self._tier.file_mapper.get_file_name(k) for k in keys]
        if _HAS_BATCH_LOOKUP_C:
            # C extension: GIL released for the entire faccessat() batch.
            return batch_lookup_C(paths)
        return (os.path.exists(p) for p in paths)


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        In order to enable KV cache sharing between multiple vLLM instances
        using the same ``root_dir`` (e.g., via a shared PVC) the environment
        variable ``PYTHONHASHSEED`` must be set to the same fixed value
        (e.g., "0") on all instances. Without this, each process initializes
        ``NONE_HASH`` (the chain-hash seed for block content hashes) with
        random bytes, producing different block filenames for identical token
        content.
    """

    medium: ClassVar[Medium] = Medium.STORAGE

    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        buckets = (
            0.0001,
            0.0005,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.5,
            1,
            5,
            10,
            12,
            14,
            16,
            18,
            20,
        )
        return {
            FsThreadPoolMetrics.JOB_DURATION_READ: OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of FS thread-pool load job duration: time "
                    "from a load job being enqueued to the thread pool "
                    "until its last task completes, in seconds."
                ),
                buckets=buckets,
            ),
            FsThreadPoolMetrics.JOB_DURATION_WRITE: OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of FS thread-pool store job duration: time "
                    "from a store job being enqueued to the thread pool "
                    "until its last task completes, in seconds."
                ),
                buckets=buckets,
            ),
            FsThreadPoolMetrics.JOB_QUEUEING_DELAY: OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of FS thread-pool queueing delay (load and "
                    "store jobs combined): time from a job being enqueued "
                    "until a worker thread picks up its first batch, in "
                    "seconds."
                ),
                buckets=buckets,
            ),
            FsThreadPoolMetrics.JOB_EXECUTION_TIME_READ: OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of FS thread-pool load job execution time: "
                    "time from a worker thread picking up the job's first "
                    "batch until its last task completes (excludes "
                    "queueing delay), in seconds."
                ),
                buckets=buckets,
            ),
            FsThreadPoolMetrics.JOB_EXECUTION_TIME_WRITE: OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of FS thread-pool store job execution time: "
                    "time from a worker thread picking up the job's first "
                    "batch until its last task completes (excludes "
                    "queueing delay), in seconds."
                ),
                buckets=buckets,
            ),
            FsThreadPoolMetrics.JOBS_IN_FLIGHT_READ: OffloadingGaugeMetadata(
                documentation=(
                    "Number of FS thread-pool load jobs submitted but not "
                    "yet fully completed (queued plus currently executing)."
                ),
            ),
            FsThreadPoolMetrics.JOBS_IN_FLIGHT_WRITE: OffloadingGaugeMetadata(
                documentation=(
                    "Number of FS thread-pool store jobs submitted but not "
                    "yet fully completed (queued plus currently executing)."
                ),
            ),
            FsThreadPoolMetrics.ACTIVE_READ_THREADS: OffloadingGaugeMetadata(
                documentation=(
                    "Number of FS thread-pool worker threads currently "
                    "executing a load (read) batch."
                ),
            ),
            FsThreadPoolMetrics.ACTIVE_WRITE_THREADS: OffloadingGaugeMetadata(
                documentation=(
                    "Number of FS thread-pool worker threads currently "
                    "executing a store (write) batch."
                ),
            ),
            FsThreadPoolMetrics.ACTIVE_READ_JOBS: OffloadingGaugeMetadata(
                documentation=(
                    "Number of distinct FS thread-pool load jobs currently "
                    "executing. Unlike active_read_threads, a job whose "
                    "batches are running on multiple threads is only "
                    "counted once."
                ),
            ),
            FsThreadPoolMetrics.ACTIVE_WRITE_JOBS: OffloadingGaugeMetadata(
                documentation=(
                    "Number of distinct FS thread-pool store jobs currently "
                    "executing. Unlike active_write_threads, a job whose "
                    "batches are running on multiple threads is only "
                    "counted once."
                ),
            ),
            FsThreadPoolMetrics.READ_BANDWIDTH_BYTES_PER_SEC: (
                OffloadingGaugeMetadata(
                    documentation=(
                        "Average FS load bandwidth (bytes/sec) of load jobs "
                        "that finished since the last collection, computed "
                        "from each job's execution time (excludes queueing "
                        "delay)."
                    ),
                )
            ),
            FsThreadPoolMetrics.WRITE_BANDWIDTH_BYTES_PER_SEC: (
                OffloadingGaugeMetadata(
                    documentation=(
                        "Average FS store bandwidth (bytes/sec) of store "
                        "jobs that finished since the last collection, "
                        "computed from each job's execution time (excludes "
                        "queueing delay)."
                    ),
                )
            ),
        }

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_kv_events: bool = False,
        locality: str | None = None,
    ):
        """
        Args:
            offloading_spec: Contains normalized offloading configuration and
                blocks_per_chunk.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            enable_kv_events: Emit BlockStored KV events for blocks
                successfully stored to this tier. Effective only when KV
                cache events are enabled globally (kv_events_config).
            locality: Whether this tier's storage is LOCAL or REMOTE relative
                to the publishing vLLM instance.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        self.locality = Locality(locality) if locality is not None else None

        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}

        # Per-job thread-pool timings, buffered between get_stats() calls.
        self._job_durations_read: list[float] = []
        self._job_durations_write: list[float] = []
        self._job_queueing_delays: list[float] = []
        self._job_execution_times_read: list[float] = []
        self._job_execution_times_write: list[float] = []
        self._job_bandwidths_read: list[float] = []
        self._job_bandwidths_write: list[float] = []

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        # Prefer O_DIRECT to bypass the page cache, but fall back to buffered
        # I/O on filesystems that reject it (e.g. overlayfs, some NFS mounts)
        # rather than failing every block.
        self._use_o_direct = probe_o_direct(os.path.dirname(config_path))
        if not self._use_o_direct:
            logger.warning(
                "O_DIRECT is not supported at '%s'; falling back to buffered "
                "I/O for the '%s' KV offload tier.",
                root_dir,
                tier_type,
            )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    def _tasks_from_jobmetadata(self, job_metadata: JobMetadata) -> Iterable[Task]:
        for key, bid in zip(job_metadata.keys, job_metadata.block_ids):
            yield Task(
                path=self.file_mapper.get_file_name(key),
                offset=int(bid) * self._block_size,
            )

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = list(job_metadata.keys)

        def make_batch_fn(batch: list[Task]) -> Callable[[], None]:
            return functools.partial(
                batch_store_block,
                paths=[t.path for t in batch],
                offsets=[t.offset for t in batch],
                view=self._primary_kv_view,
                block_size=self._block_size,
                use_o_direct=self._use_o_direct,
            )

        self._pool.enqueue_store(
            job_metadata.job_id,
            len(job_metadata.keys),
            self._tasks_from_jobmetadata(job_metadata),
            make_batch_fn=make_batch_fn,
        )

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        def make_batch_fn(batch: list[Task]) -> Callable[[], None]:
            return functools.partial(
                batch_load_block,
                paths=[t.path for t in batch],
                offsets=[t.offset for t in batch],
                view=self._primary_kv_view,
                block_size=self._block_size,
                use_o_direct=self._use_o_direct,
            )

        self._pool.enqueue_load(
            job_metadata.job_id,
            len(job_metadata.keys),
            self._tasks_from_jobmetadata(job_metadata),
            make_batch_fn=make_batch_fn,
        )

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        results = []
        for job in self._pool.get_finished():
            if job.is_load:
                self._job_durations_read.append(job.job_duration)
                self._job_execution_times_read.append(job.execution_time)
            else:
                self._job_durations_write.append(job.job_duration)
                self._job_execution_times_write.append(job.execution_time)
            self._job_queueing_delays.append(job.queueing_delay)

            if job.n_tasks > 0 and job.execution_time > 0:
                bandwidth = job.n_tasks * self._block_size / job.execution_time
                if job.is_load:
                    self._job_bandwidths_read.append(bandwidth)
                else:
                    self._job_bandwidths_write.append(bandwidth)

            if self.events is not None:
                keys = self._store_job_keys.pop(job.job_id, None)
                if job.success and keys:
                    self.events.append(
                        OffloadingEvent(
                            keys=keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
            results.append(JobResult(job_id=job.job_id, success=job.success))
        return results

    @override
    def get_stats(self) -> "OffloadingConnectorStats | None":
        stats = OffloadingConnectorStats()
        stats.set_gauge(
            FsThreadPoolMetrics.JOBS_IN_FLIGHT_READ,
            self._pool.num_inflight_read_jobs,
        )
        stats.set_gauge(
            FsThreadPoolMetrics.JOBS_IN_FLIGHT_WRITE,
            self._pool.num_inflight_write_jobs,
        )
        stats.set_gauge(
            FsThreadPoolMetrics.ACTIVE_READ_THREADS,
            self._pool.num_active_read_threads,
        )
        stats.set_gauge(
            FsThreadPoolMetrics.ACTIVE_WRITE_THREADS,
            self._pool.num_active_write_threads,
        )
        stats.set_gauge(
            FsThreadPoolMetrics.ACTIVE_READ_JOBS,
            self._pool.num_active_read_jobs,
        )
        stats.set_gauge(
            FsThreadPoolMetrics.ACTIVE_WRITE_JOBS,
            self._pool.num_active_write_jobs,
        )

        for duration in self._job_durations_read:
            stats.observe_histogram(FsThreadPoolMetrics.JOB_DURATION_READ, duration)
        for duration in self._job_durations_write:
            stats.observe_histogram(FsThreadPoolMetrics.JOB_DURATION_WRITE, duration)
        for delay in self._job_queueing_delays:
            stats.observe_histogram(FsThreadPoolMetrics.JOB_QUEUEING_DELAY, delay)
        for execution_time in self._job_execution_times_read:
            stats.observe_histogram(
                FsThreadPoolMetrics.JOB_EXECUTION_TIME_READ, execution_time
            )
        for execution_time in self._job_execution_times_write:
            stats.observe_histogram(
                FsThreadPoolMetrics.JOB_EXECUTION_TIME_WRITE, execution_time
            )

        if self._job_bandwidths_read:
            stats.set_gauge(
                FsThreadPoolMetrics.READ_BANDWIDTH_BYTES_PER_SEC,
                sum(self._job_bandwidths_read) / len(self._job_bandwidths_read),
            )
        if self._job_bandwidths_write:
            stats.set_gauge(
                FsThreadPoolMetrics.WRITE_BANDWIDTH_BYTES_PER_SEC,
                sum(self._job_bandwidths_write) / len(self._job_bandwidths_write),
            )

        self._job_durations_read.clear()
        self._job_durations_write.clear()
        self._job_queueing_delays.clear()
        self._job_execution_times_read.clear()
        self._job_execution_times_write.clear()
        self._job_bandwidths_read.clear()
        self._job_bandwidths_write.clear()
        return stats

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    @override
    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
