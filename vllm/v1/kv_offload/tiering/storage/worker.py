# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import concurrent.futures
from collections.abc import Collection
from concurrent.futures import ALL_COMPLETED, Future, wait
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobResult,
    JobTransferStats,
    TransferType,
)
from vllm.v1.kv_offload.tiering.storage.common import TransferResult
from vllm.v1.kv_offload.tiering.storage.simple_transfer_engine import (
    SimpleTransferEngine,
)

logger = init_logger(__name__)


@dataclass
class StorageJobMetadata:
    """Internal metadata for tracking job details."""

    job_id: JobId
    keys: Collection[OffloadKey]
    block_ids: Collection[int]
    storage_paths: Collection[Path]
    is_store: bool  # True for store jobs, False for load jobs

    def __post_init__(self):
        assert len(self.keys) == len(self.block_ids) == len(self.storage_paths), (
            f"#keys({len(self.keys)}) != #block_ids({len(self.block_ids)}) "
            f" != #storage_paths({len(self.storage_paths)}). Need a 1-to-1 mapping."
        )

    def transfer_type(
        self,
    ) -> TransferType:
        return ("CPU", "STORAGE") if self.is_store else ("STORAGE", "CPU")


class StorageHandler:
    # TODO (varun) : plumb num_threads
    def __init__(
        self,
        primary_kv_view: memoryview,
        total_bytes_per_block: int,
        num_threads: int = 16,
    ):
        self.primary_kv_view = primary_kv_view
        self.primary_kv = np.frombuffer(self.primary_kv_view, dtype=np.uint8).reshape(
            (-1, total_bytes_per_block)
        )
        self.total_bytes_per_block = total_bytes_per_block
        self.num_threads = num_threads

        self.transfer_engine = SimpleTransferEngine(self.primary_kv)
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_threads
        )
        self.job_meta: dict[JobId, StorageJobMetadata] = {}
        self.load_jobs: dict[JobId, Future] = {}
        self.store_jobs: dict[JobId, Future] = {}

    def transfer_async(self, storage_job_metadata: StorageJobMetadata):
        """
        Initiates an asynchronous transfer of KV data.

        Args:
            storage_job_metadata: Job metadata with information about all
             the transfer ops to be scheduled.

        Returns:
            True if transfer was submitted successfully.
        """
        job_id: JobId = storage_job_metadata.job_id
        self.job_meta[job_id] = storage_job_metadata

        is_load = not storage_job_metadata.is_store

        # fetch load/store job_fn
        job_fn = self.transfer_engine.load if is_load else self.transfer_engine.store
        f = self.executor.submit(
            job_fn,
            job_id,
            storage_job_metadata.storage_paths,
            storage_job_metadata.block_ids,
        )

        # fetch load/store job record
        job_records = self.load_jobs if is_load else self.store_jobs
        job_records[job_id] = f

    def _cleanup_job(self, job_id: int) -> None:
        meta = self.job_meta.pop(job_id, None)
        if meta is None:
            return
        if meta.is_store:
            self.store_jobs.pop(job_id)
        else:
            self.load_jobs.pop(job_id)

    def _safe_future_result(self, future: Future, job_id: JobId) -> TransferResult:
        try:
            return future.result()
        except Exception as e:
            logger.debug("Job %s failed with exception %s", self.job_meta[job_id], e)
        # Default, failure transfer result
        return TransferResult()

    def get_finished(self) -> list[JobResult]:
        """
        Get transfers finished since last call.

        Returns:
            A list of (job_id, success) of transfers.
        """
        job_results: list[JobResult] = []

        for jid, future in chain(self.load_jobs.items(), self.store_jobs.items()):
            if not future.done():
                # not all are done
                continue
            result = self._safe_future_result(future, jid)
            job_meta = self.job_meta[jid]
            job_results.append(
                JobResult(
                    job_id=jid,
                    success=result.success,
                    transfer_stats=JobTransferStats(
                        transfer_size=result.transfer_size,
                        transfer_time=result.transfer_time,
                        transfer_type=job_meta.transfer_type(),
                    ),
                )
            )

        for jr in job_results:
            self._cleanup_job(jr.job_id)

        # There could be parts of job that could have succeeded
        # Just deem the entire job unsuccessful
        # TODO (varun): A job is a collection of key loads and stores.
        # We could be more robust and account for each key load/store
        # individually. For now, we consider the entire job as failed
        # if even a part of the job succeeded.

        return job_results

    def wait(self, job_ids: set[int]) -> None:
        """
        Wait for jobs to finish (blocking). Note that this just waits for
        the jobs to finish, but does no cleanup.
        Args:
            job_ids: The set of job IDs to wait for.
        """
        for jid in job_ids:
            meta = self.job_meta.get(jid, None)
            if meta is None:
                continue

            future = self.store_jobs[jid] if meta.is_store else self.load_jobs.get(jid)
            assert future is not None
            wait([future], return_when=ALL_COMPLETED)

    def shutdown(self) -> None:
        """Shutdown the handler and release any resources."""
        # cancel all on going jobs
        for _, future in chain(self.load_jobs.items(), self.store_jobs.items()):
            future.cancel()

        # wait for all jobs to complete
        self.wait(set([jid for jid in self.job_meta]))

        # clean up state
        for jid in list(self.job_meta):
            self._cleanup_job(jid)

        assert len(self.job_meta) == 0
        assert len(self.load_jobs) == 0
        assert len(self.store_jobs) == 0
