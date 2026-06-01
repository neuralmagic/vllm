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
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.io import load_block, store_block
from vllm.v1.kv_offload.tiering.fs.space_manager import (
    PooledSpaceManager,
    SpaceManager,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished() polls job completion and returns completed JobResults.
    """

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        space_manager_args: dict[str, Any] | None = None,
    ):
        """
        Args:
            offloading_spec: contains the vllm_config, kv_cache_config
                and block_size_factor.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            space_manager_args: Optional kwargs forwarded to SpaceManager.
                Supported keys: max_tracked_blocks, enospc_low_watermark,
                ttl_evictor_args.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)

        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            gpu_blocks_per_file=offloading_spec.block_size_factor,
        )

        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        if space_manager_args is not None:
            self._space_manager: SpaceManager = PooledSpaceManager(
                root_dir=root_dir,
                file_mapper=self.file_mapper,
                **space_manager_args,
            )
        else:
            self._space_manager = SpaceManager(
                root_dir=root_dir,
                file_mapper=self.file_mapper,
            )

        self._pending_jobs: dict[int, JobMetadata] = {}

    def lookup(
        self, key: OffloadKey, req_context: ReqContext | None = None
    ) -> bool | None:
        return self._space_manager.lookup(key)

    def submit_store(self, job_metadata: JobMetadata) -> None:
        keys = list(job_metadata.keys)
        self._space_manager.prepare_store(keys)

        tasks = (
            functools.partial(
                store_block,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_store(job_metadata.job_id, len(keys), tasks)
        self._pending_jobs[job_metadata.job_id] = job_metadata

    def submit_load(self, job_metadata: JobMetadata) -> None:
        keys = list(job_metadata.keys)
        self._space_manager.prepare_load(keys)

        tasks = (
            functools.partial(
                load_block,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_load(job_metadata.job_id, len(keys), tasks)
        self._pending_jobs[job_metadata.job_id] = job_metadata

    def get_finished(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        results: list[JobResult] = []

        for job_id, success, enospc in self._pool.get_finished():
            job = self._pending_jobs.pop(job_id)
            keys = list(job.keys)
            if job.is_promotion:
                self._space_manager.complete_load(keys, success)
            else:
                self._space_manager.complete_store(keys, success, enospc)
            results.append(JobResult(job_id=job_id, success=success))

        return results

    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the thread pool and stops the space manager (including any
        TTL evictor process).
        """
        self._pool.shutdown(wait=True)
        self._space_manager.shutdown()
