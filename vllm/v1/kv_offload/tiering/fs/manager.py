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

Subprocess design:
    All I/O threads and the lookup worker run inside a dedicated subprocess
    (see ``worker.py``).  The scheduler thread communicates via three
    multiprocessing.Queues:

      cmd_queue            scheduler → subprocess  (load/store/lookup/drain/shutdown)
      io_results_queue     subprocess → scheduler  (job_id, success) + drain_ack
      lookup_results_queue subprocess → scheduler  list[(OffloadKey, bool)]

Cross-process sharing:
    In order to enable KV cache sharing between multiple vLLM instances
    using the same ``root_dir`` (e.g., via a shared PVC) the environment
    variable ``PYTHONHASHSEED`` must be set to the same fixed value
    (e.g., "0") on all instances. Without this, each process initializes
    ``NONE_HASH`` (the chain-hash seed for block content hashes) with
    random bytes, producing different block filenames for identical token
    content.
"""

import json
import multiprocessing
import os
import queue
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from vllm.logger import init_logger
from vllm.utils.system_utils import get_mp_context
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager, LookupState
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.worker import fs_subprocess_main

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class FsAsyncLookupManager(AsyncLookupManager):
    """Lookup state manager for FileSystemTierManager.

    Maintains lookup state in the scheduler thread.  The actual
    ``os.path.exists`` calls are executed by the FileSystemTierManager
    subprocess — this class only routes batches to/from the subprocess
    queues.  No background thread or process is started here.
    """

    def __init__(
        self,
        tier_type: str,
        cmd_queue: "multiprocessing.Queue[Any]",
        lookup_results_queue: "multiprocessing.Queue[list[tuple[OffloadKey, bool]]]",
    ) -> None:
        # Do not call super().__init__() — that would start a background thread.
        # Manually initialise the scheduler-owned state attrs from AsyncLookupManager.
        self._tier_type = tier_type
        self._lookup_state: dict[OffloadKey, LookupState] = {}
        self._req_keys: dict[str, set[OffloadKey]] = {}
        self._lookup_batch: list[tuple[OffloadKey, ReqContext]] = []
        self._need_to_drain: bool = False
        # Route lookup batches to the shared subprocess via cmd_queue.
        self._cmd_queue = cmd_queue
        # Expose lookup_results_queue as _pending_results so the inherited
        # drain_results() works unchanged: mp.Queue.get_nowait() raises
        # queue.Empty identically to SimpleQueue.get_nowait().
        self._pending_results = lookup_results_queue  # type: ignore[assignment]

    @override
    def flush(self) -> None:
        """Post this step's accumulated keys as a lookup command to the subprocess."""
        self._need_to_drain = True
        if self._lookup_batch:
            self._cmd_queue.put(("lookup", self._lookup_batch))
            self._lookup_batch = []

    @override
    def shutdown(self) -> None:
        """No-op: subprocess lifecycle is managed by FileSystemTierManager."""

    @override
    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        # Not called — the subprocess handles lookups directly.
        # Retained to satisfy the ABC.
        raise NotImplementedError(
            "FsAsyncLookupManager.batch_lookup is never called directly; "
            "lookups are executed inside the FileSystemTierManager subprocess."
        )


class FileSystemTierManager(SecondaryTierManager):
    """Pure-Python disk-backed secondary tier.

    All I/O threads (read/write) and the lookup worker run inside a single
    subprocess (``fs_subprocess_main`` in ``worker.py``).  This isolates the
    DualQueueThreadPool threads from the EngineCore GIL, eliminating contention
    on the scheduler/main thread.
    """

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
    ):
        """
        Args:
            offloading_spec: contains the vllm_config, kv_cache_config
                and block_size_factor.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads in the subprocess.
            n_write_threads: Number of write-priority I/O threads in the subprocess.
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

        # Derive mmap coordinates from the view.
        # SharedOffloadRegion uses: /dev/shm/vllm_offload_{instance_id}.mmap
        mmap_path = (
            f"/dev/shm/vllm_offload_{offloading_spec.vllm_config.instance_id}.mmap"
        )
        mmap_size: int = primary_kv_view.nbytes

        ctx = get_mp_context()
        self._cmd_queue: multiprocessing.Queue = ctx.Queue()
        self._io_results_queue: multiprocessing.Queue = ctx.Queue()
        self._lookup_results_queue: multiprocessing.Queue = ctx.Queue()

        self._process = ctx.Process(
            target=fs_subprocess_main,
            kwargs={
                "cmd_queue": self._cmd_queue,
                "io_results_queue": self._io_results_queue,
                "lookup_results_queue": self._lookup_results_queue,
                "tier_type": tier_type,
                "file_mapper": self.file_mapper,
                "mmap_path": mmap_path,
                "mmap_size": mmap_size,
                "block_size": self._block_size,
                "n_read_threads": n_read_threads,
                "n_write_threads": n_write_threads,
            },
            name=f"vllm_kv_fs_{tier_type}",
            daemon=True,
        )
        self._process.start()

        # Buffer for job results received during drain_jobs() blocking wait.
        self._buffered_io_results: list[tuple[JobId, bool]] = []

        self._lookup_manager = FsAsyncLookupManager(
            tier_type=tier_type,
            cmd_queue=self._cmd_queue,
            lookup_results_queue=self._lookup_results_queue,
        )

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        return self._lookup_manager.lookup(key, req_context)

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        task_list = [
            (self.file_mapper.get_file_name(key), int(bid) * self._block_size)
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        ]
        self._cmd_queue.put(("store", job_metadata.job_id, task_list))

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        task_list = [
            (self.file_mapper.get_file_name(key), int(bid) * self._block_size)
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        ]
        self._cmd_queue.put(("load", job_metadata.job_id, task_list))

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Poll the subprocess for completed load/store jobs."""
        results = self._buffered_io_results[:]
        self._buffered_io_results.clear()
        while True:
            try:
                item = self._io_results_queue.get_nowait()
            except queue.Empty:
                break
            if item[0] == "drain_ack":
                continue  # ignore stray drain_ack sentinels
            results.append(item)
        return [
            JobResult(job_id=job_id, success=success) for job_id, success in results
        ]

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight subprocess transfers finish."""
        self._cmd_queue.put(("drain",))
        while True:
            item = self._io_results_queue.get()
            if item[0] == "drain_ack":
                break
            # Buffer any job results that arrive while waiting for the ack.
            self._buffered_io_results.append(item)

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)

    @override
    def on_schedule_end(self) -> None:
        self._lookup_manager.flush()

    @override
    def shutdown(self) -> None:
        """Send shutdown signal to the subprocess and wait for it to exit."""
        self._lookup_manager.shutdown()
        self._cmd_queue.put(None)
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
