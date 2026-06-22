# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystem tier subprocess worker.

``fs_subprocess_main`` is the entry point for the daemon process spawned by
``FileSystemTierManager``.  It runs entirely inside the child process:

  - Re-opens the primary KV cache MAP_SHARED mmap so ``load_block`` /
    ``store_block`` write directly into the same physical pages as the
    scheduler without any extra data copying.
  - Owns a ``DualQueueThreadPool`` whose threads are isolated from the
    EngineCore GIL, eliminating contention on the scheduler/main thread.
  - Serves commands from ``cmd_queue``:
      ``("load",   job_id, [(path, offset), ...])``
      ``("store",  job_id, [(path, offset), ...])``
      ``("lookup", [(OffloadKey, ReqContext), ...])``
      ``("drain",)``
      ``None``  — shutdown sentinel

All arguments are plain scalars, ``FileMapper``, or ``multiprocessing.Queue``
— all picklable, compatible with both fork and spawn start methods.
"""

import functools
import mmap
import os
import queue
import signal
import threading
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.fs.io import load_block, store_block
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

logger = init_logger(__name__)


def fs_subprocess_main(
    cmd_queue: "queue.Queue[Any]",
    io_results_queue: "queue.Queue[Any]",
    lookup_results_queue: "queue.Queue[list[tuple[OffloadKey, bool]]]",
    tier_type: str,
    file_mapper: FileMapper,
    mmap_path: str,
    mmap_size: int,
    block_size: int,
    n_read_threads: int,
    n_write_threads: int,
) -> None:
    """Subprocess entry point: I/O thread pool + lookup loop.

    Args:
        cmd_queue: Receives commands from the scheduler process.
        io_results_queue: Sends completed job results and drain_ack to
            the scheduler.
        lookup_results_queue: Sends completed lookup results to the
            scheduler.
        tier_type: Tier identifier used for thread naming and logging.
        file_mapper: Maps OffloadKeys to on-disk file paths.
        mmap_path: Path to the MAP_SHARED mmap file
            (``/dev/shm/vllm_offload_{instance_id}.mmap``).
        mmap_size: Total byte size of the mmap region.
        block_size: Bytes per KV block in the primary view.
        n_read_threads: Number of load-priority I/O threads.
        n_write_threads: Number of store-priority I/O threads.
    """
    shutdown_requested = threading.Event()

    def _signal_handler(signum, frame):
        if not shutdown_requested.is_set():
            shutdown_requested.set()
            raise SystemExit()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    fd = os.open(mmap_path, os.O_RDWR)
    mmap_obj = mmap.mmap(
        fd,
        mmap_size,
        flags=mmap.MAP_SHARED,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    kv_view = memoryview(mmap_obj)
    pool = DualQueueThreadPool(
        n_read_threads,
        n_write_threads,
        thread_name_prefix=f"vllm_kv_fs_{tier_type}",
    )

    try:
        while not shutdown_requested.is_set():
            # Forward any completed I/O jobs before blocking on next command.
            for job_id, success in pool.get_finished():
                io_results_queue.put((job_id, success))

            try:
                cmd = cmd_queue.get(timeout=0.001)
            except queue.Empty:
                continue

            if cmd is None:  # shutdown sentinel
                break

            kind = cmd[0]
            if kind == "load":
                _, job_id, task_list = cmd
                tasks = [
                    functools.partial(load_block, path, kv_view, offset, block_size)
                    for path, offset in task_list
                ]
                pool.enqueue_load(job_id, len(tasks), tasks)

            elif kind == "store":
                _, job_id, task_list = cmd
                tasks = [
                    functools.partial(store_block, path, kv_view, offset, block_size)
                    for path, offset in task_list
                ]
                pool.enqueue_store(job_id, len(tasks), tasks)

            elif kind == "lookup":
                _, batch = cmd
                results: list[tuple[OffloadKey, bool]] = [
                    (key, os.path.exists(file_mapper.get_file_name(key)))
                    for key, _ in batch
                ]
                if results:
                    lookup_results_queue.put(results)

            elif kind == "drain":
                pool.wait_idle()
                for job_id, success in pool.get_finished():
                    io_results_queue.put((job_id, success))
                io_results_queue.put(("drain_ack",))

    except SystemExit:
        pass
    finally:
        pool.shutdown(wait=False)
        kv_view.release()
        mmap_obj.close()
        os.close(fd)
