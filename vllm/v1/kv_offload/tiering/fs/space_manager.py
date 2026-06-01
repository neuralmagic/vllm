# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from pathlib import Path
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.fs.thread_pool import TaskResult
from vllm.v1.kv_offload.tiering.fs.ttl_evictor import (
    TTLEvictor,
    TTLEvictorHandle,
    get_disk_usage_fraction_form_statvfs,
    safe_remove,
)

logger = init_logger(__name__)

# How often (seconds) to check disk usage when clearing the ENOSPC flag.
ENOSPC_CHECK_INTERVAL_S: float = 5.0


class SpaceManager:
    """
    Base space manager — no pool tracking.

    All methods are no-ops except lookup(), which falls back to os.path.exists.
    Used when space_manager_args is not provided to FileSystemTierManager.
    """

    def __init__(self, root_dir: str, file_mapper: FileMapper):
        self.root_dir = root_dir
        self.file_mapper = file_mapper

    def lookup(self, key: OffloadKey) -> bool:
        return os.path.exists(self.file_mapper.get_file_name(key))

    def prepare_load(self, keys: list[OffloadKey]) -> None:
        pass

    def complete_load(self, results: list[TaskResult]) -> None:
        pass

    def prepare_store(self, keys: list[OffloadKey]) -> None:
        pass

    def complete_store(self, results: list[TaskResult]) -> None:
        pass

    def shutdown(self) -> None:
        pass


class PooledSpaceManager(SpaceManager):
    """
    Space manager with an in-memory LRU block pool.

    All public methods are called exclusively from the scheduler thread, so
    no locking is required.

    Tracks up to max_tracked_blocks file-backed KV blocks and provides:

    - Fast pool-first lookup: avoids os.path.exists when the block is already
      known to the pool.
    - Load pinning: ref_cnt is incremented during active loads so blocks are
      not evicted from under a running transfer.
    - ENOSPC handling: complete_store() sets the flag when a job fails with
      ENOSPC. prepare_store() then hard-deletes LRU-evictable blocks to make
      room, and clears the flag once disk usage recovers (time-gated statvfs).

    BlockStatus semantics (reused from the CPU tier):
        ref_cnt == -1  in-flight store, not yet readable
        ref_cnt ==  0  ready on disk, evictable
        ref_cnt  >  0  pinned by one or more active loads, not evictable
    """

    def __init__(
        self,
        root_dir: str,
        file_mapper: FileMapper,
        max_tracked_blocks: int = 10_000,
        enospc_low_watermark: float = 0.8,
        ttl_evictor_args: dict[str, Any] | None = None,
    ):
        super().__init__(root_dir, file_mapper)
        self._max_tracked_blocks = max_tracked_blocks
        self._low_watermark = enospc_low_watermark

        self._policy: LRUCachePolicy = LRUCachePolicy(cache_capacity=max_tracked_blocks)
        self._enospc: bool = False
        self._last_statvfs_ts: float = 0.0

        self._evictor_handle: TTLEvictorHandle | None = None
        if ttl_evictor_args is not None:
            self._evictor_handle = TTLEvictor.spawn(
                root_dir=Path(root_dir),
                **ttl_evictor_args,
            )

    def lookup(self, key: OffloadKey) -> bool:
        """Pool-first lookup. Falls back to os.path.exists on a pool miss."""
        block = self._policy.get(key)
        if block is not None and block.is_ready:
            return True
        return os.path.exists(self.file_mapper.get_file_name(key))

    def prepare_load(self, keys: list[OffloadKey]) -> None:
        """Pin pool blocks so they cannot be evicted during an active load."""
        for key in keys:
            block = self._policy.get(key)
            if block is not None:
                block.ref_cnt += 1

    def complete_load(self, results: list[TaskResult]) -> None:
        """Unpin pool blocks once a load completes. Remove on failure."""
        for r in results:
            block = self._policy.get(r.key)
            if block is None:
                continue
            if block.ref_cnt > 0:
                block.ref_cnt -= 1
            else:
                logger.warning(
                    "complete_load: ref_cnt already 0 for key %s; "
                    "missing prepare_load?",
                    r.key,
                )
            if not r.success:
                self._policy.remove(r.key)

    def _maybe_make_space(self, num_new_keys: int, protected: set[OffloadKey]) -> bool:
        free_slots = self._max_tracked_blocks - len(self._policy.blocks)
        need = num_new_keys - free_slots

        if need == 0:
            return True

        evicted = self._policy.evict(need, protected)
        if evicted is None:
            # Pool fully pinned — accept without tracking.
            return False

        if self._enospc:
            for key, _ in evicted:
                # TODO (varun): this is a syscall on a hot-path.
                # it is better to delegate to a deleter thread or the
                # TTLEvictor.
                # hard remove
                safe_remove(Path(self.file_mapper.get_file_name(key)))
        return True

    def prepare_store(self, keys: list[OffloadKey]) -> None:
        """
        Register incoming store keys with the pool.

        Stores are never denied — if the disk is full the job proceeds and
        fails naturally via ENOSPC. When under ENOSPC pressure, LRU-evictable
        blocks are hard-deleted to make room before inserting the new keys.
        """
        self._maybe_clear_enospc()

        new_keys = [k for k in keys if self._policy.get(k) is None]
        if not new_keys:
            return

        if not self._maybe_make_space(len(new_keys), protected=set(keys)):
            return

        for key in new_keys:
            self._policy.insert(key, BlockStatus(0))

    def complete_store(self, results: list[TaskResult]) -> None:
        """
        Update the pool for each completed store task.

        - existed: file was already on disk (not owned) → remove from pool.
        - success: newly written → mark ready.
        - failure: remove from pool; set ENOSPC flag if applicable.
        """
        for r in results:
            if r.enospc:
                self._enospc = True
            block = self._policy.get(r.key)
            if block is None:
                if not r.existed and r.success:  # noqa: SIM102
                    # Block was written but not tracked (pool was full at
                    # prepare_store time). Insert it now, evicting if needed.
                    if self._maybe_make_space(1, set()):
                        status = BlockStatus(0)
                        status.ref_cnt = 0
                        self._policy.insert(r.key, status)
                continue

            if not r.existed and r.success:
                # successful store with actual write
                # Created by this task.
                if block.is_ready:
                    logger.warning("multiple stores detected")
                else:
                    block.ref_cnt += 1
                continue

            if r.existed or not r.success:
                # not owned by us or a failed store
                self._policy.remove(r.key)

    def shutdown(self) -> None:
        if self._evictor_handle is not None:
            self._evictor_handle.stop()
            self._evictor_handle = None

    # --- internals ---

    def _maybe_clear_enospc(self) -> None:
        """Time-gated statvfs check. Clears _enospc if disk usage recovered."""
        if not self._enospc:
            return
        now = time.time()
        if now - self._last_statvfs_ts < ENOSPC_CHECK_INTERVAL_S:
            return
        self._last_statvfs_ts = now
        usage = get_disk_usage_fraction_form_statvfs(Path(self.root_dir))
        if usage is not None and usage < self._low_watermark:
            logger.info(
                "PooledSpaceManager: disk usage %.1f%% < low watermark %.1f%%, "
                "clearing ENOSPC flag.",
                usage * 100,
                self._low_watermark * 100,
            )
            self._enospc = False
