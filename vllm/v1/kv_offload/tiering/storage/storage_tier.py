# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Storage tier offloading (Shared Storage / Filesystem).  Note that the
tier only supports filesystems that are POSIX based.
"""

from collections.abc import Collection, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

# TODO (Varun) : Move the LRU policy out of the CPU folder
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    get_offload_block_hash,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.common import prepare_store_blocks
from vllm.v1.kv_offload.cpu.offload_block_tracker import OffloadBlockTracker
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    PrimaryTierMetadata,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.storage.worker import StorageHandler, StorageJobMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class StorageLoadStoreSpec(LoadStoreSpec):
    """
    Spec for loading/storing a KV block to secondary storage / filesystem.
    """

    def __init__(self, storage_paths: list[Path]):
        self.storage_paths = storage_paths

    def __repr__(self) -> str:
        return repr(self.storage_paths)

    @staticmethod
    def medium() -> str:
        return "STORAGE"


class StorageSecondaryTier(SecondaryTierManager):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        primary_tier_meta: PrimaryTierMetadata,
        storage_root_path: str,
        max_storage_size_gb: int,
        enable_events: bool = False,
    ):
        """
        Initialize the example secondary tier.

        Args:
            vllm_config: Global vLLM configuration.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            storage_path: Path to store the offloaded KV cache.
        """
        super().__init__(vllm_config, kv_cache_config, primary_tier_meta)

        self.medium = StorageLoadStoreSpec.medium()
        self.vllm_config = vllm_config
        self.primary_tier_meta = primary_tier_meta
        self.kv_cache_config = kv_cache_config
        self.storage_root_path = Path(storage_root_path)
        self.max_storage_size_gb = max_storage_size_gb
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

        # Add a vllm-config based base path.
        self.base_path = self.storage_root_path / self.vllm_config.compute_hash()

        self.max_blocks = (
            self.max_storage_size_gb * 1024 * 1024 * 1024
        ) // primary_tier_meta.bytes_per_block
        self.block_tracker = OffloadBlockTracker(self.max_blocks)

        ## Completed jobs waiting to be retrieved by get_finished()
        self.completed_jobs: list[JobResult] = []

        self.cache_policy = LRUCachePolicy(self.max_blocks)

        self._pending_jobs: dict[JobId, StorageJobMetadata] = {}

        # TODO(varun) : StorageHandler -> StorageWorker
        self.handler = StorageHandler(
            primary_kv_view=primary_tier_meta.kv_view,
            total_bytes_per_block=primary_tier_meta.bytes_per_block,
        )

    def _hash_to_file(self, key: OffloadKey) -> Path:
        hash_hex = get_offload_block_hash(key).hex()
        group_idx = get_offload_group_idx(key)
        subfolder1, subfolder2 = hash_hex[:3], hash_hex[3:5]

        return (
            self.base_path
            / subfolder1
            / f"{subfolder2}_g{group_idx}"
            / f"{hash_hex}.bin"
        )

    def _get_load_store_spec(
        self, keys: Collection[OffloadKey]
    ) -> StorageLoadStoreSpec:
        return StorageLoadStoreSpec([self._hash_to_file(k) for k in keys])

    def _evict_keys(self, to_evict: Collection[OffloadKey]) -> None:
        for key in to_evict:
            # TODO (varun) : Protect against premature eviction
            self._hash_to_file(key).unlink(missing_ok=True)

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Check whether a block exists in this secondary tier.

        Args:
            key: Offload key to look up.
            req_context: Per-request context.

        Returns:
            True if the block is present, False if not found.
        """
        # TODO(varun): For re-use across vllm instances, we could
        # check if the hash file is available and consider that a hit.
        # But this means a syscall for every lookup.
        # Another approach is to populate the cache based during startup.
        block = self.cache_policy.get(key)
        if block is None:
            return False
        if not block.is_ready:
            return None  # write in-flight; caller should retry
        return True

    def submit_store(self, job_metadata: JobMetadata) -> None:
        """
        Submit an async job to store blocks from primary tier to this tier.

        Args:
            job_metadata: Job metadata including job_id, keys, and
                          spec for reading blocks from the primary tier.
        """

        job_id = job_metadata.job_id
        keys = job_metadata.keys  # keys to store
        primary_block_ids = job_metadata.block_ids  # primary tier block ids
        req_context = job_metadata.req_context

        assert len(keys) == len(primary_block_ids), (
            f"Length mismatch: {len(keys)} keys but {len(primary_block_ids)} block_ids"
        )

        # We are actively demoting the block.
        assert not job_metadata.is_promotion

        pso = self.prepare_store(keys, req_context)
        if pso is None:
            # Store preparation failed
            return
        assert isinstance(pso.store_spec, StorageLoadStoreSpec)
        # Not all keys are scheduled a store. For example, we ignore keys
        # that are stored already
        key_to_primary_block_ids = {k: b for k, b in zip(keys, primary_block_ids)}
        primary_blocks_to_store = [
            key_to_primary_block_ids[sk] for sk in pso.keys_to_store
        ]

        # Create internal job metadata
        internal_job_metadata = StorageJobMetadata(
            job_id=job_id,
            keys=pso.keys_to_store,
            block_ids=primary_blocks_to_store,
            storage_paths=pso.store_spec.storage_paths,
            is_store=True,
        )

        ## TODO (varun) : Actually submit a store job on a helper thread
        self.handler.transfer_async(internal_job_metadata)

        self._pending_jobs[job_id] = internal_job_metadata

    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        prepare_store_blocks_output = prepare_store_blocks(
            keys, self.cache_policy, self.block_tracker
        )
        if prepare_store_blocks_output is None:
            # Cannot prepare storage blocks
            return None

        to_evict = prepare_store_blocks_output.to_evict

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        # build store specs for allocated blocks
        store_spec = self._get_load_store_spec(prepare_store_blocks_output.store_keys)

        # materialize evicted keys
        # TODO (varun): This blocks the main thread
        self._evict_keys(to_evict)

        return PrepareStoreOutput(
            keys_to_store=prepare_store_blocks_output.store_keys,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    def submit_load(self, job_metadata: JobMetadata) -> None:
        """
        Submit an async job to load blocks from this tier to primary tier.

        Args:
            job_metadata: Job metadata including job_id, keys, and
                          spec for writing blocks into the primary tier.
        """
        job_id = job_metadata.job_id
        keys = job_metadata.keys
        primary_block_ids = job_metadata.block_ids

        assert len(keys) == len(primary_block_ids), (
            f"Length mismatch: {len(keys)} keys but {len(primary_block_ids)} block_ids"
        )

        assert all([self.cache_policy.get(k) is not None for k in keys]), (
            "Can't find block to load"
        )

        # Create internal job metadata
        internal_job_metadata = StorageJobMetadata(
            job_id=job_id,
            keys=keys,
            storage_paths=[self._hash_to_file(k) for k in keys],
            block_ids=primary_block_ids,
            is_store=False,
        )

        self.handler.transfer_async(internal_job_metadata)

        self._pending_jobs[job_id] = internal_job_metadata

    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        """
        Prepare storage dataobjects for load.
        """
        for key in keys:
            block = self.cache_policy.get(key)
            assert block is not None, f"Block {key!r} not found in cache"
            assert block.is_ready, f"Block {key!r} is not ready for reading"
            block.ref_cnt += 1

    def get_finished(self) -> Iterable[JobResult]:
        """
        Poll for finished async jobs.

        Returns:
            Iterable of JobResult objects for all jobs that have
            finished since the last call.
        """

        # Move pending jobs to completed
        job_results: list[JobResult] = self.handler.get_finished()
        for job_result in job_results:
            job_id = job_result.job_id
            # TODO (varun): This is expensive - get is_store from
            # elsewhere.
            is_store = self._pending_jobs[job_id].is_store
            if is_store:
                self._complete_store_job(job_result)
            else:
                self._complete_load_job(job_result)

        return job_results

    def _complete_store_job(self, result: JobResult):
        stored_keys: list[OffloadKey] = []
        job_id = result.job_id
        success = result.success
        job_meta: StorageJobMetadata = self._pending_jobs.pop(job_id)

        if success:
            for key in job_meta.keys:
                block = self.cache_policy.get(key)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    stored_keys.append(key)
        else:
            # (varun) : If stores for some keys passed. This will
            # leave unaccounted files in the storage.
            for key in job_meta.keys:
                block = self.cache_policy.get(key)
                if block is not None and not block.is_ready:
                    self.cache_policy.remove(key)
                    self.block_tracker.free_block(block)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    def _complete_load_job(self, result: JobResult):
        """Complete a load job."""
        job_meta = self._pending_jobs.pop(result.job_id)
        if result.success:
            for key in job_meta.keys:
                block = self.cache_policy.get(key)
                assert block is not None, f"Block {key!r} not found"
                assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
                assert block.ref_cnt != 1, "Multiple Storage loads in progress!"
                block.ref_cnt -= 1
        else:
            logger.debug("Storage Job {job_meta} failed!")
            for key in job_meta.keys:
                # There can be only one load for a key. This is safe.
                block = self.cache_policy.get(key)
                if block is not None and not block.is_ready:
                    self.cache_policy.remove(key)
                    self.block_tracker.free_block(block)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as recently used (move to end of LRU list).

        Args:
            keys: Blocks to mark as recently used.
            req_context: Per-request context.
        """
        self.cache_policy.touch(keys)

    @staticmethod
    def get_tier_type() -> str:
        return StorageLoadStoreSpec.medium()

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()
