# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Collection
from dataclasses import dataclass

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec, OffloadKey
from vllm.v1.kv_offload.cpu.offload_block_tracker import OffloadBlockTracker
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


@dataclass
class PrepareStoreBlocksOutput:
    to_evict: list[OffloadKey]
    store_blocks: list[BlockStatus]
    store_keys: list[OffloadKey]


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.
    """

    @staticmethod
    def medium() -> str:
        return "CPU"


def prepare_store_blocks(
    keys: Collection[OffloadKey],
    cache_policy: CachePolicy,
    block_tracker: OffloadBlockTracker,
) -> PrepareStoreBlocksOutput | None:
    # filter out blocks that are already stored
    keys_to_store = [k for k in keys if cache_policy.get(k) is None]

    if not keys_to_store:
        return PrepareStoreBlocksOutput(to_evict=[], store_blocks=[], store_keys=[])

    num_blocks_to_evict = len(keys_to_store) - block_tracker.get_num_free_blocks()

    to_evict: list[OffloadKey] = []
    if num_blocks_to_evict > 0:
        # Blocks from the original input are excluded from eviction candidates:
        # a block that was already stored must remain in the cache after this call.
        protected = set(keys)
        evicted = cache_policy.evict(num_blocks_to_evict, protected)
        if evicted is None:
            return None
        for key, block in evicted:
            block_tracker.free_block(block)
            to_evict.append(key)

    blocks = block_tracker.allocate_blocks(keys_to_store)
    assert len(blocks) == len(keys_to_store), (
        "Block pool did not allocate the expected number of blocks"
    )

    for key, block in zip(keys_to_store, blocks):
        cache_policy.insert(key, block)

    return PrepareStoreBlocksOutput(
        to_evict=to_evict, store_blocks=blocks, store_keys=keys_to_store
    )
