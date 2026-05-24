# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Common utility to generally track/allocate offload blocks
"""

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus

# TODO (varun): Move this to cpu/policies maybe ?


class OffloadBlockTracker:
    def __init__(self, num_blocks: int):
        self.num_blocks: int = num_blocks
        self.num_allocated_blocks: int = 0
        self.free_list: list[int] = []

    def get_num_free_blocks(self) -> int:
        num_fresh_blocks = max(0, self.num_blocks - self.num_allocated_blocks)
        return len(self.free_list) + num_fresh_blocks

    def allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
        num_fresh = min(len(keys), self.num_blocks - self.num_allocated_blocks)
        num_reused = len(keys) - num_fresh
        assert len(self.free_list) >= num_reused

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(self.num_allocated_blocks))
            self.num_allocated_blocks += 1
        assert self.num_allocated_blocks <= self.num_blocks, (
            f" {self.num_blocks=}, {self.num_allocated_blocks=}"
        )

        # allocate reused blocks
        for _ in range(num_reused):
            blocks.append(BlockStatus(self.free_list.pop()))
        return blocks

    def free_block(self, block: BlockStatus) -> None:
        self.free_list.append(block.block_id)
