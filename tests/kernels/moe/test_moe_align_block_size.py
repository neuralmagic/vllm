# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size_triton)


BLOCK_SIZE = [32, 64, 128, 256]
NUM_TOKENS = [1, 3, 7, 16, 256, 2256, 4096]
TOPK = [1, 4, 16, 64]
NUM_EXPERTS = [64, 160, 256, 257, 260, 264]

@pytest.mark.parametrize(
    "block_size,num_tokens,topk,num_experts",
    list(itertools.product(BLOCK_SIZE, NUM_TOKENS, TOPK, NUM_EXPERTS)))
def test_moe_align_block_size_compare_implementations(block_size, num_tokens,
                                                      topk, num_experts):
    topk_ids = torch.stack([
        torch.randperm(num_experts, dtype=torch.int32, device="cuda")[:topk]
        for _ in range(num_tokens)
    ])

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)

    sorted_ids_cuda = torch.empty((max_num_tokens_padded, ),
                                  dtype=torch.int32,
                                  device=topk_ids.device)
    sorted_ids_cuda.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids_cuda = torch.zeros((max_num_m_blocks, ),
                                  dtype=torch.int32,
                                  device=topk_ids.device)
    num_tokens_post_pad_cuda = torch.empty((1),
                                           dtype=torch.int32,
                                           device=topk_ids.device)

    sorted_ids_triton = torch.empty_like(sorted_ids_cuda)
    sorted_ids_triton.fill_(topk_ids.numel())
    expert_ids_triton = torch.zeros_like(expert_ids_cuda)
    num_tokens_post_pad_triton = torch.empty_like(num_tokens_post_pad_cuda)

    ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids_cuda,
        expert_ids_cuda,
        num_tokens_post_pad_cuda,
    )

    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids_triton,
        expert_ids_triton,
        num_tokens_post_pad_triton,
    )

    assert torch.allclose(expert_ids_cuda, expert_ids_triton), (
        f"Expert IDs mismatch for block_size={block_size}, "
        f"num_tokens={num_tokens}, topk={topk}\n"
        f"CUDA expert_ids: {expert_ids_cuda}\n"
        f"Triton expert_ids: {expert_ids_triton}")

    assert torch.allclose(
        num_tokens_post_pad_cuda, num_tokens_post_pad_triton), (
            f"Num tokens post pad mismatch for block_size={block_size}, "
            f"num_tokens={num_tokens}, topk={topk}\n"
            f"CUDA num_tokens_post_pad: {num_tokens_post_pad_cuda}\n"
            f"Triton num_tokens_post_pad: {num_tokens_post_pad_triton}")

def _test_moe_align_block_size_expert_map(topk_ids: torch.Tensor,
                                          block_size: int,
                                          global_num_experts: int,
                                          local_num_experts: int,
                                          ep_rank: int,
                                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return sorted_ids and expert_ids, num_tokens_post_pad
    """

    def test_correctness(sorted_ids: torch.Tensor,
                         expert_ids: torch.Tensor,
                         expert_map: torch.Tensor):
        num_tokens, num_topk = topk_ids.size()

        sorted_ids_cpu = sorted_ids.to("cpu") 
        expert_ids_cpu = expert_ids.to("cpu")
        topk_ids_cpu = topk_ids.to("cpu")
        expert_map_cpu = expert_map.to("cpu")

        for row_id, token_id in enumerate(sorted_ids_cpu):
            assert token_id <= num_tokens * num_topk
            if token_id == num_tokens * num_topk:
                continue
            
            token_idx = token_id // num_topk 
            expected_expert_id = expert_map_cpu[topk_ids_cpu[token_idx, token_id % num_topk]]
            block_id = row_id // block_size
            assert expert_ids_cpu[block_id] == expected_expert_id

            # This topk-id has been verified to be correct. Mark it as done by
            # turning it to -1.
            topk_ids_cpu[token_idx, token_id % num_topk] = -1

        # Any unverified topk-id must no belong to this slice of experts.
        # Turn expert ids from other slices to -1.
        topk_ids_cpu = torch.where(topk_ids_cpu != -1, expert_map_cpu[topk_ids_cpu], topk_ids_cpu) 
        assert torch.all(topk_ids_cpu == -1)

    def build_expert_map():
        expert_map = torch.full((global_num_experts, ),
                                fill_value=-1,
                                dtype=torch.int32)
        s = ep_rank * local_num_experts
        e = s + local_num_experts
        expert_map[s:e] = torch.tensor(list(range(local_num_experts)))
        return expert_map.to(device="cuda")

    max_num_tokens_padded = topk_ids.numel() + local_num_experts * (block_size - 1)
    max_num_m_blocks = max_num_tokens_padded // block_size

    sorted_ids = torch.empty((max_num_tokens_padded, ),
                                  dtype=torch.int32,
                                  device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())

    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)

    expert_ids = torch.zeros((max_num_m_blocks, ),
                                  dtype=torch.int32,
                                  device=topk_ids.device)

    expert_map = build_expert_map()

    #print (f"topk ids {topk_ids} | expert map {expert_map}", flush=True)

    ops.moe_align_block_size(
        topk_ids,
        local_num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map,
    )

    #torch.cuda.synchronize()
    #print (f"expert_ids {expert_ids.shape} | sorted tok ids {sorted_ids.shape}", flush=True)
    #torch.set_printoptions(profile="full")
    #print (f"expert_ids {expert_ids} | sorted tok ids {sorted_ids}", flush=True)
    #torch.set_printoptions(profile="default")

    test_correctness(sorted_ids, expert_ids, expert_map)


#@pytest.mark.parametrize("block_size", [128])
#@pytest.mark.parametrize("num_tokens", [4])
#@pytest.mark.parametrize("topk", [8])
#@pytest.mark.parametrize("num_experts", [64])
#@pytest.mark.parametrize("num_ep", [4])

NUM_EP = [4]
@pytest.mark.parametrize(
    "block_size,num_tokens,topk,num_experts, num_ep",
    list(itertools.product(BLOCK_SIZE, NUM_TOKENS, TOPK, NUM_EXPERTS, NUM_EP)))
def test_moe_align_block_size_expert_map(block_size, num_tokens,
                                         topk, num_experts,
                                         num_ep):
    topk_ids = torch.stack([
        torch.randperm(num_experts, dtype=torch.int32, device="cuda")[:topk]
        for _ in range(num_tokens)
    ])

    for ep_rank in range(num_ep):
        _test_moe_align_block_size_expert_map(topk_ids = topk_ids,
                                              block_size = block_size,
                                              global_num_experts = num_experts,
                                              local_num_experts =  num_experts // num_ep,
                                              ep_rank = ep_rank)

if __name__ == "__main__":
    pytest.main([__file__])
