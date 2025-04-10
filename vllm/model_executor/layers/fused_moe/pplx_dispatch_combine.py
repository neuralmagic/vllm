# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

import pplx_kernels as pplx
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.utils import _fp8_quantize
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)
from vllm.logger import init_logger


logger = init_logger(__name__)

# Note use: layer.get_all_to_all() to get an AllToAll instance
# The max_num_tokens, world_size and dp_size must be the same
# as the ones used to create the AllToAll.  Unfortunately, there's
# no way(?) to extract this info from AllToAll
class PplxDispatchCombine(mk.FusedMoEQuantizeDispatchCombine):

    def __init__(self,
                 a2a: pplx.AllToAll,
                 max_num_tokens: int,
                 world_size: int,
                 dp_size: int,
                 rank: int,
                 quant_dtype: Optional[torch.dtype] = None,
                 block_shape: Optional[List[int]] = None):
        super().__init__()
        self.a2a = a2a
        self.block_shape = block_shape
        self.max_num_tokens = max_num_tokens                             # not right?
        self.dp_num_tokens = max_num_tokens #* (world_size // dp_size)   # this is not right
        self.world_size = world_size  # debug
        self.rank = rank              # debug
        self.quant_dtype = quant_dtype
        print(f"max_num_tokens = {max_num_tokens}")
        print(f"dp_num_tokens = {self.dp_num_tokens}")
        print(f"world_size = {world_size}")
        print(f"dp_size = {dp_size}")

    def dispatch(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        rank_topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        #assert expert_map is None ????

        # Is this always going to be a1.device?
        device = a1.device

        if self.quant_dtype == torch.float8_e4m3fn:
            per_act_token = a1_scale.numel(
            ) != 1 if a1_scale is not None else (
                a2_scale.numel() != 1 if a2_scale is not None else False)

            a1q, a1q_scale = _fp8_quantize(
                a1,
                a1_scale,
                self.block_shape,
                per_act_token,
            )
        else:
            a1q = a1
            a1q_scale = a1_scale

        expert_num_tokens = torch.zeros(
            num_experts,
            dtype=torch.int32,
            device=a1.device,
        )
        expert_num_tokens.fill_(-1)

        expert_x = torch.empty(
            (num_experts, self.dp_num_tokens, a1q.shape[-1]),
            dtype=a1q.dtype,
            device=a1.device,
        )
        expert_x.fill_(torch.nan)

        expert_x_scale: Optional[torch.Tensor] = None
        if a1q.dtype.itemsize == 1:
            float32_size = torch.float32.itemsize
            block_size = (self.block_shape[0] if self.block_shape is not None
                          else 1) * float32_size
            expert_x_scale = torch.empty(
                (
                    num_experts,
                    expert_x.size(1),
                    (expert_x.size(2) + block_size - 1) // block_size,
                ),
                dtype=torch.float32,
                device=a1.device,
            )

        # This argument is optional, defaults to indices.shape[0]
        bound_m = None #get_forward_context().dp_metadata.dp_rank_num_tokens

        # TODO: optimize this?
        indices = rank_topk_ids.to(dtype=torch.uint32)

        # rank_topk_ids is (num_tokens, experts_per_token)
        torch.set_printoptions(profile="full")

        print(f"num_experts = {num_experts}")
        print(f"A1Q = {a1q.shape}")
        print(f"num tokens = {expert_num_tokens.shape}")
        print(f"expert_x = {expert_x.shape}")
        if expert_x_scale is not None:
            print(f"expert_x_scale = {expert_x_scale.shape}")
        print(f"bound_m = {bound_m}")
        #print(f"topk_ids = {rank_topk_ids.shape} {rank_topk_ids}")
        print(f"indices = {indices.shape} {torch.unique(indices.flatten())}")

        #######
        expert_token_from: list[list[tuple[int, int]]] = [
            [] for _ in range(num_experts)
        ]
        i_rank = self.rank
        for token_idx in range(a1q.shape[0]):
            for expert_idx in indices[token_idx]:
                expert_token_from[expert_idx].append((i_rank, token_idx))
        #######

        print("DISPATCH")
        self.a2a.dispatch(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=expert_x,
            out_expert_x_scale=expert_x_scale,
            dp_x=a1q,
            dp_x_scale=a1q_scale,
            indices=indices,
            bound_m=bound_m,
        )

        # expert_num_tokens, use this to reformat expert_x/expert_x_scale
        #print(f"expert_num_tokens = {expert_num_tokens} {(expert_num_tokens > 0).nonzero()}")
        print(f"expert_num_tokens = {(expert_num_tokens > 0).nonzero()}")

        print(f"EXPERT_X = {expert_x.shape} total={expert_x.numel()} nan={torch.isnan(expert_x).sum()}")
        for i in range(expert_x.shape[0]):
            if torch.isnan(expert_x[i]).sum() < expert_x[i].numel():
                print(i)
        print("END_EXPERT_X")

        num_local_experts = num_experts // self.world_size
        for i_rank in range(self.world_size):
            if i_rank != self.rank:
                continue
            for i_local_expert in range(num_local_experts):
                expert_idx = i_rank * num_local_experts + i_local_expert
                cnt_tokens = int(expert_num_tokens[i_local_expert].item())
                logger.debug(
                    "Expert #%d on Rank %d: %d tokens",
                    expert_idx,
                    self.rank,
                    cnt_tokens,
                )
                #assert cnt_tokens == len(expert_token_from[expert_idx]), f"{cnt_tokens} != {len(expert_token_from[expert_idx])}"
                #cnt_from_dp_rank = [0] * self.dp_num_tokens
                for i_token in range(cnt_tokens):
                    src_dp_rank, src_token_idx = expert_token_from[expert_idx][i_token]
                    #cnt_from_dp_rank[src_dp_rank] += 1
                    dst_x = expert_x[i_local_expert, i_token]
                    logger.debug(
                        "  x[%d] (from DP Rank %d Token %d): %s",
                        i_token,
                        src_dp_rank,
                        src_token_idx,
                        dst_x.cpu(),
                    )

        #return expert_x, expert_x_scale
        return (expert_x.view(-1, expert_x.shape[-1]),
                expert_x_scale.view(-1, expert_x_scale[-1]) if expert_x_scale is not None else expert_x_scale)

    def combine(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        # This argument is optional
        bound_m = None #get_forward_context().dp_metadata.dp_rank_num_tokens

        assert output.shape[0] == self.max_num_tokens
        assert output.shape[1] == fused_expert_output.shape[-1]

        self.a2a.combine(out_tokens=output,
                         indices=topk_ids.to(torch.uint32),
                         weights=topk_weights,
                         expert_y=fused_expert_output,
                         bound_m=bound_m)
