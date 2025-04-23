# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

import pplx_kernels as pplx
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.utils import _fp8_quantize
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)
from vllm.logger import init_logger


logger = init_logger(__name__)

# Note use: layer.get_all_to_all() to get an AllToAll instance
# The max_num_tokens, world_size and dp_size must be the same
# as the ones used to create the AllToAll.
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
        self.max_num_tokens = max_num_tokens
        self.world_size = world_size  # debug
        self.dp_size = dp_size
        self.rank = rank              # debug
        self.quant_dtype = quant_dtype
        if False:
            print(f"max_num_tokens = {max_num_tokens}")
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        #assert expert_map is None?

        logger.debug(f"DISPATCH START {self.rank}")

        assert a1.dim() == 2
        assert a1.shape[0] <= self.max_num_tokens

        num_tokens, hidden_dim = a1.shape   # M, K
        device = a1.device                  # Is this always going to be a1.device?

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

        # XXXXXXXXX TODO: distribute experts properly
        assert num_experts % self.world_size == 0
        num_local_experts = num_experts // self.world_size

        logger.debug(f"{self.rank}: {num_local_experts}, device={device}")

        expert_num_tokens = torch.empty(
            num_local_experts,
            dtype=torch.int32,
            device=device,
        )
        #expert_num_tokens.fill_(-1)  # debugging, remove later

        num_dp = self.world_size // self.dp_size
        logger.debug(f"GOT HERE A {self.rank}: {self.max_num_tokens} {num_dp} {hidden_dim}")
        expert_x = torch.empty(
            (num_local_experts, self.max_num_tokens * num_dp, hidden_dim),
            dtype=a1q.dtype,
            device=device,
        )
        expert_x.fill_(0) #torch.nan   # debugging, remove later

        logger.debug(f"GOT HERE B {self.rank}")

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
                device=device,
            )

        logger.debug(f"GOT HERE C {self.rank}")

        # This argument is optional, defaults to indices.shape[0]
        #bound_m = get_forward_context().dp_metadata.dp_rank_num_tokens

        # This causes a deadlock????
        #bound_m = torch.tensor([num_tokens], dtype=torch.uint32, device=device)
        bound_m = None

        # TODO: optimize this?
        indices = rank_topk_ids.to(dtype=torch.uint32)

        logger.debug(f"GOT HERE D {self.rank}")

        # rank_topk_ids is (num_tokens, experts_per_token)
        if False:
            torch.set_printoptions(profile="full")
            logger.debug(f"num_experts = {num_experts}")
            logger.debug(f"A1Q = {a1q.shape}")
            logger.debug(f"expert num tokens = {expert_num_tokens.shape}")
            logger.debug(f"expert_x = {expert_x.shape}")
            if expert_x_scale is not None:
                logger.debug(f"expert_x_scale = {expert_x_scale.shape}")
            logger.debug(f"bound_m = {bound_m}")
            #logger.debug(f"topk_ids = {rank_topk_ids.shape} {rank_topk_ids}")
            logger.debug(f"indices = {indices.shape} {torch.unique(indices.flatten())}")
            logger.debug("DISPATCH")

        #######
        if False:
            expert_token_from: list[list[tuple[int, int]]] = [
                [] for _ in range(num_experts)
            ]
            for i_rank in range(num_dp):
                for token_idx in range(num_tokens):
                    for expert_idx in indices[token_idx]:
                        expert_token_from[expert_idx].append((i_rank, token_idx))
        #######

        #print (f"a2a : ws {self.a2a.world_size} | ds {self.a2a.dp_size} | max_num_tokens {self.a2a.max_num_tokens} | rank {self.a2a.rank} | num_experts {self.a2a.num_experts} | experts_per_token {self.a2a.experts_per_token}")
        #print (f"topk indices {indices}")
        #print (f"doing quant {self.quant_dtype} ")

        #a1q_decoy = torch.empty_like(a1q, device = a1q.device, dtype=a1q.dtype)
        #for i in range(num_tokens):
        #    a1q_decoy[i, :].fill_(i)
        #a1q_scale_decoy = None
        #if a1q_scale:
        #    a1q_scale_decoy = torch.ones_like(a1q_scale, device=a1_scale.device, dtype = a1_scale.dtype)

        #a1q = a1q_decoy
        #a1q_scale = a1q_scale_decoy

        #print (f"a1q_decoy {a1q_decoy}")

        self.a2a.dispatch(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=expert_x,
            out_expert_x_scale=expert_x_scale,
            dp_x=a1q,
            dp_x_scale=a1q_scale,
            indices=indices,
            bound_m=bound_m,
        )


        logger.debug(f"GOT HERE E {self.rank}")

        # expert_num_tokens, use this to reformat expert_x/expert_x_scale
        if False:
            logger.debug(f"expert_num_tokens = {(expert_num_tokens > 0).nonzero()}")

            for i_rank in range(self.world_size):  # why world size here and num_dp above?
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
                    assert cnt_tokens == len(expert_token_from[expert_idx]), f"{cnt_tokens} != {len(expert_token_from[expert_idx])}"
                    cnt_from_dp_rank = [0] * num_dp
                    #logger.debug(f"CNT_TOKENS {cnt_tokens}")
                    for i_token in range(cnt_tokens):
                        src_dp_rank, src_token_idx = expert_token_from[expert_idx][i_token]
                        cnt_from_dp_rank[src_dp_rank] += 1
                        dst_x = expert_x[i_local_expert, i_token]
                        logger.debug(
                            "  x[%d] (from DP Rank %d Token %d): %s",
                            i_token,
                            src_dp_rank,
                            src_token_idx,
                            dst_x.cpu(),
                        )

        logger.debug(f"DISPATCH DONE {self.rank}")

        #print (f"expert_x {expert_x.shape} | expert_x_scale {expert_x_scale.shape} | expert_num_tokens {expert_num_tokens.shape}")

        return expert_x, expert_x_scale, expert_num_tokens

    def combine(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        device = fused_expert_output.device
        #device = torch.device("cuda", self.rank)
        #device = get_dp_group().device
        #assert fused_expert_output.device == device

        logger.debug(f"COMBINE START {self.rank}")

        # This argument is optional
        #bound_m = get_forward_context().dp_metadata.dp_rank_num_tokens
        #num_tokens = fused_expert_output.shape[0]   # M
        #bound_m = torch.tensor([num_tokens], dtype=torch.uint32, device=device)
        bound_m = None

        assert output.shape[0] <= self.max_num_tokens
        assert output.shape[1] == fused_expert_output.shape[-1]

        #topk_weights_decoy = torch.ones_like(topk_weights, device=topk_weights.device, dtype=topk_weights.dtype)
        #topk_weights = topk_weights_decoy
        #output.fill_(0)

        self.a2a.combine(out_tokens=output,
                         indices=topk_ids.to(torch.uint32),
                         weights=topk_weights,
                         expert_y=fused_expert_output,
                         bound_m=bound_m)

        #if fused_expert_output.shape[0] == 6:
        #if self.rank == 0:
        #    print (f"combine fused expert output : {fused_expert_output}")
        #    print (f"combine output : {output}")

        #print (output)

        logger.debug(f"COMBINE END {self.rank}")
