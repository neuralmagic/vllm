# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

import pplx_kernels as pplx
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)


def rank_chunk(num, r, w):
    rem = num % w
    return (num // w) + (1 if r < rem else 0)


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
                 block_shape: Optional[List[int]] = None,
                 per_act_token: bool = False):
        super().__init__()
        assert max_num_tokens > 0
        self.a2a = a2a
        self.block_shape = block_shape
        self.max_num_tokens = max_num_tokens
        self.world_size = world_size
        self.dp_size = dp_size
        self.rank = rank
        self.quant_dtype = quant_dtype
        self.per_act_token = per_act_token

    def dispatch(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        rank_topk_weights: torch.Tensor,
        rank_topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        num_tokens = a1.shape[0]  # M
        hidden_dim = a1.shape[-1]  # K

        assert rank_topk_ids.shape[0] == num_tokens
        # assert expert_map is None, "NYI"

        # Is this always going to be a1.device?
        device = a1.device

        if apply_router_weight_on_input:
            topk = rank_topk_ids.shape[1]
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, \
                "apply_router_weight_on_input is only implemented for topk=1"
            a1 = a1 * rank_topk_weights.to(a1.dtype)

        # print("PER ACT TOKEN:", self.per_act_token)
        
        # print("A1 DTYPE", a1.dtype)
        # print("A1 SHAPES:", a1.shape, a1_scale.shape)
        repeat_cols = 4
        if self.per_act_token:
            repeat_rows = 1
            a1q, a1q_scale = moe_kernel_quantize_input(a1, None,
                                                    self.quant_dtype,
                                                    self.per_act_token,
                                                    self.block_shape)
        else:
            repeat_rows = a1.shape[0]
            a1q, a1q_scale = moe_kernel_quantize_input(a1, a1_scale,
                                                    self.quant_dtype,
                                                    self.per_act_token,
                                                    self.block_shape)

        a1q_scale = a1q_scale.repeat(repeat_rows, repeat_cols)

        # # # # print("pplx_dispatch_combine a1:", a1)   
        # print("full a1_scale:", a1_scale)
        # # # # print("pplx_dispatch_combine a1q:", a1q)
        # print("full a1q_scale:", a1q_scale)

        rem_experts = num_experts % self.world_size
        num_local_experts = ((num_experts // self.world_size) +
                             (1 if self.rank < rem_experts else 0))
        # print("num_local_experts in dispatch:", num_local_experts, self.rank)

        expert_num_tokens = torch.empty(
            num_local_experts,
            dtype=torch.int32,
            device=device,
        )

        num_dp = self.world_size // self.dp_size
        expert_x = torch.empty(
            (num_local_experts, self.max_num_tokens * num_dp, hidden_dim),
            dtype=a1q.dtype,
            device=device,
        )

        # print("Block shape:", self.block_shape, ", hidden_dim:", hidden_dim)

        expert_x_scale: Optional[torch.Tensor] = None
        if a1q.dtype.itemsize == 1:
            float32_size = torch.float32.itemsize
            block_size = (self.block_shape[0] if self.block_shape is not None
                          else 1) * float32_size
            # print("block_size:", block_size, "expert_x size 2:", expert_x.size(2))
            expert_x_scale = torch.empty(
                (
                    num_local_experts,
                    expert_x.size(1),
                    (expert_x.size(2) + block_size - 1) // block_size,
                ),
                dtype=torch.float32,
                device=device,
            )

        # This argument is optional, defaults to indices.shape[0]
        bound_m = torch.tensor([num_tokens], dtype=torch.uint32, device=device)

        # TODO: optimize this?
        indices = rank_topk_ids.to(dtype=torch.uint32)

        # print("before dispatch:", expert_num_tokens)
        # print("dispatch shapes:", expert_x.shape, expert_x_scale.shape,
        #       a1q.shape, a1q_scale.shape, rank_topk_ids.shape)
        # print("dispatch types:", expert_x.dtype, expert_x_scale.dtype,
        #       a1q.dtype, a1q_scale.dtype, rank_topk_ids.dtype)
        # print("dispatch indices:", indices)
        self.a2a.dispatch(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=expert_x,
            out_expert_x_scale=expert_x_scale,
            dp_x=a1q,
            dp_x_scale=a1q_scale,
            indices=indices,
            bound_m=bound_m,
        )
        # print("dispatched:", expert_x.shape, expert_x_scale.shape,
        #       expert_num_tokens.shape)
        # print("dispatched types:", expert_x.dtype, expert_x_scale.dtype,
        #       a1q.dtype, a1q_scale.dtype, rank_topk_ids.dtype)
        # print("expert x:", expert_x)
        # print("expert a1q:", a1q)
        # print("expert_x_scale x:", expert_x_scale)
        # print("expert_num_tokens x:", expert_num_tokens)
        # # print("after dispatch:", expert_num_tokens)
        if self.per_act_token:
            expert_x_scale = expert_x_scale[:, :, 0:1]
        else:
            expert_x_scale = expert_x_scale[0:1, 0:1, 0:1]
        # print("returned:", expert_x.shape, expert_x_scale.shape,
        #       expert_num_tokens.shape)
        return expert_x, expert_x_scale, expert_num_tokens

    def combine(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        # This argument is optional
        num_tokens = output.shape[0]  # M
        bound_m = torch.tensor([num_tokens],
                               dtype=torch.uint32,
                               device=fused_expert_output.device)

        assert topk_ids.shape[0] <= num_tokens
        assert output.shape[0] <= self.max_num_tokens, \
            f"{output.shape[0]} <= {self.max_num_tokens}"
        assert output.shape[1] == fused_expert_output.shape[-1]

        # Set weights to 1 if we did them in dispatch. This is hacky.
        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        # torch.set_printoptions(profile="full")
        # print("output:", output)
        # print("fused_expert_output:", fused_expert_output)
        # print("OUTPUT SHAPE:", output.shape)
        # print("FUSED EXPERT OUTPUT SHAPE:", fused_expert_output.shape)
        # torch.set_printoptions(profile="default")
        self.a2a.combine(out_tokens=output,
                         indices=topk_ids.to(torch.uint32),
                         weights=topk_weights,
                         expert_y=fused_expert_output,
                         bound_m=bound_m)
        # print("combined output:", output)
