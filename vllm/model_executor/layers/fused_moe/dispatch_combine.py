# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Tuple

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    _moe_unpermute_and_reduce)
from vllm.model_executor.layers.fused_moe.utils import _fp8_quantize


class StandardDispatchCombine(mk.FusedMoEQuantizeDispatchCombine):

    def __init__(self,
                 quant_dtype: Optional[torch.dtype] = None,
                 block_shape: Optional[list[int]] = None):
        super().__init__()
        self.block_shape = block_shape
        self.quant_dtype = quant_dtype

    def dispatch(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
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

        return a1q, a1q_scale, None

    def combine(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        _moe_unpermute_and_reduce(output, fused_expert_output, None,
                                  topk_weights)
