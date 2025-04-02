# SPDX-License-Identifier: Apache-2.0
"""Fused MoE kernel."""
from typing import Optional

import torch

from vllm import _custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.utils import (_resize_cache,
                                                        _fp8_perm)


#TODO make the grouped gemm kernel consistent with scaled gemm kernel
def cutlass_moe_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    ab_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    ab_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.half,
) -> torch.Tensor:
    """
    This function computes a a8w8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.

    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1_q (torch.Tensor): The first set of fp8-quantized expert weights.
        Shape: [num_experts, K, 2N] (the weights are passed transposed)
    - w2_q (torch.Tensor): The second set of fp8-quantized expert weights.
        Shape: [num_experts, N, K] (the weights are passed transposed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts] or [num_experts, 2N]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts] or [num_experts, K]
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - ab_strides1 (torch.Tensor): The input and weights strides of the first
        grouped gemm.
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    - ab_strides2 (torch.Tensor): The input and weights strides of the second
        grouped gemm.
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [M]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [M]
    - out_dtype (torch.Tensor): The output tensor type.

    Returns:
    - torch.Tensor: The fp16 output tensor after applying the MoE layer.
    """

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.float8_e4m3fn
    assert w2_q.dtype == torch.float8_e4m3fn
    assert a.shape[1] == w1_q.shape[1], "Hidden size mismatch w1"
    assert w1_q.shape[2] == w2_q.shape[1] * 2, "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert a1_scale is None or a1_scale.dim(
    ) == 0 or a1_scale.shape[0] == 1 or a1_scale.shape[0] == a.shape[
        0], "Input scale shape mismatch"
    assert w1_scale.dim() == 1 or w1_scale.shape[1] == 1 or w1_scale.shape[
        1] == w1_q.shape[2], "W1 scale shape mismatch"
    assert w2_scale.dim() == 1 or w2_scale.shape[1] == 1 or w2_scale.shape[
        1] == w2_q.shape[2], "W2 scale shape mismatch"
    assert w1_q.shape[0] == w2_q.shape[0], "Weights expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[
        0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[
        0], "w2 scales expert number mismatch"
    assert a2_scale is None or a1_scale is None or a2_scale.shape == a1_scale.shape, "Intermediate scale shape mismatch"  # noqa: E501
    assert ab_strides1.shape[0] == w1_q.shape[
        0], "AB Strides 1 expert number mismatch"
    assert c_strides1.shape[0] == w1_q.shape[
        0], "C Strides 1 expert number mismatch"
    assert ab_strides2.shape[0] == w2_q.shape[
        0], "AB Strides 2 expert number  mismatch"
    assert c_strides2.shape[0] == w2_q.shape[
        0], "C Strides 2 expert number mismatch"
    assert out_dtype in [torch.half, torch.bfloat16], "Invalid output dtype"

    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(1)
    n = w2_q.size(1)

    topk = topk_ids.size(1)
    per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
        a2_scale.numel() != 1 if a2_scale is not None else False)

    a_q, a1_scale = ops.scaled_fp8_quant(
        a, a1_scale, use_per_token_if_dynamic=per_act_token)
    device = a_q.device

    expert_offsets = torch.empty((num_experts + 1),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes1 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes2 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    ops.get_cutlass_moe_mm_data(topk_ids, expert_offsets, problem_sizes1,
                                problem_sizes2, a_map, c_map, num_experts, n,
                                k)

    rep_a_q = a_q.view(dtype=torch.uint8)[a_map].view(dtype=a_q.dtype)
    rep_a1_scales = a1_scale[a_map] if per_act_token else a1_scale

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=out_dtype)
    c2 = torch.empty((m * topk, k), device=device, dtype=out_dtype)

    ops.cutlass_moe_mm(c1, rep_a_q, w1_q, rep_a1_scales, w1_scale,
                       expert_offsets[:-1], problem_sizes1, ab_strides1,
                       ab_strides1, c_strides1)

    intermediate = torch.empty((m * topk, n), device=device, dtype=out_dtype)
    torch.ops._C.silu_and_mul(intermediate, c1)

    intemediate_q, a2_scale = ops.scaled_fp8_quant(
        intermediate, a2_scale, use_per_token_if_dynamic=per_act_token)

    ops.cutlass_moe_mm(c2, intemediate_q, w2_q, a2_scale, w2_scale,
                       expert_offsets[:-1], problem_sizes2, ab_strides2,
                       ab_strides2, c_strides2)

    return (c2[c_map].view(m, topk, k) *
            topk_weights.view(m, topk, 1).to(out_dtype)).sum(dim=1)


class CutlassDispatch(mk.FusedMoEDispatchQuantize):
    def __init__(self):
        super().__init__()

    def apply(
            self,
            hidden_states: torch.Tensor,
            hidden_states_scale: Optional[torch.Tensor],
            topk_ids: torch.Tensor,
            num_experts: int,
            expert_map: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        m = hidden_states.size(0)
        k = w1_q.size(1)
        n = w2_q.size(1)
        device = hidden_states.device

        # a2_scale.numel() != 1 if a2_scale is not None else False
        per_act_token = hidden_states_scale.numel() != 1 if hidden_states_scale is not None else False

        expert_offsets = torch.empty((num_experts + 1),
                                     dtype=torch.int32,
                                     device=device)
        problem_sizes1 = torch.empty((num_experts, 3),
                                     dtype=torch.int32,
                                     device=device)
        problem_sizes2 = torch.empty((num_experts, 3),
                                     dtype=torch.int32,
                                     device=device)

        a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
        c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

        ops.get_cutlass_moe_mm_data(topk_ids,
                                    expert_offsets,
                                    problem_sizes1,
                                    problem_sizes2,
                                    a_map, c_map,
                                    num_experts,
                                    n,
                                    k)

        rep_a_q = _fp8_perm(hidden_states, a_map)
        rep_a1_scales = hidden_states_scale[a_map] if per_act_token else hidden_states_scale

        return rep_a_q, rep_a1_scales, expert_offsets, c_map


class CutlassExperts(mk.FusedMoEExperts):
    def __init__(
            self,
            ab_strides1: torch.Tensor,
            c_strides1: torch.Tensor,
            ab_strides2: torch.Tensor,
            c_strides2: torch.Tensor,
    ):
        super().__init__()
        self.ab_strides1 = ab_strides1
        self.c_strides1 = c_strides1
        self.ab_strides2 = ab_strides2
        self.c_strides2 = c_strides2

    def workspace_shapes(
            self,
            M: int,
            N: int,
            K: int,
            topk: int,
            num_experts: int
    ) -> Tuple[int, int]:
        workspace1 = M * topk * N
        workspace2 = M * topk * K
        # return tuples????
        return (workspace1, workspace2)

    def apply(
            self,
            q_hidden_states: torch.Tensor,
            w1: torch.Tensor,
            w2: torch.Tensor,
            inplace: bool,
            activation: str,
            expert_offsets: torch.Tensor,
            w1_scale: Optional[torch.Tensor],
            w2_scale: Optional[torch.Tensor],
            a1_scale: Optional[torch.Tensor],
            a2_scale: Optional[torch.Tensor],
            workspace13: torch.Tensor,
            workspace2: torch.Tensor,
    ) -> torch.Tensor: # or None?  assume inplace?
        # chunking in here or in ModularFusedMoEKernel? ignore for now
        M = q_hidden_states.shape[0]
        E, N, _ = w1.shape
        _, K, _ = w2.shape
        topk = X
        device = q_hidden_states.device

        # fix names
        c1 = _resize_cache(workspace13, (M * topk, N))
        c2 = _resize_cache(workspace13, (M * topk, K))
        c3 = _resize_cache(workspace2, (M * topk, N // 2))

        # HACK, share these with other bits
        problem_sizes1 = torch.empty((E, 3),
                                     dtype=torch.int32,
                                     device=device)
        problem_sizes2 = torch.empty((E, 3),
                                     dtype=torch.int32,
                                     device=E)

        per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
            a2_scale.numel() != 1 if a2_scale is not None else False)

        ops.cutlass_moe_mm(c1, q_hidden_states, w1, a1_scale, w1_scale,
                           expert_offsets[:-1],
                           problem_sizes1,
                           self.ab_strides1,
                           self.ab_strides1,
                           self.c_strides1)

        if activation == "silu":
            torch.ops._C.silu_and_mul(c3, c1)
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(c3, c1)
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        intemediate_q, a2_scale = ops.scaled_fp8_quant(
            c3, a2_scale, use_per_token_if_dynamic=per_act_token)

        ops.cutlass_moe_mm(c2, intemediate_q, w2, a2_scale, w2_scale,
                       expert_offsets[:-1], problem_sizes2, self.ab_strides2,
                           self.ab_strides2, self.c_strides2)

        return c2


class CutlassUnpermuteCombine(mk.FusedMoEUnpermuteCombine):
    def __init__(self, out_dtype):
        super().__init__()
        self.out_dtype = out_dtype

    def apply(
            self,
            out: torch.Tensor,
            hidden_states: torch.Tensor,
            topk_weights: torch.Tensor,
            inv_perm: Optional[torch.Tensor],
    ) -> torch.Tensor:
        M, topk = topk_weights.shape
        K = hidden_states.shape[1]
        hidden_states = hidden_states[inv_perm, ...]
        hidden_states = hidden_states.view(M, topk, K)
        out = hidden_states.mul_(topk_weights.view(M, topk, 1).to(self.out_dtype)).sum(dim=1)
        return out


def modular_cutlass_moe_fp8(
        ab_strides1: torch.Tensor,
        c_strides1: torch.Tensor,
        ab_strides2: torch.Tensor,
        c_strides2: torch.Tensor,
        out_dtype,
) -> mk.ModularFusedMoEKernel:
    return mk.ModularFusedMoEKernel(
        CutlassDispatch(),
        CutlassExperts(
            ab_strides1,
            c_strides1,
            ab_strides2,
            c_strides2,
        ),
        CutlassUnpermuteCombine(out_dtype),
    )
