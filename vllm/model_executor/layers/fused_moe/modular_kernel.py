# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

# TODO: add comments


class FusedMoEQuantizeDispatchCombine(ABC):

    @abstractmethod
    def dispatch(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # returns (quantized+dispatched a,
        #          quantized+dispatched a1_scales)
        raise NotImplementedError

    @abstractmethod
    def combine(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,  # not reduced or weighted
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        raise NotImplementedError


# store weights, etc. here
class FusedMoEPermuteExpertsUnpermute(ABC):

    @abstractmethod
    def workspace_shapes(
        self,
        a_dtype: torch.dtype,
        M: int,
        N: int,
        K: int,
        topk: int,
        num_experts: int
    ) -> Tuple[int, int, torch.dtype]:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        a1q: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


# Note: only intended for use with a single model layer (due to temp buffers,
# constants, etc.)
class FusedMoEModularKernel(torch.nn.Module):  # should this be a module?

    def __init__(
        self,
        dispatch_combine: FusedMoEQuantizeDispatchCombine,
        fused_experts: FusedMoEPermuteExpertsUnpermute,
    ):
        super().__init__()
        self.dispatch_combine = dispatch_combine
        self.fused_experts = fused_experts

    def forward(
        self,
        a1: torch.Tensor,  # aka hidden states
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Note: extracting the problem shape from the weight and activation tensors is
        # tricky.  It needs to be done this way specifically due to subtle issues with
        # particular kernels, e.g. the int4 kernels divide the trailing dimension by
        # two, so it's not "correct" to extract N or K from the trailing dimension of
        # w1 or w2.  Similarly, some kernels transpose the weights, so this needs to
        # be kept in mind.
        # TODO: make this a method/utility function, e.g. problem_size(a, w1, w2, topk_ids, ...)
        M, _ = a1.shape
        E, N, _ = w1.shape
        K = w2.shape[1]
        if global_num_experts == -1:
            global_num_experts = E
        top_k = topk_ids.shape[1]

        output = a1 if inplace else torch.empty_like(a1)

        workspace13_shape, workspace2_shape, workspace_dtype = (
            self.fused_experts.workspace_shapes(
                a1.dtype,
                M,
                N,
                K,
                top_k,
                global_num_experts
            )
        )

        # We can reuse the memory between cache1 and cache3 because by the time
        # we need cache3, we're done with cache1
        workspace13 = torch.empty(workspace13_shape,
                                  device=a1.device,
                                  dtype=workspace_dtype)
        workspace2 = torch.empty(workspace2_shape,
                                 device=a1.device,
                                 dtype=workspace_dtype)

        a1q, a1q_scale = self.dispatch_combine.dispatch(
            a1,
            a1_scale,
            a2_scale,
            topk_ids,
            global_num_experts,
            expert_map,
        )

        fused_out = self.fused_experts.apply(
            a1q,
            w1,
            w2,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1q_scale,
            a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
        )

        self.dispatch_combine.combine(output, fused_out, topk_weights, topk_ids)

        return output
