# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import (
    get_dp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.utils.deep_gemm import (
    get_mk_alignment_for_contiguous_layout,
    is_deep_gemm_supported,
)

logger = init_logger(__name__)


class DeepGemmMegaExperts(mk.FusedMoEExpertsMonolithic):
    """DeepGemm-based fused MoE expert implementation."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        router: FusedMoERouter,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.block_shape == get_mk_alignment_for_contiguous_layout()
        assert (
            quant_config.quant_dtype == torch.float8_e4m3fn
            or quant_config.quant_dtype == "nvfp4"
        )
        assert not quant_config.per_act_token_quant
        assert not quant_config.per_out_ch_quant
        self.router = router

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return is_deep_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.SWIGLUSTEP]  # ???????

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # NOTE(rob): discovered an IMA with this combination. Needs investigation.
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        assert self.block_shape is not None
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        import deep_gemm

        #
        # call experts
        #
        topk_weights, topk_ids = self.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        top_k = topk_ids.size(0)

        # Allocate symmetric memory buffer
        # NOTES: requires PyTorch >= 2.9
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            get_dp_group(),
            self.moe_config.num_experts,
            self.moe_config.max_num_tokens,
            top_k,
            self.moe_config.hidden_dim,
            self.moe_config.intermediate_size_per_partition,
        )

        # Transform weights (FP4 with UE8M0 SF) into the required layout
        transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
            w1, w2
        )

        num_tokens, K = hidden_states.shape

        x, x_scales = moe_kernel_quantize_input(
            hidden_states,
            a1q_scale,
            self.quant_config.quant_dtype,
            self.quant_config.per_act_token_quant,
            self.quant_config.block_shape,
        )

        # Copy inputs into the buffer before each call
        # You may fuse these into previous kernels
        buffer.x[:num_tokens].copy_(x)
        buffer.x_sf[:num_tokens].copy_(x_scales)
        buffer.topk_idx[:num_tokens].copy_(topk_ids)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)

        # Run the fused mega MoE kernel
        y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        deep_gemm.fp8_fp4_mega_moe(y, transformed_l1, transformed_l2, buffer)

        return y
