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
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.utils.deep_gemm import (
    is_deep_gemm_supported,
)

logger = init_logger(__name__)


class DeepGemmMegaExperts(mk.FusedMoEExpertsModular):
    """DeepGemm-based fused MoE expert implementation."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        # router: FusedMoERouter,
        top_k: int,
    ):
        import deep_gemm

        super().__init__(moe_config=moe_config, quant_config=quant_config)
        # assert quant_config.block_shape == get_mk_alignment_for_contiguous_layout()
        # assert (
        #    quant_config.quant_dtype == torch.float8_e4m3fn
        #    or quant_config.quant_dtype == "nvfp4"
        # )
        assert not quant_config.per_act_token_quant
        assert not quant_config.per_out_ch_quant
        # self.router = router

        # Allocate symmetric memory buffer
        # NOTES: requires PyTorch >= 2.9
        self.buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            get_dp_group().device_group,
            moe_config.num_experts,
            moe_config.max_num_tokens,
            top_k,
            moe_config.hidden_dim,  # XXXX for quant
            moe_config.intermediate_size_per_partition,
        )

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
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, K)
        return (workspace1, workspace2, output)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # noqa: B027
        import deep_gemm

        # Transform weights (FP4 with UE8M0 SF) into the required layout
        # TODO: real names
        (
            (layer.w13_weight, layer.w13_weight_scale),
            (layer.w2_weight, layer.w2_weight_scale),
        ) = deep_gemm.transform_weights_for_mega_moe(
            (layer.w13_weight, layer.w13_weight_scale),
            (layer.w2_weight, layer.w2_weight_scale),
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        import deep_gemm

        num_tokens = hidden_states.shape[0]

        from deep_gemm.utils import per_token_cast_to_fp8  # FIX

        assert a1q_scale is None
        hidden_states, a1q_scale = per_token_cast_to_fp8(
            hidden_states, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
        )

        # hidden_states and a1q_scale are already FP8 + UE8M0 packed scales
        # from the prepare_finalize step. Just copy into the symmetric buffer.
        self.buffer.x[:num_tokens].copy_(hidden_states)
        self.buffer.x_sf[:num_tokens].copy_(a1q_scale)
        self.buffer.topk_idx[:num_tokens].copy_(topk_ids)
        self.buffer.topk_weights[:num_tokens].copy_(topk_weights)

        # Run the fused mega MoE kernel
        deep_gemm.fp8_fp4_mega_moe(
            output,
            (w1, self.quant_config.w1_scale),
            (w2, self.quant_config.w2_scale),
            self.buffer,
        )

    def apply_monolithic(
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

        raise AssertionError("never get here")

        #
        # call experts
        #
        topk_weights, topk_ids = self.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

        num_tokens, K = hidden_states.shape

        # Copy inputs into the buffer before each call
        # You may fuse these into previous kernels
        self.buffer.x[:num_tokens].copy_(hidden_states)
        self.buffer.x_sf[:num_tokens].copy_(a1q_scale)
        self.buffer.topk_idx[:num_tokens].copy_(topk_ids)
        self.buffer.topk_weights[:num_tokens].copy_(topk_weights)

        # Run the fused mega MoE kernel
        y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        deep_gemm.fp8_fp4_mega_moe(y, w1, w2, self.buffer)

        return y
