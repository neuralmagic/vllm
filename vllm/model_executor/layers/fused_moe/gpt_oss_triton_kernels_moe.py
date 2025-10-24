# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_triton_kernels

logger = init_logger(__name__)

if has_triton_kernels():
    try:
        import triton_kernels.swiglu
        from triton_kernels.matmul_ogs import (
            Epilogue,
            FnName,
            FnSpecs,
            FusedActivation,
            GatherIndx,
            RoutingData,
            ScatterIndx,
            matmul_ogs,
        )
        from triton_kernels.matmul_ogs_details.opt_flags import (
            reset_opt_flags_constraints,
            update_opt_flags_constraints,
        )
        from triton_kernels.numerics_details.mxfp import (
            downcast_to_mxfp,
            quantize_mxfp8_fn,
            upcast_from_mxfp,
        )
        from triton_kernels.tensor import (
            BIT,
            Bitmatrix,
            SparseMatrix,
            Tensor,
            make_ragged_tensor_metadata,
        )
        from triton_kernels.topk import topk as triton_topk
    except (AttributeError, ImportError) as e:
        logger.error(
            "Failed to import Triton kernels. Please make sure your triton "
            "version is compatible. Error: %s",
            e,
        )


@triton.jit
def pack_bitmatrix(
    bitmatrix,
    topk_ids,
    n_rows,  # n_rows in bitmatrix / topk_ids
    bm_cols: tl.constexpr,  # n int32_t bitpacks in bitmatrix
    n_expts_act,  # num_topk
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Packs topk_ids into a bitmatrix.
    code reference:
    https://github.com/triton-lang/triton/blob/dd1bbc52b34d202dfe5ffea1e04fb16166c5c04e/python/triton_kernels/bench/distributed.py#L264
    """
    pid_m = tl.program_id(0)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    offsets = offsets_m[:, None] * n_expts_act + offsets_k[None, :]
    mask = (offsets_m < n_rows)[:, None] & (offsets_k < n_expts_act)[None, :]
    indices = tl.load(topk_ids + offsets, mask=mask, other=-1)
    div = indices // 32
    rem = indices % 32
    one = tl.cast(1, tl.uint32)

    # Iterate through all the relevant bitmatrix columns.
    for i in range(bm_cols):
        # When BLOCK_SIZE_K=32, offs is just the column index.
        offs = tl.arange(0, BLOCK_SIZE_K // 32) + i * (BLOCK_SIZE_K // 32)
        # All topks that need to go into this column has the correct bit set.
        # Other bits are 0. x is a 2D tensor.
        x = tl.where(
            div[:, :, None] == offs[None, None, :], (one << rem)[:, :, None], 0
        )
        # Reduce x to get a single int32_t bitpack.
        y = tl.reduce_or(x, axis=1)
        bitmatrix_ptrs = bitmatrix + offsets_m[:, None] * bm_cols + offs[None, :]
        tl.store(bitmatrix_ptrs, y, mask=offsets_m[:, None] < n_rows)


def routing_from_bitmatrix(
    bitmatrix: "Bitmatrix",
    expt_scal: torch.Tensor,
    expt_indx: torch.Tensor,
    n_expts_tot: int,
    n_expts_act: int,
):
    sparse_logits = SparseMatrix(
        indx=expt_indx,
        vals=expt_scal,
        mask=bitmatrix,
    )
    dispatch_index = sparse_logits.mask_metadata.col_sorted_indx
    combine_indx = sparse_logits.mask_metadata.row_sorted_indx
    ragged_batch_metadata = make_ragged_tensor_metadata(
        sparse_logits.mask_metadata.col_sum,
        dispatch_index.shape[0],
    )
    gate_scal = sparse_logits.vals.flatten()[combine_indx]
    routing_data = RoutingData(
        gate_scal,
        ragged_batch_metadata.block_sizes,
        n_expts_tot,
        n_expts_act,
        ragged_batch_metadata,
    )
    gather_idx = GatherIndx(combine_indx, dispatch_index)
    scatter_idx = ScatterIndx(dispatch_index, combine_indx)
    return routing_data, gather_idx, scatter_idx


def routing(
    logits: "torch.Tensor | Tensor",
    n_expts_act: int,
    sm_first: bool = False,
    expt_indx: torch.Tensor | None = None,
    n_rows: torch.Tensor | None = None,
):
    if sm_first:
        logits = torch.softmax(logits, dim=-1)
    sparse_logits = triton_topk(
        logits,
        n_expts_act,
        apply_softmax=not sm_first,
        y_indx=expt_indx,
        n_rows=n_rows,
    )
    return routing_from_bitmatrix(
        sparse_logits.mask,
        sparse_logits.vals,
        sparse_logits.indx,
        logits.shape[-1],
        n_expts_act,
    )


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: str = "silu",
    quant_config: FusedMoEQuantConfig | None = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    routing_data, gather_idx, scatter_idx = routing(
        gating_output, topk, sm_first=not renormalize
    )

    return triton_kernel_fused_experts(
        None,
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_idx,
        scatter_idx,
        activation=activation,
        quant_config=quant_config,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )


# This is a triton implementation of the fused_experts function
def triton_kernel_fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    routing_data,  # RoutingData
    gather_indx,  # GatherIndx
    scatter_indx,  # ScatterIndx
    activation: str = "silu",
    quant_config: FusedMoEQuantConfig | None = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    a1q_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if quant_config is None:
        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

    # type check, uint8 means mxfp4
    assert hidden_states.dtype == torch.bfloat16
    assert quant_config.w1_bias is None or quant_config.w1_bias.dtype == torch.float32
    assert quant_config.w2_bias is None or quant_config.w2_bias.dtype == torch.float32

    # Shape check, only check non-mxfp4
    assert hidden_states.shape[-1] == w1.shape[-2]
    assert w2.shape[-1] == w1.shape[1]

    E, _, N = w1.shape

    if global_num_experts == -1:
        global_num_experts = E

    act = FusedActivation(
        FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")),
        (swiglu_alpha, swiglu_limit),
        2,
    )
    gammas = routing_data.gate_scal if routing_data else None

    def quant_mxfp8(x):
        return downcast_to_mxfp(x, out_quant_type=torch.float8_e4m3fn, axis=1)

    def mxfp8_epilogue():
        # for out mxfp8
        quantize_mxfp8_spec = FnSpecs(
            FnName.QUANTIZE_MXFP8.name, quantize_mxfp8_fn, (), ()
        )
        return Epilogue(quantize_mxfp8_spec, tuple(), tuple(), effective_itemsize=6.0)

    def bf16_bf16_args(x: torch.Tensor, m: int, n: int):
        """
        output x, x_scale, output_shape, output_dtype, out_scale_shape, epilogue
        """
        assert x.dtype == torch.bfloat16
        return x, None, (1, m, n), torch.bfloat16, None, None

    def mxfp8_bf16_args(x: torch.Tensor, m: int, n: int):
        x_fp8, x_scale = quant_mxfp8(x)
        return x_fp8, x_scale, (1, m, n), torch.bfloat16, None, None

    def mxfp8_mxfp8_args(x: torch.Tensor, m: int, n: int):
        x_fp8, x_scale = quant_mxfp8(x)
        return (
            x_fp8,
            x_scale,
            (1, m, n),
            torch.float8_e4m3fn,
            (m, n // 32),
            mxfp8_epilogue(),
        )

    def mm1_output_m():
        return gather_indx.src_indx.shape[0]

    assert hidden_states.dtype == torch.bfloat16
    args_fn = bf16_bf16_args
    if quant_config._a1.dtype == "mxfp8":
        # args_fn = mxfp8_mxfp8_args
        args_fn = mxfp8_bf16_args

    w1_pc = quant_config.w1_precision
    (
        x_q,
        w1_pc.act_scale,
        out_shape,
        w1_pc.out_dtype,
        out_scale_shape,
        epilogue,
    ) = args_fn(hidden_states, mm1_output_m(), w1.size(-1) // 2)

    intermediate_cache1 = torch.zeros(
        out_shape, dtype=w1_pc.out_dtype, device=hidden_states.device
    )

    if out_scale_shape is not None:
        w1_pc.out_scale = torch.ones(
            out_scale_shape, dtype=torch.uint8, device=hidden_states.device
        )

    reset_opt_flags_constraints()
    intermediate_cache1 = matmul_ogs(
        x_q,
        w1,
        quant_config.w1_bias,
        routing_data,
        gather_indx=gather_indx,
        precision_config=w1_pc,
        gammas=gammas if apply_router_weight_on_input else None,
        fused_activation=act,
        epilogue=epilogue,
        y=intermediate_cache1,
    )

    # if intermediate_cache1.dtype == torch.float8_e4m3fn:
    #    intermediate_cache1 = upcast_from_mxfp(
    #        intermediate_cache1, w1_pc.out_scale, target_dtype=torch.bfloat16, axis=1
    #    )

    w2_pc = quant_config.w2_precision
    out_shape = (1, hidden_states.size(0), hidden_states.size(1))
    if quant_config._a2.dtype == "mxfp8":
        # assert intermediate_cache1.dtype == torch.float8_e4m3fn and w1_pc.out_scale is not None
        # w2_pc.act_scale = w1_pc.out_scale
        # w2_pc.out_dtype = torch.bfloat16

        assert intermediate_cache1.dtype == torch.bfloat16
        intermediate_cache1, w2_pc.act_scale = quant_mxfp8(intermediate_cache1)
        w2_pc.out_dtype = torch.bfloat16

    # assert output_tensor is None
    intermediate_cache3 = torch.zeros(
        out_shape, dtype=w2_pc.out_dtype, device=intermediate_cache1.device
    )

    # print (f"hidden states {hidden_states.size()} | intermediate cache 1 {intermediate_cache1.size()} | out scales {w1_pc.out_scale.size()}")

    intermediate_cache3 = matmul_ogs(
        intermediate_cache1,
        w2,
        quant_config.w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=w2_pc,
        gammas=None if apply_router_weight_on_input else gammas,
        y=intermediate_cache3,
    )

    reset_opt_flags_constraints()
    # update_opt_flags_constraints({"block_m": 16, "is_persistent": True, "epilogue_subtile" : None})

    return intermediate_cache3


def make_routing_data(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local_experts: int,
) -> tuple["RoutingData", torch.Tensor, torch.Tensor]:
    topk_ids = topk_ids.to(torch.int16)
    topk_weights = topk_weights.to(torch.bfloat16)

    n_rows, num_topk = topk_ids.size()

    BLOCK_SIZE_M = 512
    BLOCK_SIZE_K = 32

    bm_cols = triton.cdiv(num_local_experts, BLOCK_SIZE_K)  # n_bitpacks
    bitmatrix = torch.zeros(
        (n_rows, bm_cols), dtype=torch.uint32, device=topk_ids.device
    )

    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    pack_bitmatrix[grid](
        bitmatrix,
        topk_ids,
        n_rows,
        bm_cols,
        num_topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    bitmatrix_shape = [n_rows, bm_cols * 32]
    bitmatrix_shape_max = [n_rows, None]
    bitmatrix = Bitmatrix(
        bitmatrix,
        dtype=BIT,
        shape=bitmatrix_shape,
        shape_max=bitmatrix_shape_max,
    )

    # matmul_ogs expects invalid topk_weights to be -1s
    topk_weights = torch.where(topk_ids == -1, -1.0, topk_weights)
    routing_data, gather_indx, scatter_indx = routing_from_bitmatrix(
        bitmatrix, topk_weights, topk_ids, num_local_experts, num_topk
    )

    return routing_data, gather_indx, scatter_indx


class BaseOAITritonExperts(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(self, quant_config: FusedMoEQuantConfig):
        super().__init__(quant_config)

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Weight application and reduction happens in the fused_experts kernel.
        return TopKWeightAndReduceNoOP()

    def _make_routing_data(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_local_experts: int,
    ) -> tuple["RoutingData", torch.Tensor, torch.Tensor]:
        return make_routing_data(topk_ids, topk_weights, num_local_experts)


class OAITritonExperts(BaseOAITritonExperts):
    def __init__(self, quant_config: FusedMoEQuantConfig):
        # TODO (varun) : Enable activation quantization
        assert quant_config.use_mxfp4_w4a16, "Supports only mxfp4_w4a16"
        super().__init__(quant_config)

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.Standard,
            mk.FusedMoEActivationFormat.Standard,
        )

    def supports_chunking(self) -> bool:
        return True

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # workspace are allocated inside the kernel
        workspace1 = (M, K)
        workspace2 = (0, 0)
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if expert_map is not None:
            topk_ids = expert_map[topk_ids]

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        routing_data, gather_indx, scatter_indx = self._make_routing_data(
            topk_ids, topk_weights, local_num_experts
        )

        experts_output = triton_kernel_fused_experts(
            None,
            hidden_states,
            w1,
            w2,
            routing_data,
            gather_indx,
            scatter_indx,
            activation=activation,
            quant_config=self.quant_config,
            apply_router_weight_on_input=False,
            global_num_experts=local_num_experts,
            expert_map=None,  # applied already
            a1q_scale=a1q_scale,
        )

        output.copy_(experts_output, non_blocking=True)
