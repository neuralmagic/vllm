# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit-test DeepGEMM FP8 kernels (no DeepEP).
Compare DeepGEMM path against the Triton fallback inside vLLM's fused_experts.
"""

import math

import pytest
import torch

from vllm.config import VllmConfig, get_current_vllm_config
# vLLM fused-expert reference (Triton fallback + DeepGEMM option)
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_mega_moe import (
    DeepGemmMegaExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    make_moe_prepare_and_finalize_no_dp_ep,
)
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.utils.deep_gemm import (
    calc_diff,
    is_deep_gemm_supported,
    per_block_cast_to_fp8,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_dp_group,
)

from .parallel_utils import ProcessGroupInfo, parallel_launch

BLOCK_SIZE = [128, 128]


# activation fp8 x weights fp4
def run_single_case(
    pgi: ProcessGroupInfo,
    m: int,
    n: int,
    k: int,
    topk: int,
    num_experts: int,
    block_size: list[int],
):
    """
    Run one (M,N,K) configuration on a single GPU and assert DeepGEMM ==
    Triton baseline within tolerance.
    """
    import deep_gemm

    router = create_fused_moe_router(topk, num_experts)

    tokens_bf16 = (
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        .clamp_min_(-1)
        .clamp_max_(1)
    )
    _, a1_scale = per_token_group_quant_fp8(tokens_bf16, block_size[1])

    # expert weight tensors
    (w1_bf16, w1, w1_s, w1_gs), (w2_bf16, w2, w2_s, w2_gs) = make_test_weights(
        num_experts,
        n,
        k,
        torch.bfloat16,
        "nvfp4",
        False,
        block_size,
    )

    (dg_w1, dg_w1_s), (dg_w2, dg_w2_s) = deep_gemm.transform_weights_for_mega_moe(
        (w1, w1_s), (w2, w2_s)
    )

    a1_gscale = torch.ones((e,), device="cuda", dtype=torch.float32)
    a2_gscale = torch.ones((e,), device="cuda", dtype=torch.float32)
    a1_scale = a1_gscale
    a2_scale = a2_gscale

    quant_config = FusedMoEQuantConfig.make(
        "nvfp4",
        per_act_token_quant=False,
        block_shape=block_shape,
        w1_scale=dg_w1_s,
        w2_scale=dg_w2_s,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        g1_alphas=(1 / w1_gs) if w1_gs is not None else None,
        g2_alphas=(1 / w2_gs) if w2_gs is not None else None,
    )

    vllm_config = get_current_vllm_config()

    moe_parallel_config = FusedMoEParallelConfig.make(
        tp_size_=get_tensor_model_parallel_world_size(),
        pcp_size_=1,
        dp_size_=get_dp_group().world_size,
        sp_size_=1,
        vllm_parallel_config=vllm_config.parallel_config,
    )

    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=k,
        intermediate_size_per_partition=n, # // dp_size
        num_local_experts=num_experts,  # // dp_size
        num_logical_experts=num_experts,
        moe_parallel_config=moe_parallel_config,
        in_dtype=torch.bfloat16, # or fp8?
        max_num_tokens=next_power_of_2(m),
        activation=MoEActivation.SILU,
        device=vllm_config.device_config.device,
        routing_method=RoutingMethodType.DeepSeekV3,
        moe_parallel_config=moe_parallel_config,
    )

    deep_gemm_experts = mk.FusedMoEKernel(
        prepare_finalize=make_moe_prepare_and_finalize_no_dp_ep(False),
        fused_experts=DeepGemmMegaExperts(
            moe_config=moe_config,
            quant_config=quant_config,
            top_k=topk,
        ),
        inplace=False,
    )

    #############################

    router_logits = torch.randn(m, num_experts, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = router.select_experts(tokens_bf16, router_logits)

    # triton reference
    out_triton = fused_experts(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        quant_config=quant_config,
    )

    # DeepGemm
    out_deepgemm = deep_gemm_experts.apply(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        router_logits=router_logits,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        apply_router_weight_on_input=False,
        expert_map=None,
    )
    diff = calc_diff(out_deepgemm, out_triton)
    assert diff < 0.001, f"Diff exceeded 1%: {diff}"


# Note: N <= 512 will disable the deepgemm path due to performance issues.
MNKs = [
    # (1024, 768, 128),
    # (2048, 768, 512),
    # (512, 1024, 1024),
    # (4096, 4096, 1024),
    (512, 2048, 2048),
]

TOPKS = [2, 6]
NUM_EXPERTS = [32]


@pytest.mark.parametrize(("m", "n", "k"), MNKs)
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("world_dp_size", [(2, 1)])
@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
def test_deep_gemm_mega_moe(
    m,
    n,
    k,
    topk,
    num_experts,
    world_dp_size,
    workspace_init,
):
    set_random_seed(7)

    world_size, dp_size = world_dp_size

    if topk > num_experts:
        pytest.skip(f"topk={topk} > num_experts={num_experts}")

    parallel_launch(
        world_size,
        run_single_case,
        m,
        n,
        k,
        topk,
        num_experts,
        BLOCK_SIZE,
    )
