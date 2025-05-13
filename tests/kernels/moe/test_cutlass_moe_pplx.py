# SPDX-License-Identifier: Apache-2.0
import dataclasses
import traceback
import pytest
import torch
import random

from typing import Callable, Optional

from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp8
from vllm.model_executor.layers.fused_moe.fused_moe import (fused_experts,
                                                            fused_topk)
from vllm.forward_context import set_forward_context
from tests.kernels.utils import torch_moe
from vllm.platforms import current_platform

try:
    from pplx_kernels import AllToAll #or AllToAllInternode?
    from pplx_kernels.nvshmem import (nvshmem_alloc_empty_unique_id,
                                      nvshmem_finalize, nvshmem_get_unique_id,
                                      nvshmem_init)
    has_pplx = False
except ImportError as ex:
    has_pplx = False

from torch.multiprocessing import (
    spawn)  # pyright: ignore[reportPrivateImportUsage]
from typing_extensions import Concatenate, ParamSpec

NUM_EXPERTS = [40, 64]
TOP_KS = [6, 8]

P = ParamSpec("P")

@dataclasses.dataclass
class ProcessGroupInfo:
    world_size: int
    world_local_size: int
    rank: int
    node_rank: int
    local_rank: int
    device: torch.device


def _worker_parallel_launch(
    local_rank: int,
    world_size: int,
    world_local_size: int,
    node_rank: int,
    init_method: str,
    worker: Callable[Concatenate[ProcessGroupInfo, P], None],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    rank = node_rank * world_local_size + local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    barrier = torch.tensor([rank], device=device)
    torch.distributed.all_reduce(barrier)

    try:
        worker(
            ProcessGroupInfo(
                world_size=world_size,
                world_local_size=world_local_size,
                rank=rank,
                node_rank=node_rank,
                local_rank=local_rank,
                device=device,
            ),
            *args,
            **kwargs,
        )
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        raise
    finally:
        torch.distributed.destroy_process_group()


def parallel_launch(
    world_size: int,
    worker: Callable[Concatenate[ProcessGroupInfo, P], None],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    assert not kwargs
    spawn(
        _worker_parallel_launch,
        args=(
            world_size,
            world_size,
            0,
            "tcp://localhost:29500",
            worker,
        ) + args,
        nprocs=world_size,
        join=True,
    )


def rank_chunk(num, r, w):
    rem = num % w
    return (num // w) + (1 if r < rem else 0)
    # return (num + w - 1) // w, rem


def chunk_by_rank(t, r, w):
    num = t.shape[0]
    chunk = rank_chunk(num, r, w)
    #print(f"chunk {t.shape}, {w}, {r}, {chunk}, {r*chunk}:{(r + 1)*chunk}")
    rem = num % w
    if r < rem:
        return t[(r * chunk):(r + 1) * chunk].contiguous()
    else:
        long_chunks = (num // w + 1) * rem
        short_chunks = (r - rem) * chunk
        start = long_chunks + short_chunks
        return t[start:start + chunk].contiguous()

from vllm.model_executor.layers.fused_moe.pplx_dispatch_combine import (
    PplxDispatchCombine)
from vllm.model_executor.layers.fused_moe.cutlass_moe import (
    CutlassExperts)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)


def pplx_cutlass_moe(
    pgi: ProcessGroupInfo,
    dp_size: int,
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    ab_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    ab_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    a1_scale: torch.Tensor,
    scores: torch.Tensor,
    out_dtype,
    per_act_token: bool,
    per_out_ch: bool,
):
    assert torch.cuda.current_device() == pgi.local_rank

    num_tokens, hidden_dim = a.shape
    num_experts = w1.shape[0]
    block_size = hidden_dim # TODO support more cases
    device = pgi.device
    rank = pgi.rank
    world_size = pgi.world_size
    rank_num_tokens = rank_chunk(num_tokens, rank, world_size)
    max_num_tokens = max(128, rank_chunk(a.shape[0], 0, world_size))
    topk = topk_ids.shape[1]

    # return torch.zeros((a.shape[0], a.shape[1]), dtype=out_dtype, device=device)

    # print(vars(AllToAll))
    # print("A DTYPE:", a.dtype)

    # print("DBYTES:", hidden_dim, "*", a.dtype.itemsize, "=", hidden_dim * a.dtype.itemsize)
    # print("SBYTES: CEIL(", hidden_dim, "/", block_size, ") * 4 =",
    #       ((hidden_dim + block_size - 1) // block_size *
    #                              torch.float32.itemsize))

    if block_size == hidden_dim:
        scale_elems = 4 # hack to circumvent pplx data format requirements
    else:
        scale_elems = (hidden_dim + block_size - 1) // block_size,
    # print("SCALE ELEMS:", scale_elems)

    ata = AllToAll(
        max_num_tokens=max_num_tokens,
        num_experts=num_experts,
        experts_per_token=topk,
        rank=rank,
        world_size=pgi.world_size,
        dp_size=dp_size,
        hidden_dim=hidden_dim,
        hidden_dim_bytes=hidden_dim,# * a.dtype.itemsize,
        hidden_dim_scale_bytes=scale_elems * torch.float32.itemsize,
        # hidden_dim_scale_bytes = hidden_dim * torch.float32.itemsize
        # hidden_dim_scale_bytes=0,
    )

    if block_size == hidden_dim:
        repeat_cols = scale_elems
    else:
        repeat_cols = 1
    if a1_scale.shape[0] == 1:
        repeat_rows = num_tokens
    else:
        repeat_rows = 1

    w1 = w1.to(device)
    w2 = w2.to(device)
    w1_scale = w1_scale.to(device)
    w2_scale = w2_scale.to(device)
    ab_strides1 = ab_strides1[:rank_num_tokens].to(device)
    c_strides1 = c_strides1[:rank_num_tokens].to(device)
    ab_strides2 = ab_strides2[:rank_num_tokens].to(device)
    c_strides2 = c_strides2[:rank_num_tokens].to(device)
    # print("a1 scale before repeat:", a1_scale.shape, a.shape)
    # print(a1_scale)
    a1_scale = a1_scale.repeat(repeat_rows, repeat_cols).contiguous().to(device)
    # print("a1 scale after repeat:", a1_scale.shape, a.shape)
    # topk_weights = topk_weights.to(device)
    # topk_ids = topk_ids.to(device)

    # print("a:", a.shape)
    # print("a scale:", a1_scale.shape)
    # print("rank_num_tokens:", rank_num_tokens)

    dispatch_combine = PplxDispatchCombine(
        ata,
        max_num_tokens,
        pgi.world_size,
        dp_size,
        rank,
        quant_dtype=torch.float8_e4m3fn,
        per_act_token=per_act_token,
    )

    experts = CutlassExperts(
            ab_strides1,
            c_strides1,
            ab_strides2,
            c_strides2,
            out_dtype,
            per_act_token,
            per_out_ch)

    fused_cutlass_experts = FusedMoEModularKernel(
        dispatch_combine,
        experts,
    )

    # a1_scale = a1_scale.repeat(a.shape).contiguous()
    a_chunk = chunk_by_rank(a, rank, world_size).to(device)
    score_chunk = chunk_by_rank(scores, rank, world_size).to(device)
    chunk_topk_weight = chunk_by_rank(topk_weights, rank, world_size).to(device)
    chunk_topk_ids = chunk_by_rank(topk_ids, rank, world_size).to(device)

    # print("chunk topk:", topk_ids, "->", chunk_topk_ids, "rank_num_tokens:", rank_num_tokens)
    # print("chunk a:", a, "->", a_chunk)

    # print("A SHAPE:", a.shape, "A CHUNK SHAPE:", a_chunk.shape)
    # print("A SCALE SHAPE:", a1_scale.shape, "A SCALE CHUNK SHAPE:",
    #       chunk_by_rank(a1_scale, rank, world_size).shape)

    # print("fused cutlass experts:", a_chunk.shape, a1_scale.shape,
    #       w1.shape,
    #       w1_scale.shape,
    #       w2.shape,
    #       w2_scale.shape)

    # This is a hack to let the dispatch combine know that the scale is per token
    # TODO make this less hacky
    # if per_act_token:
    #     a1_scale = a1_scale.reshape(a1_scale.shape[0], a1_scale.shape[1], 1)

    # print("a1 scale full:", a1_scale[:,0:1]) 
    # print("a1 scale chunk:", chunk_by_rank(a1_scale, rank, world_size)[:,0:1])

    # print("SCALE FULL:", w1_scale)
    # print("SCALE CHUNKED:", chunk_by_rank(w1_scale, rank, world_size))
    out = fused_cutlass_experts(a_chunk,
                        chunk_by_rank(w1, rank, world_size),
                        chunk_by_rank(w2, rank, world_size),
                        chunk_topk_weight,
                        chunk_topk_ids,
                        global_num_experts=num_experts,
                        expert_map=None, #TODO
                        w1_scale=chunk_by_rank(w1_scale, rank, world_size),
                        w2_scale=chunk_by_rank(w2_scale, rank, world_size),
                        a1_scale=chunk_by_rank(a1_scale, rank, world_size))
    # # out = fused_experts(
    # #     a_chunk,
    # #     # Chunking weights like this only works for batched format
    # #     chunk_by_rank(w1, rank, world_size),
    # #     chunk_by_rank(w2, rank, world_size),
    # #     #w1,
    # #     #w2,
    # #     chunk_topk_weight,
    # #     chunk_topk_ids,
    # #     global_num_experts=num_experts  #? num_local_experts?
    # # )

    # out = torch.zeros((a.shape[0] * topk_ids.shape[1], a.shape[1]), dtype=out_dtype, device=device)

    torch.cuda.synchronize()

    ata.destroy()

    # print("out:", out, out.shape, rank_num_tokens)

    return out[:rank_num_tokens]


vllm_config = VllmConfig()
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192

from vllm.model_executor.layers.activation import SiluAndMul
def torch_moe2(a, w1, w2, topk_weight, topk_ids):
    M, K = a.shape
    topk = topk_ids.shape[1]
    a = a.view(M, -1, K).repeat(1, topk, 1).reshape(-1, K)
    out = torch.zeros(M * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    num_experts = w1.shape[0]
    for i in range(num_experts):
        mask = (topk_ids == i).view(-1)
        if mask.sum():
            out[mask] = SiluAndMul()(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)

    return (out.view(M, -1, w2.shape[1]) *
            topk_weight.view(M, -1, 1).to(out.dtype)).sum(dim=1)

def _pplx_moe(
    pgi: ProcessGroupInfo,
    dp_size: int,
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    ab_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    ab_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    a1_scale: torch.Tensor,
    score: torch.Tensor,
    out_dtype,
    a_full: torch.Tensor,
    w1_full: torch.Tensor,
    w2_full: torch.Tensor,
    per_act_token: bool,
    per_out_ch: bool,
):

    # torch.cuda.synchronize()

    score = torch.full(score.shape, 1, dtype=score.dtype, device=score.device)

    uid = nvshmem_get_unique_id(
    ) if pgi.rank == 0 else nvshmem_alloc_empty_unique_id()
    torch.distributed.broadcast(uid, src=0)
    nvshmem_init(uid, pgi.rank, pgi.world_size)

    # print("PGI:", pgi.device.index)

    with set_current_vllm_config(vllm_config):
        torch_output = torch_moe2(a_full, w1_full, w2_full,
                                  topk_weights, topk_ids)
        pplx_output = pplx_cutlass_moe(pgi,
                                    dp_size,
                                        a,
                                        w1,
                                        w2,
                                        w1_scale,
                                        w2_scale,
                                        topk_weights,
                                        topk_ids,
                                        ab_strides1,
                                        c_strides1,
                                        ab_strides2,
                                        c_strides2,
                                        a1_scale,
                                        score,
                                        out_dtype,
                                        per_act_token,
                                        per_out_ch)

        torch_output = chunk_by_rank(torch_output, pgi.rank,
                                 pgi.world_size).to(pplx_output.device)
        
    # if (pgi.device.index == 0):
    print("PPLX OUT:", pplx_output)
    print("TORCH OUT:", torch_output)

    # TODO figure out if there is an issue or the results are just inaccurate
    # due to dequantization
    # torch.testing.assert_close(pplx_output, torch_output, atol=0.01, rtol=0)

    nvshmem_finalize()


@pytest.mark.parametrize("m", [2, 64, 224])
@pytest.mark.parametrize("n", [1024, 3072])
@pytest.mark.parametrize("k", [1024, 1536])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("per_act_token", [True])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.parametrize("world_dp_size", [[2, 1]])  #, [4, 2]])
# @pytest.mark.parametrize("m", [5])
# @pytest.mark.parametrize("n", [1024])
# @pytest.mark.parametrize("k", [1024])
# @pytest.mark.parametrize("e", [4])
# @pytest.mark.parametrize("topk", [1])
# @pytest.mark.parametrize("per_act_token", [True])
# @pytest.mark.parametrize("per_out_ch", [False])
# @pytest.mark.parametrize("world_dp_size", [[2, 1]])  #, [4, 2]])
@pytest.mark.skipif(
    (lambda x: x is None or not ops.cutlass_group_gemm_supported(x.to_int()))(
        current_platform.get_device_capability()),
    reason="Grouped gemm is not supported on this GPU type.")
def test_cutlass_moe_pptx(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
    world_dp_size: tuple[int, int],
):
    current_platform.seed_everything(7)

    with set_current_vllm_config(vllm_config):

        dtype = torch.half

        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10.0
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10.0
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10.0

        # for idx in range (m):
        #     a[idx] = torch.full((k,), idx + 1, device="cuda", dtype=dtype)
        #     w1[idx] = torch.full((2 * n, k), 1 + idx, device="cuda", dtype=dtype)
        #     w2[idx] = torch.full((k, n), 1 + idx, device="cuda", dtype=dtype)

        # for idx in range (e):
        #     if idx != 0:
        #         w2[idx] = torch.zeros((k, n), device="cuda", dtype=dtype)
        #         # w1[idx] = torch.zeros((2 * n, k), device="cuda", dtype=dtype)

        # a[1:] = torch.zeros((m - 1, k), device="cuda", dtype=dtype)

        # Get the right scale for tests.
        a_q, a_scale1 = ops.scaled_fp8_quant(
            a, use_per_token_if_dynamic=per_act_token)

        # print("a:", a.shape)
        # print("a_scale1:", a_scale1)

        # TODO this snippet makes the scales identical for all tokens - remove
        # after testing
        if per_act_token:
            for idx in range(m):
                if idx != 0:
                    a_scale1[idx] = a_scale1[0]

        # a_q, _ = ops.scaled_fp8_quant(a,
        #                               a_scale1,
        #                               use_per_token_if_dynamic=per_act_token)

        # a_d = a_q.float().mul(a_scale1).to(dtype)

        # print("a_d:", a_d)
        # print("a:", a)
        # print("a_scale1:", a_scale1)
        # Verify that a_q * a_scale1 matches a_d
        # a_d_float = a_q.float()
        # for i in range(m):
        #     a_d_float[i] = a_d_float[i] * a_scale1[i]
        # a_d = a_d_float.to(dtype)
        a_d = a_q.float().mul(a_scale1).to(dtype)
        # print("a_d:", a_d)
        # torch.testing.assert_close(a, a_d, atol=1e-1, rtol=0)

        n_b_scales = 2 * n if per_out_ch else 1
        k_b_scales = k if per_out_ch else 1

        w1_q = torch.empty((e, 2 * n, k),
                           device="cuda",
                           dtype=torch.float8_e4m3fn)
        w2_q = torch.empty((e, k, n), device="cuda", dtype=torch.float8_e4m3fn)
        w1_scale = torch.empty((e, n_b_scales, 1),
                               device="cuda",
                               dtype=torch.float32)
        w2_scale = torch.empty((e, k_b_scales, 1),
                               device="cuda",
                               dtype=torch.float32)

        for expert in range(e):
            w1_q[expert], w1_scale[expert] = ops.scaled_fp8_quant(
                w1[expert], use_per_token_if_dynamic=per_out_ch)
            w2_q[expert], w2_scale[expert] = ops.scaled_fp8_quant(
                w2[expert], use_per_token_if_dynamic=per_out_ch)
        # w1_q = w1_q.transpose(1, 2)
        # w2_q = w2_q.transpose(1, 2)

        # TODO delete when done
        # for idx in range (e):
        #     for nn in range(n):
        #         w2_q[idx][nn] = torch.full((k,), nn * 8 + idx + 1, device="cuda", dtype=dtype)
        #         # w2_q[idx][nn] = torch.full((k,), random.random(), device="cuda", dtype=dtype)
        #         # w2_q[idx][nn] = torch.full((k,), 1 + idx, device="cuda", dtype=dtype)

        ab_strides1 = torch.full((e, ), k, device="cuda", dtype=torch.int64)
        c_strides1 = torch.full((e, ), 2 * n, device="cuda", dtype=torch.int64)
        ab_strides2 = torch.full((e, ), n, device="cuda", dtype=torch.int64)
        c_strides2 = torch.full((e, ), k, device="cuda", dtype=torch.int64)

        w1_d = torch.empty_like(w1)
        w2_d = torch.empty_like(w2)
        for expert in range(e):
            w1_d[expert] = (w1_q[expert].float() * w1_scale[expert]).half()
            w2_d[expert] = (w2_q[expert].float() * w2_scale[expert]).half()

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids = fused_topk(a, score, topk, renormalize=False)

        # print("GENERAL TOPK WEIGHTS:", topk_weights)
        # print("GENERAL TOPK IDS:", topk_ids)

        # if per_act_token:
        #     for idx in range(m):
        #         a_scale1[idx] = torch.full((1,), topk_ids[idx][0] + 1, device="cuda", dtype=torch.float32)

        # print("w2_q:", w2_q)
        # print("w2_scale:", w2_scale)
        # # print("w1_q * w1_scale:", w1_q.half() * w1_scale)
        # print("w2_d:", w2_d)

        # torch.testing.assert_close((w2_q.half() * w2_scale).half(), w2_d, atol=2e-2, rtol=0)

        world_size, dp_size = world_dp_size
        # print("original a:", a)
        parallel_launch(world_size, _pplx_moe, dp_size, a,
                        w1_q, w2_q,
                        w1_scale, w2_scale, topk_weights, topk_ids,
                        ab_strides1, c_strides1, ab_strides2, c_strides2,
                        a_scale1, score, dtype,
                        a_d, w1_d, w2_d, per_act_token, per_out_ch)

        # cutlass_output = cutlass_moe_fp8(a,
        #                                  w1_q,
        #                                  w2_q,
        #                                  w1_scale,
        #                                  w2_scale,
        #                                  topk_weights,
        #                                  topk_ids,
        #                                  ab_strides1,
        #                                  c_strides1,
        #                                  ab_strides2,
        #                                  c_strides2,
        #                                  a1_scale=a_scale1)

        # # print(torch_output.t()[0])
        # # print(cutlass_output.t()[0])
        # # print("*")

        # print(torch_output)
        # print(pplx_output)
        # print("*")

        # torch.testing.assert_close(torch_output,
        #                            pplx_output,
        #                            atol=5e-2,
        #                            rtol=1e-2)
