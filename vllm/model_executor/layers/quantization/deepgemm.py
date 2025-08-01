# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging

import torch
from tqdm import tqdm

from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils import direct_register_custom_op
from vllm.utils.deep_gemm import fp8_gemm_nt

logger = logging.getLogger(__name__)

warmup_cache: set[torch.Size] = set()


def warmup_fp8_gemm_nt(w: torch.Tensor, ws: torch.Tensor):
    key = w.size()
    if key in warmup_cache:
        return

    n, k = w.size()
    block_m = 128
    MAX_M = 8192
    MAX_BLOCKS = 8192 // 128

    device = w.device
    a1q = torch.empty((MAX_M, k), device=device).to(torch.float8_e4m3fn)
    a1q_scales = torch.empty((MAX_M, k // block_m),
                             device=device,
                             dtype=torch.float32)
    out = torch.empty((MAX_M, n), device=device, dtype=torch.bfloat16)

    pbar = tqdm(total=MAX_BLOCKS,
                desc=f"DeepGemm(fp8_gemm_nt) warmup (MAX_M={MAX_M})")
    num_tokens = MAX_M
    while num_tokens > 0:
        fp8_gemm_nt((a1q[:num_tokens], a1q_scales[:num_tokens]), (w, ws),
                    out[:num_tokens])
        pbar.update(1)
        num_tokens = num_tokens - block_m

    warmup_cache.add(w.size())


def prepare_block_fp8_matmul_inputs(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> tuple[int, int, int, torch.Tensor]:
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]
    assert A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]

    M = A.numel() // A.shape[-1]

    assert B.ndim == 2
    assert B.is_contiguous()
    assert Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N, )
    C = A.new_empty(C_shape, dtype=output_dtype)

    return M, N, K, C


def w8a8_block_fp8_matmul_deepgemm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    M, N, K, C = prepare_block_fp8_matmul_inputs(A, B, As, Bs, block_size,
                                                 output_dtype)
    # Deepgemm only supports output tensor type as bfloat16
    assert C.dtype == torch.bfloat16
    warmup_fp8_gemm_nt(B, Bs)
    fp8_gemm_nt((A, As), (B, Bs), C)
    return C


def w8a8_block_fp8_matmul_deepgemm_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    M, N, K, C = prepare_block_fp8_matmul_inputs(A, B, As, Bs, block_size,
                                                 output_dtype)
    return C


direct_register_custom_op(
    op_name="w8a8_block_fp8_matmul_deepgemm",
    op_func=w8a8_block_fp8_matmul_deepgemm,
    mutates_args=[],
    fake_impl=w8a8_block_fp8_matmul_deepgemm_fake,
    dispatch_key=current_platform.dispatch_key,
)
