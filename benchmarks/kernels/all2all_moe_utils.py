# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import math

import torch


@dataclasses.dataclass
class QuantArgs:
    quant_dtype: torch.dtype
    per_act_token: bool
    block_size: list[int]


@dataclasses.dataclass
class ModelArgs:
    num_experts: int
    K: int  # hidden-size
    N: int  # first Gemm N
    topk: int  # experts per token
    quant_args: QuantArgs


DEEPSEEK_MODEL_ARGS = ModelArgs(
    num_experts=256,
    K=7168,
    N=2048,
    topk=8,
    quant_args=QuantArgs(
        quant_dtype=torch.float8_e4m3fn, per_act_token=True, block_size=[128, 128]
    ),
)

MODELS = {
    "deepseek-ai/DeepSeek-R1": DEEPSEEK_MODEL_ARGS,
    "deepseek-ai/DeepSeek-V3": DEEPSEEK_MODEL_ARGS,
}


def per_block_cast_to_fp8(
    x: torch.Tensor, block_size_n: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (
            int(math.ceil(m / 128)) * 128,
            int(math.ceil(n / block_size_n)) * block_size_n,
        ),
        dtype=x.dtype,
        device=x.device,
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, block_size_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    x_scaled_sub = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    scales = (x_amax / 448.0).view(x_view.size(0), x_view.size(2))
    return x_scaled_sub, scales


def make_block_quant_fp8_weights(
    e: int,
    n: int,
    k: int,
    block_size: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return weights w1, w2, w1q, w2q, w1_scale, w2_scale
    """
    dtype = torch.bfloat16
    device = torch.cuda.current_device()

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    w1_bf16 = torch.randn((e, 2 * n, k), dtype=dtype, device=device) / 10
    w1_bf16 = w1_bf16.clamp(min=fp8_min, max=fp8_max).to(dtype=dtype)

    w2_bf16 = torch.randn((e, k, n), dtype=dtype, device=device) / 10
    w2_bf16 = w2_bf16.clamp(min=fp8_min, max=fp8_max).to(dtype=dtype)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles_w1 = ((2 * n) + block_n - 1) // block_n
    k_tiles_w1 = (k + block_k - 1) // block_k
    n_tiles_w2 = (k + block_n - 1) // block_n
    k_tiles_w2 = (n + block_k - 1) // block_k

    w1 = torch.empty_like(w1_bf16, dtype=torch.float8_e4m3fn, device=device)
    w2 = torch.empty_like(w2_bf16, dtype=torch.float8_e4m3fn, device=device)

    w1_s = torch.empty((e, n_tiles_w1, k_tiles_w1), device=device, dtype=torch.float32)
    w2_s = torch.empty((e, n_tiles_w2, k_tiles_w2), device=device, dtype=torch.float32)

    assert w1_s.shape == (e, (2 * n + 127) // 128, (k + 127) // 128)
    assert (w2.shape[-2] + block_n - 1) // block_n == w2_s.shape[-2]

    for i in range(e):
        w1[i], w1_s[i] = per_block_cast_to_fp8(w1_bf16[i])
        w2[i], w2_s[i] = per_block_cast_to_fp8(w2_bf16[i])

    return w1, w2, w1_s, w2_s


def make_non_quant_weights(
    e: int,
    n: int,
    k: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return weights w1, w2
    """
    device = torch.cuda.device()
    w1 = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device=device, dtype=dtype) / 10

    return w1, w2
