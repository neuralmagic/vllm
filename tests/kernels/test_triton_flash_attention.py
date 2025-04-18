# SPDX-License-Identifier: Apache-2.0
"""Tests for the triton_flash_attention kernel

Run `pytest tests/kernels/test_triton_flash_attention.py`.
"""
import pytest
import torch

import vllm._custom_ops as ops
# yapf: disable
from vllm.attention.ops.triton_flash_attention import (SUPPORTED_LAYOUTS,
                                                       MetaData,
                                                       compute_alibi_tensor,
                                                       scale_fp8,
                                                       triton_attention,
                                                       triton_attention_rocm)
from vllm.platforms import current_platform

# yapf: enable


class ReferenceAttention:

    def __init__(self, Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, use_alibi, dtype,
                 input_metadata):
        self.Z = Z
        self.HQ = HQ
        self.HK = HK
        self.N_CTX_Q = N_CTX_Q
        self.N_CTX_K = N_CTX_K
        self.D_HEAD = D_HEAD
        self.use_alibi = use_alibi
        self.dtype = dtype
        self.input_metadata = input_metadata

    def fwd(self, q, k, v):
        scores = torch.einsum('bhqd,bhkd->bhqk', q,
                              k).float() * self.input_metadata.sm_scale
        if self.input_metadata.causal:
            mask = torch.tril(torch.ones(self.N_CTX_Q,
                                         self.N_CTX_K,
                                         device="cuda"),
                              diagonal=self.N_CTX_K - self.N_CTX_Q)
            scores[:, :, mask == 0] = float("-inf")

        if self.input_metadata.bias is not None:
            scores += self.input_metadata.bias

        if self.use_alibi:
            scores += compute_alibi_tensor(self.input_metadata.alibi_slopes,
                                           self.N_CTX_Q, self.N_CTX_K)

        p = torch.softmax(scores, dim=-1)
        if self.input_metadata.causal:
            # If N_CTX_Q > N_CTX_K, there's at least one row of all -infs going
            # into softmax. This creates a row of NaNs as -inf - -inf == NaN.
            # So we fix this by converting the NaNs to 0s, which is what they
            # should be out of the softmax.
            nan_mask = torch.isnan(p)
            p[nan_mask == 1] = 0
        ref_out = torch.einsum('bhqk,bhkd->bhqd', p.to(self.dtype), v)
        # compare
        if self.input_metadata.layout == 'bshd':
            ref_out = ref_out.transpose(1, 2).clone()
        return ref_out

    def fwd_fp8(self, q_quantized, k_quantized, v_quantized):
        q = (q_quantized.to(torch.float16) * self.input_metadata.q_descale).to(
            self.dtype)
        k = (k_quantized.to(torch.float16) * self.input_metadata.k_descale).to(
            self.dtype)
        v = (v_quantized.to(torch.float16) * self.input_metadata.v_descale).to(
            self.dtype)
        result = self.fwd(q, k, v)
        if self.input_metadata.o_scale is not None:
            result, _ = scale_fp8(result, self.input_metadata.o_scale)
        return result

    def fwd_fp8_kv(self, q, k_quantized, v_quantized):
        k_descale, v_descale = (self.input_metadata.k_descale,
                                self.input_metadata.v_descale)
        k_dequantized = (k_quantized.to(torch.float32) *
                         k_descale.to(torch.float32)).to(self.dtype)
        v_dequantized = (v_quantized.to(torch.float32) *
                         v_descale.to(torch.float32)).to(self.dtype)
        return self.fwd(q, k_dequantized, v_dequantized)

    def varlen_fwd(self, q, k, v, is_mqa=False):
        ref_out = torch.empty_like(q)
        if is_mqa:
            # Make KV look like HQ/HK "groups" of HK. Later, we will reshape so
            # the size aligns with Q.
            k_ref = k.view(k.shape[0], k.shape[1], 1,
                           k.shape[2]).expand(-1, -1, self.HQ // self.HK, -1)
            v_ref = v.view(v.shape[0], v.shape[1], 1,
                           v.shape[2]).expand(-1, -1, self.HQ // self.HK, -1)
        else:
            k_ref = k
            v_ref = v

        for i in range(0, self.input_metadata.num_contexts):
            start_q, start_k = self.input_metadata.cu_seqlens_q[
                i], self.input_metadata.cu_seqlens_k[i]
            end_q, end_k = self.input_metadata.cu_seqlens_q[
                i + 1], self.input_metadata.cu_seqlens_k[i + 1]
            k_curr = k_ref[start_k:end_k]
            v_curr = v_ref[start_k:end_k]
            if is_mqa:
                k_curr = k_curr.reshape(k_curr.shape[0], -1, k_curr.shape[3])
                v_curr = v_curr.reshape(v_curr.shape[0], -1, v_curr.shape[3])
            scores = torch.einsum('qhd,khd->qhk', q[start_q:end_q],
                                  k_curr).float()
            p = torch.softmax(scores * self.input_metadata.sm_scale,
                              dim=-1).half()
            ref_out[start_q:end_q] = torch.einsum('qhk,khd->qhd', p, v_curr)
        return ref_out


def quantize_input(q,
                   k,
                   v,
                   input_metadata: MetaData,
                   fp8_kv=False,
                   use_o_scale=False):
    is_supported_layout = input_metadata.layout in SUPPORTED_LAYOUTS
    assert is_supported_layout, "Got unsupported layout."

    q_descale = None
    if not fp8_kv:
        q, q_descale = scale_fp8(q)
    k, k_descale = scale_fp8(k)
    v, v_descale = scale_fp8(v)

    # In real world use case, the p scale would be a parameter trained by the
    # model.
    p_scale = p_descale = None

    o_scale = torch.rand(1, device="cuda") if use_o_scale else None

    # We are not multiplying the scales together to get
    # qk_desale / o_descale e.g.
    # qk_desale = q_descale * k_descale
    # o_desale = p_descale * v_descale
    # it results in very small fp e.g. 0,0002, losing precision.
    # They are applied on the run.
    input_metadata.set_eight_bit_params(
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        # By default p_scaling is not enabled
        p_scale=p_scale,
        p_descale=p_descale,
        o_scale=o_scale)

    return q, k, v


def input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout):
    assert layout in SUPPORTED_LAYOUTS, "Got unsupported layout."

    current_platform.seed_everything(0)

    # Initialize q, k, v
    if layout == 'bhsd':
        q_tensor_shape = (Z, HQ, N_CTX_Q, D_HEAD)
        k_tensor_shape = (Z, HK, N_CTX_K, D_HEAD)
    elif layout == 'bshd':
        q_tensor_shape = (Z, N_CTX_Q, HQ, D_HEAD)
        k_tensor_shape = (Z, N_CTX_K, HK, D_HEAD)
    q = torch.randn(q_tensor_shape,
                    dtype=dtype,
                    device="cuda",
                    requires_grad=False)
    k = torch.randn(k_tensor_shape,
                    dtype=dtype,
                    device="cuda",
                    requires_grad=False)
    v = torch.randn(k_tensor_shape,
                    dtype=dtype,
                    device="cuda",
                    requires_grad=False)

    sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = N_CTX_Q
    input_metadata.max_seqlens_k = N_CTX_K
    input_metadata.layout = layout
    return q, k, v, input_metadata


def varlen_input_helper(Z,
                        HQ,
                        HK,
                        N_CTX_Q,
                        N_CTX_K,
                        D_HEAD,
                        dtype,
                        equal_seqlens=False):
    current_platform.seed_everything(0)

    # Random sequence lengths. Using N_CTX as kind of max of sum of individual
    # seqs
    if not equal_seqlens:
        max_seqlens_q = N_CTX_Q // Z
        max_seqlens_k = N_CTX_K // Z
        seqlens_q = torch.randint(1,
                                  max_seqlens_q + 1, (Z, ),
                                  dtype=torch.int32)
        seqlens_k = torch.randint(1,
                                  max_seqlens_k + 1, (Z, ),
                                  dtype=torch.int32)
    else:
        seqlens_q = torch.full((Z, ), N_CTX_Q // Z)
        seqlens_k = torch.full((Z, ), N_CTX_K // Z)

    # Calculate cumulative sequence lengths
    cu_seqlens_q = torch.cat([
        torch.tensor([0], dtype=torch.int32),
        seqlens_q.cumsum(dim=0, dtype=torch.int32)
    ])
    cu_seqlens_k = torch.cat([
        torch.tensor([0], dtype=torch.int32),
        seqlens_k.cumsum(dim=0, dtype=torch.int32)
    ])
    cu_seqlens_q = cu_seqlens_q.to(device="cuda")
    cu_seqlens_k = cu_seqlens_k.to(device="cuda")

    # Initialize q, k, v with variable lengths
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()
    q = torch.randn((total_q, HQ, D_HEAD), dtype=dtype,
                    device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.randn((total_k, HK, D_HEAD), dtype=dtype,
                    device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.randn((total_k, HK, D_HEAD), dtype=dtype,
                    device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)
    return q, k, v, input_metadata


@pytest.mark.parametrize('Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 12, 1, 1, 64),
    (4, 48, 48, 1, 1, 128),
    (4, 48, 24, 3, 3, 128),
    (4, 4, 4, 128, 128, 65),
    (4, 4, 4, 113, 123, 1),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('use_alibi', [True, False])
@pytest.mark.parametrize('layout', ['bshd', 'bhsd'])
def test_op_fwd(Z,
                HQ,
                HK,
                N_CTX_Q,
                N_CTX_K,
                D_HEAD,
                causal,
                use_alibi,
                layout,
                dtype=torch.float16):
    current_platform.seed_everything(0)
    q, k, v, input_metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD,
                                           dtype, layout)
    input_metadata.eight_bit_dtype_torch = current_platform.fp8_dtype()
    if causal:
        input_metadata.need_causal()

    if use_alibi:
        # for n heads the set of slopes is the geometric sequence that starts
        # 2^(-8/n)
        alibi_slopes = torch.tensor(
            [2**(-8 / HQ * i) for i in range(1, HQ + 1)],
            dtype=torch.float32,
            device="cuda").repeat(Z, 1)
        input_metadata.need_alibi(alibi_slopes, Z, HQ)
    else:
        alibi_slopes = None

    o = torch.empty_like(q)

    # triton implementation
    tri_out, _ = triton_attention_rocm(q, k, v, o, input_metadata)

    # Transpose here if layout is bshd so we have same reference code for all
    # layouts
    if layout == 'bshd':
        q = q.transpose(1, 2).clone()
        k = k.transpose(1, 2).clone()
        v = v.transpose(1, 2).clone()
    # Replicate K and V if using MQA/GQA
    if HQ != HK:
        k = k.view(k.shape[0], k.shape[1], -1, k.shape[2],
                   k.shape[3]).expand(-1, -1, HQ // HK, -1,
                                      -1).reshape(k.shape[0], -1, k.shape[2],
                                                  k.shape[3])
        v = v.view(v.shape[0], v.shape[1], -1, v.shape[2],
                   v.shape[3]).expand(-1, -1, HQ // HK, -1,
                                      -1).reshape(v.shape[0], -1, v.shape[2],
                                                  v.shape[3])

    ref_impl = ReferenceAttention(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD,
                                  use_alibi, dtype, input_metadata)
    ref_out = ref_impl.fwd(q, k, v)

    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 12, 1, 1, 64),
    (4, 48, 48, 1, 1, 128),
    (4, 48, 24, 3, 3, 128),
    (4, 4, 4, 128, 128, 65),
    (4, 4, 4, 113, 123, 1),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('use_alibi', [True, False])
@pytest.mark.parametrize('layout', ['bshd', 'bhsd'])
@pytest.mark.parametrize('persistent', ['fixed', 'dynamic'])
def test_op_persistent_fwd(Z,
                           HQ,
                           HK,
                           N_CTX_Q,
                           N_CTX_K,
                           D_HEAD,
                           causal,
                           use_alibi,
                           layout,
                           persistent,
                           dtype=torch.float16):
    current_platform.seed_everything(0)
    q, k, v, input_metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD,
                                           dtype, layout)
    input_metadata.eight_bit_dtype_torch = current_platform.fp8_dtype()
    if causal:
        input_metadata.need_causal()

    if use_alibi:
        # for n heads the set of slopes is the geometric sequence that starts
        # 2^(-8/n)
        alibi_slopes = torch.tensor(
            [2**(-8 / HQ * i) for i in range(1, HQ + 1)],
            dtype=torch.float32,
            device="cuda").repeat(Z, 1)
        input_metadata.need_alibi(alibi_slopes, Z, HQ)
    else:
        alibi_slopes = None

    input_metadata.set_persistent(persistent)

    o = torch.empty_like(q)

    # triton implementation
    tri_out, _ = triton_attention_rocm(q, k, v, o, input_metadata)

    # Transpose here if layout is bshd so we have same reference code for all
    # layouts
    if layout == 'bshd':
        q = q.transpose(1, 2).clone()
        k = k.transpose(1, 2).clone()
        v = v.transpose(1, 2).clone()
    # Replicate K and V if using MQA/GQA
    if HQ != HK:
        k = k.view(k.shape[0], k.shape[1], -1, k.shape[2],
                   k.shape[3]).expand(-1, -1, HQ // HK, -1,
                                      -1).reshape(k.shape[0], -1, k.shape[2],
                                                  k.shape[3])
        v = v.view(v.shape[0], v.shape[1], -1, v.shape[2],
                   v.shape[3]).expand(-1, -1, HQ // HK, -1,
                                      -1).reshape(v.shape[0], -1, v.shape[2],
                                                  v.shape[3])

    ref_impl = ReferenceAttention(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD,
                                  use_alibi, dtype, input_metadata)
    ref_out = ref_impl.fwd(q, k, v)

    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 1, 1, 64),
    (4, 48, 1, 1, 128),
    (4, 48, 3, 3, 128),
    (4, 4, 128, 128, 65),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('layout', ['bhsd'])
@pytest.mark.parametrize('use_o_scale', [True, False])
def test_op_fwd_fp8(Z,
                    H,
                    N_CTX_Q,
                    N_CTX_K,
                    D_HEAD,
                    causal,
                    layout,
                    use_o_scale,
                    dtype=torch.float32):
    current_platform.seed_everything(0)

    # Disable grad to save memory it won't run into OOM on CI machine.
    q, k, v, input_metadata = input_helper(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD,
                                           dtype, layout)
    input_metadata.eight_bit_dtype_torch = current_platform.fp8_dtype()
    if causal:
        input_metadata.need_causal()

    o = torch.empty_like(q) if not use_o_scale else None

    q_quantized, k_quantized, v_quantized = quantize_input(
        q, k, v, input_metadata, use_o_scale=use_o_scale)

    tri_out, _ = triton_attention_rocm(q_quantized, k_quantized, v_quantized,
                                       o, input_metadata)

    ref_impl = ReferenceAttention(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD, False,
                                  dtype, input_metadata)
    ref_out = ref_impl.fwd_fp8(q_quantized, k_quantized, v_quantized)

    # compare
    torch.testing.assert_close(ref_out.to(torch.float32),
                               tri_out.to(torch.float32),
                               atol=2e-2,
                               rtol=2e-2)


# TODO(luka): this test is supposed to replicate the arguments
#  as they occur after the attention fusion pass.
#  But sadly it fails with a maxdiff.
@pytest.mark.parametrize("fp8_kvcache", [True])
@pytest.mark.parametrize("N", [
    1,
])  # [256] for CUDAGraph
@pytest.mark.skip()
def test_fused_fwd(fp8_kvcache: bool, N: int):
    current_platform.seed_everything(0)
    torch.set_default_device("cuda")

    MAX_SEQLENS = 64
    q = torch.rand(MAX_SEQLENS * N, 32, 128, dtype=torch.bfloat16)
    k = torch.rand(MAX_SEQLENS * N, 8, 128, dtype=torch.bfloat16)
    v = torch.rand(MAX_SEQLENS * N, 8, 128, dtype=torch.bfloat16)

    cu_seqlens = torch.arange(0, N + 1, dtype=torch.int32) * MAX_SEQLENS

    # fp8_scales = tuple(torch.rand(1) for _ in range(4)) if fp8_kvcache else None # noqa: E501
    fp8_scales = (torch.tensor([1.0]), torch.tensor([0.0508]),
                  torch.tensor([0.0508]),
                  torch.tensor([1.0])) if fp8_kvcache else None
    # input_scale = torch.tensor([0.0019])
    input_scale = torch.tensor([0.0019])

    FP8_DTYPE = current_platform.fp8_dtype()
    oq_fused = torch.empty(MAX_SEQLENS * N, 32, 128, dtype=FP8_DTYPE)
    o_unfused = torch.empty(MAX_SEQLENS * N, 32, 128, dtype=torch.bfloat16)

    call_attn = lambda o, scale: triton_attention(q,
                                                  k,
                                                  v,
                                                  o,
                                                  cu_seqlens,
                                                  cu_seqlens,
                                                  MAX_SEQLENS,
                                                  MAX_SEQLENS,
                                                  causal=True,
                                                  sm_scale=0.08838,
                                                  bias=None,
                                                  fp8_scales=fp8_scales,
                                                  input_scale=scale)

    call_attn(oq_fused, input_scale)

    call_attn(o_unfused, None)
    oq_unfused, _ = ops.scaled_fp8_quant(o_unfused.view(-1, 4096), input_scale)

    out_fused = oq_fused.view(-1, 4096).to(torch.float32)
    out_unfused = oq_unfused.to(torch.float32)

    torch.testing.assert_close(out_fused, out_unfused, atol=1e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 1, 1, 64),
    (4, 48, 1, 1, 128),
    (4, 48, 3, 3, 128),
    (4, 4, 128, 128, 65),
    (4, 4, 113, 123, 1),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('layout', ['bhsd'])
def test_op_fwd_fp8_kv(Z,
                       H,
                       N_CTX_Q,
                       N_CTX_K,
                       D_HEAD,
                       causal,
                       layout,
                       dtype=torch.float32):
    current_platform.seed_everything(0)

    q, k, v, input_metadata = input_helper(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD,
                                           dtype, layout)
    input_metadata.eight_bit_dtype_torch = current_platform.fp8_dtype()
    if causal:
        input_metadata.need_causal()

    o = torch.empty_like(q)

    _, k_quantized, v_quantized = quantize_input(q,
                                                 k,
                                                 v,
                                                 input_metadata,
                                                 fp8_kv=True)

    tri_out, _ = triton_attention_rocm(q, k_quantized, v_quantized, o,
                                       input_metadata)

    ref_impl = ReferenceAttention(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD, False,
                                  dtype, input_metadata)
    ref_out = ref_impl.fwd_fp8_kv(q, k_quantized, v_quantized)

    torch.testing.assert_close(ref_out, tri_out, atol=3e-2, rtol=8e-1)


@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 1, 1, 64),
    (4, 48, 1, 1, 128),
    (4, 48, 3, 3, 128),
    (4, 4, 128, 128, 65),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('use_bias', [True])
@pytest.mark.parametrize('dtype', [torch.bfloat16])
def test_op_fwd_bias(Z, H, N_CTX_Q, N_CTX_K, D_HEAD, causal, use_bias, dtype):
    current_platform.seed_everything(0)
    sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    q, k, v, input_metadata = input_helper(Z,
                                           H,
                                           H,
                                           N_CTX_Q,
                                           N_CTX_K,
                                           D_HEAD,
                                           dtype,
                                           layout='bhsd')
    input_metadata.eight_bit_dtype_torch = current_platform.fp8_dtype()
    if causal:
        input_metadata.need_causal()
    if use_bias:
        bias = torch.randn((1, H, N_CTX_Q, N_CTX_K),
                           dtype=dtype,
                           device="cuda")
        input_metadata.need_bias(bias, N_CTX_Q, N_CTX_K)
    else:
        bias = None
    o = torch.empty_like(q)

    # triton implementation
    tri_out, _ = triton_attention_rocm(q, k, v, o, input_metadata)

    ref_impl = ReferenceAttention(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD, False,
                                  dtype, input_metadata)
    ref_out = ref_impl.fwd(q, k, v)

    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(4, 48, 256, 64),
                                                 (4, 48, 512, 64),
                                                 (4, 48, 128, 128)])
@pytest.mark.parametrize('causal', [True, False])
def test_op_varlen_fwd(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):

    q, k, v, input_metadata = varlen_input_helper(Z, H, H, N_CTX, N_CTX,
                                                  D_HEAD, dtype)

    tri_out = torch.empty_like(q)
    triton_attention_rocm(q, k, v, tri_out, input_metadata)

    ref_impl = ReferenceAttention(Z, H, H, N_CTX, N_CTX, D_HEAD, False, dtype,
                                  input_metadata)
    ref_out = ref_impl.varlen_fwd(q, k, v, is_mqa=False)

    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, HQ, HK, N_CTX, D_HEAD', [(2, 48, 24, 128, 64),
                                                      (4, 48, 12, 256, 64),
                                                      (4, 48, 4, 512, 64),
                                                      (4, 64, 16, 128, 128)])
@pytest.mark.parametrize('causal', [False])
def test_op_varlen_mqa_fwd(Z,
                           HQ,
                           HK,
                           N_CTX,
                           D_HEAD,
                           causal,
                           dtype=torch.float16):
    q, k, v, input_metadata = varlen_input_helper(Z, HQ, HK, N_CTX, N_CTX,
                                                  D_HEAD, dtype)
    input_metadata.eight_bit_dtype_torch = current_platform.fp8_dtype()

    tri_out = torch.empty_like(q)
    triton_attention_rocm(q, k, v, tri_out, input_metadata)

    ref_impl = ReferenceAttention(Z, HQ, HK, N_CTX, N_CTX, D_HEAD, False,
                                  dtype, input_metadata)
    ref_out = ref_impl.varlen_fwd(q, k, v, is_mqa=True)

    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)
