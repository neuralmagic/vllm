# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for QuantFP8 Group Quantization implementation."""

import pytest
import torch

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape)
from vllm.platforms import current_platform


@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("hidden_dim",
                         [256, 512, 513])  # Include non-divisible
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("seed", [42])
@torch.inference_mode()
def test_quantfp8_group_basic(batch_size: int, hidden_dim: int,
                              group_size: int, seed: int) -> None:
    current_platform.seed_everything(seed)

    x = torch.randn(
        (batch_size, hidden_dim), dtype=torch.bfloat16, device="cuda") * 8

    # Create QuantFP8 with group quantization
    group_shape = GroupShape(1, group_size)
    quant_op = QuantFP8(static=False,
                        group_shape=group_shape,
                        column_major_scales=False)

    expected_num_groups = (hidden_dim + group_size - 1) // group_size

    # Test CUDA implementation (only supports divisible dimensions)
    if hidden_dim % group_size == 0:
        x_quant_cuda, scales_cuda = quant_op.forward_cuda(x.clone())
        assert x_quant_cuda.shape == x.shape
        assert scales_cuda.shape == (batch_size, expected_num_groups)

    # Test PyTorch native implementation
    x_quant_native, scales_native = quant_op.forward_native(x.clone())
    assert x_quant_native.shape == x.shape
    assert scales_native.shape == (batch_size, expected_num_groups)

    # Test column_major_scales
    quant_op_col = QuantFP8(static=False,
                            group_shape=group_shape,
                            column_major_scales=True)
    _, scales_col = quant_op_col.forward_native(x.clone())
    assert scales_col.shape == (expected_num_groups, batch_size)


@pytest.mark.parametrize("seed", [42])
@torch.inference_mode()
def test_quantfp8_group_multidimensional(seed: int) -> None:
    current_platform.seed_everything(seed)

    group_size = 64

    # Test with 3D input
    batch1, batch2, hidden_dim = 4, 8, 512
    x_3d = torch.randn(
        (batch1, batch2, hidden_dim), dtype=torch.bfloat16, device="cuda") * 8

    group_shape = GroupShape(1, group_size)
    quant_op = QuantFP8(static=False,
                        group_shape=group_shape,
                        column_major_scales=False)

    x_quant, scales = quant_op.forward_native(x_3d.clone())
    assert x_quant.shape == x_3d.shape
    assert scales.shape == (batch1, batch2, hidden_dim // group_size)

    # Test column_major_scales with multi-dim
    quant_op_col = QuantFP8(static=False,
                            group_shape=group_shape,
                            column_major_scales=True)
    _, scales_col = quant_op_col.forward_native(x_3d.clone())
    assert scales_col.shape == (batch1, hidden_dim // group_size, batch2)

    # Test with 4D input
    batch1, batch2, batch3, hidden_dim = 2, 3, 4, 256
    x_4d = torch.randn((batch1, batch2, batch3, hidden_dim),
                       dtype=torch.bfloat16,
                       device="cuda") * 8

    x_quant_4d, scales_4d = quant_op.forward_native(x_4d.clone())
    assert x_quant_4d.shape == x_4d.shape
    assert scales_4d.shape == (batch1, batch2, batch3,
                               hidden_dim // group_size)

    _, scales_4d_col = quant_op_col.forward_native(x_4d.clone())
    assert scales_4d_col.shape == (batch1, batch2, hidden_dim // group_size,
                                   batch3)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("hidden_dim", [1024])
@pytest.mark.parametrize("group_size", [128])
@pytest.mark.parametrize("seed", [42])
@torch.inference_mode()
def test_quantfp8_group_cuda_native_consistency(batch_size: int,
                                                hidden_dim: int,
                                                group_size: int,
                                                seed: int) -> None:
    """Compare CUDA and native implementations for consistency."""
    current_platform.seed_everything(seed)

    x = torch.randn(
        (batch_size, hidden_dim), dtype=torch.bfloat16, device="cuda") * 8

    group_shape = GroupShape(1, group_size)
    quant_op = QuantFP8(static=False,
                        group_shape=group_shape,
                        column_major_scales=False)

    # Run both implementations
    x_quant_cuda, scales_cuda = quant_op.forward_cuda(x.clone())
    x_quant_native, scales_native = quant_op.forward_native(x.clone())

    # Check shapes match
    assert x_quant_cuda.shape == x_quant_native.shape
    assert scales_cuda.shape == scales_native.shape

    # Scales should match
    assert torch.allclose(scales_cuda, scales_native, rtol=1e-9, atol=1e-8)

    # Quantized values should mostly match, with rare rounding differences
    # FP8 rounding at boundaries can differ between CUDA and PyTorch
    diff_count = (x_quant_cuda != x_quant_native).sum().item()
    diff_ratio = diff_count / x_quant_cuda.numel()
    assert diff_ratio < 0.002, f"Too many differences: {diff_ratio:.4%}"


@pytest.mark.parametrize("seed", [42])
@torch.inference_mode()
def test_quantfp8_group_edge_cases(seed: int) -> None:
    current_platform.seed_everything(seed)

    batch_size = 16
    group_size = 64

    # Test with single group (group_size >= hidden_dim)
    x_small = torch.randn(
        (batch_size, 32), dtype=torch.bfloat16, device="cuda") * 8
    group_shape = GroupShape(1, group_size)
    quant_op = QuantFP8(static=False,
                        group_shape=group_shape,
                        column_major_scales=False)

    x_quant_small, scales_small = quant_op.forward_native(x_small.clone())
    assert x_quant_small.shape == x_small.shape
    assert scales_small.shape == (batch_size, 1)

    # Test with zero inputs
    x_zero = torch.zeros((batch_size, 256),
                         dtype=torch.bfloat16,
                         device="cuda")
    x_quant_zero, scales_zero = quant_op.forward_native(x_zero.clone())
    assert x_quant_zero.shape == x_zero.shape
    assert (scales_zero > 0).all(), "Scales should be clamped to minimum"

    # Test very large values
    x_large = torch.full((batch_size, 256),
                         1000.0,
                         dtype=torch.bfloat16,
                         device="cuda")
    x_quant_large, scales_large = quant_op.forward_native(x_large.clone())
    assert x_quant_large.shape == x_large.shape
    # FP8 max is typically 448 or 224, so scales should be > 1
    assert (scales_large > 1.0).all(), "Large values should have scales > 1"


@pytest.mark.parametrize(
    "batch_size,hidden_dim,group_size",
    [
        (16, 256, 16),  # Small
        (64, 1024, 64),  # Medium
        (128, 2048, 128),  # Large
        (8, 513, 64),  # Non-divisible (native only)
    ])
@pytest.mark.parametrize("seed", [42])
@torch.inference_mode()
def test_quantfp8_group_various_configs(batch_size: int, hidden_dim: int,
                                        group_size: int, seed: int) -> None:
    current_platform.seed_everything(seed)

    x = torch.randn(
        (batch_size, hidden_dim), dtype=torch.bfloat16, device="cuda") * 8
    group_shape = GroupShape(1, group_size)
    quant_op = QuantFP8(static=False,
                        group_shape=group_shape,
                        column_major_scales=False)

    expected_num_groups = (hidden_dim + group_size - 1) // group_size

    x_quant_native, scales_native = quant_op.forward_native(x.clone())
    assert x_quant_native.shape == x.shape
    assert scales_native.shape == (batch_size, expected_num_groups)

    if hidden_dim % group_size == 0:
        x_quant_cuda, scales_cuda = quant_op.forward_cuda(x.clone())
        assert x_quant_cuda.shape == x.shape
        assert scales_cuda.shape == (batch_size, expected_num_groups)
        assert torch.allclose(scales_cuda, scales_native, rtol=1e-9, atol=1e-8)
