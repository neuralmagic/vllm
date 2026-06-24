#!/usr/bin/env python3
"""
Direct comparison: CompressedTensorsLinearTransformMethod vs QutlassMxFP4LinearMethod

Tests if both methods produce the same output on identical inputs.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kl_div
from compressed_tensors.transform.utils.hadamard import deterministic_hadamard_matrix
from vllm._custom_ops import fusedQuantizeMx, hadacore_transform
from vllm.utils.flashinfer import (
    flashinfer_mxfp4_quantize,
    flashinfer_scaled_fp4_mm,
)
from vllm.model_executor.layers.quantization.qutlass_utils import to_blocked

# Test configuration
batch_size = 128
seq_len = 128
in_features = 4096  # hidden_size
out_features = 4096
head_dim = 128


def plot_distributions(out_0, out_1, out_2, save_path="output_distributions.png"):
    """Plot histograms of output distributions for all methods"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Flatten tensors to 1D for plotting
    data = [
        ("Baseline (BF16)", out_0.cpu().float().flatten().numpy()),
        ("Method 1 (Hadacore)", out_1.cpu().float().flatten().numpy()),
        ("Method 2 (fusedQuantizeMx)", out_2.cpu().float().flatten().numpy()),
    ]

    # Plot individual histograms
    for idx, (name, values) in enumerate(data):
        ax = axes[idx]
        ax.hist(values, bins=100, alpha=0.7, edgecolor="black")
        ax.set_title(
            f"{name}\nRange: [{values.min():.2f}, {values.max():.2f}], Mean: {values.mean():.4f}, Std: {values.std():.2f}"
        )
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved distribution plot to {save_path}")
    plt.close()

    # Also create an overlay comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for name, values in data:
        ax.hist(values, bins=100, alpha=0.4, label=name, edgecolor="none")

    ax.set_title("Output Distribution Comparison")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    overlay_path = "output_distributions_overlay.png"
    plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
    print(f"Saved overlay plot to {overlay_path}")
    plt.close()


def method_0_baseline(x, weight, h):
    """Full precision baseline (BF16 with explicit Hadamard matmul)
    X -> Had(X)
    W -> Had^-1(W)
    return X@W.T
    """
    batch_size, seq_len, in_features = x.shape
    out_features = weight.shape[0]
    head_dim = h.shape[0]

    # Apply Hadamard transform using explicit matrix multiplication
    x_flat = x.flatten(end_dim=-2)  # [batch*seq, in_features]
    x_reshaped = x_flat.view(
        -1, in_features // head_dim, head_dim
    )  # [batch*seq, num_heads, head_dim]
    x_transformed = torch.matmul(x_reshaped, h.T)  # [batch*seq, num_heads, head_dim]
    x_transformed = x_transformed.view(-1, in_features)  # [batch*seq, in_features]

    # Apply Hadamard to weights
    weight_reshaped = weight.view(out_features, in_features // head_dim, head_dim)
    weight_transformed = torch.matmul(weight_reshaped, h.T)
    weight_transformed = weight_transformed.view(out_features, in_features)

    # BF16 matmul
    out_flat = torch.mm(x_transformed, weight_transformed.T)
    return out_flat.view(batch_size, seq_len, out_features), None


def method_1_hadacore(x, weight, h):
    """CompressedTensorsLinearTransformMethod pathway"""
    batch_size, seq_len, in_features = x.shape
    out_features = weight.shape[0]

    # Transform and quantize activations
    x_flat = x.flatten(end_dim=-2)
    x_transformed = hadacore_transform(x_flat, inplace=False)
    x_fp4, x_scales = flashinfer_mxfp4_quantize(x_transformed)

    # Transform and quantize weights
    weight_reshaped = weight.view(out_features, in_features // head_dim, head_dim)
    weight_transformed = torch.matmul(weight_reshaped, h.T)
    weight_transformed = weight_transformed.view(out_features, in_features)
    weight_fp4, weight_scales = flashinfer_mxfp4_quantize(weight_transformed)

    # FP4 matmul
    out_flat = flashinfer_scaled_fp4_mm(
        x_fp4,
        weight_fp4,
        x_scales,
        weight_scales,
        alpha=None,
        out_dtype=torch.bfloat16,
        backend="cudnn",
        block_size=32,
        use_nvfp4=False,
    )

    return out_flat.view(batch_size, seq_len, out_features), x_scales


def method_2_qutlass_mxfp4(x, weight, h):
    """QutlassMxFP4LinearMethod pathway"""
    batch_size, seq_len, in_features = x.shape
    out_features = weight.shape[0]

    # Fused quantize activations (h should already be normalized)
    x_flat = x.flatten(end_dim=-2)
    # x_fp4, x_scales = fusedQuantizeMx(x_flat, h / 3, method="abs_max")
    x_fp4, x_scales = fusedQuantizeMx(x_flat, h, method="quest")
    x_scales_blocked = to_blocked(x_scales, backend="triton").view(x_scales.shape)

    # Transform and quantize weights (same as Method 1)
    weight_reshaped = weight.view(out_features, in_features // head_dim, head_dim)
    weight_transformed = torch.matmul(weight_reshaped, h.T)
    weight_transformed = weight_transformed.view(out_features, in_features)
    weight_fp4, weight_scales = flashinfer_mxfp4_quantize(weight_transformed)

    # FP4 matmul
    out_flat = flashinfer_scaled_fp4_mm(
        x_fp4,
        weight_fp4,
        x_scales_blocked,
        weight_scales,
        alpha=None,
        out_dtype=torch.bfloat16,
        backend="cudnn",
        block_size=32,
        use_nvfp4=False,
    )

    return out_flat.view(batch_size, seq_len, out_features), x_scales_blocked


def main():
    torch.manual_seed(42)
    device = "cuda"

    print("=" * 80)
    print("MSE vs Full Precision Baseline")
    print("=" * 80)
    print(
        f"Config: batch={batch_size}, seq={seq_len}, features={in_features}, head_dim={head_dim}"
    )
    print()

    # Create input activations (simulating post-LayerNorm: mean~0, std~1)
    x = torch.randn(
        batch_size, seq_len, in_features, dtype=torch.bfloat16, device=device
    )
    # Normalize to mean=0, std=1 (like LayerNorm output)
    x = (x - x.mean()) / x.std()

    # Create weight matrix (simulating Xavier/Kaiming init for linear layer)
    # Xavier: std = sqrt(2 / (fan_in + fan_out))
    std = (2.0 / (in_features + out_features)) ** 0.5
    weight = (
        torch.randn(out_features, in_features, dtype=torch.bfloat16, device=device)
        * std
    )

    # Create Hadamard matrix (normalized)
    h = deterministic_hadamard_matrix(head_dim, dtype=torch.bfloat16, device=device) / (
        head_dim**0.5
    )

    # Run methods (each takes x, weight, h_normalized and handles everything internally)
    out_0, _ = method_0_baseline(x.clone(), weight.clone(), h.clone())
    out_1, x_scales_1 = method_1_hadacore(x.clone(), weight.clone(), h.clone())
    out_2, x_scales_2 = method_2_qutlass_mxfp4(x.clone(), weight.clone(), h.clone())

    # =========================================================================
    # Diagnostics: Compare scales
    # =========================================================================
    print("Scale dtypes and shapes:")
    print(f"  Method 1: dtype={x_scales_1.dtype}, shape={x_scales_1.shape}")
    print(f"  Method 2: dtype={x_scales_2.dtype}, shape={x_scales_2.shape}")
    print()

    # Interpret E8M0 properly: scale = 2^(exponent - 127)
    # For Method 1: view as uint8, interpret as E8M0
    x_scales_1_uint8 = x_scales_1.view(torch.uint8)
    x_scales_1_decoded = torch.pow(2.0, x_scales_1_uint8.float() - 127.0)

    # For Method 2: already might be decoded
    x_scales_2_float = x_scales_2.float()

    print("Decoded scale statistics:")
    print(
        f"  Method 1: min={x_scales_1_decoded.min().item():.6f}, max={x_scales_1_decoded.max().item():.6f}, mean={x_scales_1_decoded.mean().item():.6f}"
    )
    print(
        f"  Method 2: min={x_scales_2_float.min().item():.6f}, max={x_scales_2_float.max().item():.6f}, mean={x_scales_2_float.mean().item():.6f}"
    )

    # =========================================================================
    # Results
    # =========================================================================
    print(
        "Baseline (BF16):                       range=[{:.2f}, {:.2f}], mean={:.4f}, std={:.2f}".format(
            out_0.min().item(),
            out_0.max().item(),
            out_0.mean().item(),
            out_0.std().item(),
        )
    )

    mse_1 = ((out_1 - out_0) ** 2).mean().item()
    print(
        "Method 1 (Hadacore):                   range=[{:.2f}, {:.2f}], mean={:.4f}, std={:.2f}, MSE={:.2f}".format(
            out_1.min().item(),
            out_1.max().item(),
            out_1.mean().item(),
            out_1.std().item(),
            mse_1,
        )
    )

    mse_2 = ((out_2 - out_0) ** 2).mean().item()
    print(
        "Method 2 (QutlassMxFP4):               range=[{:.2f}, {:.2f}], mean={:.4f}, std={:.2f}, MSE={:.2f}".format(
            out_2.min().item(),
            out_2.max().item(),
            out_2.mean().item(),
            out_2.std().item(),
            mse_2,
        )
    )
    print()

    # =========================================================================
    # Distribution Metrics
    # =========================================================================
    print("Distribution Metrics vs Baseline:")

    # Convert to numpy for scipy
    out_0_np = out_0.cpu().float().flatten().numpy()
    out_1_np = out_1.cpu().float().flatten().numpy()
    out_2_np = out_2.cpu().float().flatten().numpy()

    # KL divergence (requires histograms)
    bins = np.linspace(
        min(out_0_np.min(), out_1_np.min(), out_2_np.min()),
        max(out_0_np.max(), out_1_np.max(), out_2_np.max()),
        1000,
    )

    hist_0, _ = np.histogram(out_0_np, bins=bins, density=True)
    hist_1, _ = np.histogram(out_1_np, bins=bins, density=True)
    hist_2, _ = np.histogram(out_2_np, bins=bins, density=True)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    hist_0 = hist_0 + eps
    hist_1 = hist_1 + eps
    hist_2 = hist_2 + eps

    # Normalize
    hist_0 = hist_0 / hist_0.sum()
    hist_1 = hist_1 / hist_1.sum()
    hist_2 = hist_2 / hist_2.sum()

    kl_1 = np.sum(kl_div(hist_1, hist_0))
    kl_2 = np.sum(kl_div(hist_2, hist_0))

    print(f"  Method 1: KL divergence={kl_1:.6f}")
    print(f"  Method 2: KL divergence={kl_2:.6f}")
    print()

    # =========================================================================
    # Correlation Analysis (to check if values are permuted/shuffled)
    # =========================================================================
    print("Correlation with Baseline (1.0 = perfect match, 0.0 = uncorrelated):")

    # Pearson correlation coefficient
    corr_1 = np.corrcoef(out_0_np, out_1_np)[0, 1]
    corr_2 = np.corrcoef(out_0_np, out_2_np)[0, 1]

    print(f"  Method 1: Correlation={corr_1:.6f}")
    print(f"  Method 2: Correlation={corr_2:.6f}")
    print()

    best_mse = min(mse_1, mse_2)
    if best_mse == mse_1:
        print(f"✅ Method 1 is best")
    else:
        print(f"✅ Method 2 is best")

    # Plot distributions
    plot_distributions(out_0, out_1, out_2)


if __name__ == "__main__":
    main()
