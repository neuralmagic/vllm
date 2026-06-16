# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter

from vllm._custom_ops import fusedQuantizeMx
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsScheme,
    CompressedTensorsW4A4Mxfp4,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
    CompressedTensorsLinearTransformMethod,
    TransformTuple,
)
from vllm.model_executor.layers.quantization.qutlass_utils import to_blocked
from vllm.utils.flashinfer import (
    flashinfer_scaled_fp4_mm,
)

__all__ = ["is_qutlass_mxfp4_scheme", "QutlassMxFP4LinearMethod"]


def is_qutlass_mxfp4_scheme(
    quant_scheme: CompressedTensorsScheme | None,
    input_tfms: dict[int, TransformTuple],
) -> bool:
    return isinstance(quant_scheme, CompressedTensorsW4A4Mxfp4) and len(input_tfms) >= 1


class QutlassMxFP4LinearMethod(CompressedTensorsLinearTransformMethod):
    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        # initializes mxfp4 qparams
        assert isinstance(layer.scheme, (CompressedTensorsW4A4Mxfp4,))
        ret = super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

        assert self.input_transform is not None
        assert len(self.input_transform.weight.partitions) >= 1
        for partition in self.input_transform.weight.partitions.values():
            assert partition.data.shape[0] == layer.scheme.group_size

        return ret

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)

        assert self.input_transform is not None
        h = self.input_transform.weight.partitions[0].data
        h_normalized = (h * self.input_transform.scales[0]).to(torch.bfloat16)
        layer.hadamard_matrix = Parameter(h_normalized, requires_grad=False)

        # MXFP4 uses E8M0 scales without a global scale factor
        # The GEMM computes alpha * sum(fp4_a * scale_a * fp4_w * scale_w)
        # For MXFP4, alpha is just 1.0 since there's no global scale adjustment
        layer.fused_alpha = Parameter(
            torch.tensor([1.0], dtype=torch.float32, device=h.device),
            requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert bias is None
        output_size = layer.output_size_per_partition
        output_shape = [*x.shape[:-1], output_size]

        x_flat = x.contiguous().flatten(end_dim=-2)

        # fusedQuantizeMx performs Hadamard transform + MXFP4 quantization
        # Returns E2M1 packed values and E8M0 scales
        x_fp4, x_scales = fusedQuantizeMx(x_flat, layer.hadamard_matrix)

        x_scales_blocked = to_blocked(x_scales, backend="triton").view(x_scales.shape)

        out = flashinfer_scaled_fp4_mm(
            x_fp4,
            layer.weight,
            x_scales_blocked,
            layer.weight_scale,
            layer.fused_alpha,
            x.dtype,
            backend="cutlass",
            block_size=32,
            use_nvfp4=False,
        )

        # MXFP4 packs 2 values per byte (same as NVFP4)
        # Output size matches input without needing slicing
        # But we keep the pattern consistent with NVFP4
        if out.shape[-1] != output_size:
            out = out[..., :output_size]

        if self.output_transform is not None:
            for part_id, (start, length) in enumerate(self.partition_ranges):
                out[:, start : start + length] = self.output_transform(
                    out[:, start : start + length].clone(), part_id=part_id
                )

        return out.view(*output_shape)
