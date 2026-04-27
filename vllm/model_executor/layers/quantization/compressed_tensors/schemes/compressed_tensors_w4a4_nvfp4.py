# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

__all__ = ["CompressedTensorsW4A4Fp4"]


class CompressedTensorsW4A4Fp4(CompressedTensorsScheme):
    def __init__(self):
        self.kernel = init_nvfp4_linear_kernel()
        self.group_size = 16

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Global Weight Scale
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Rename CT checkpoint names to standardized names
        layer.weight = layer.weight_packed
        del layer.weight_packed

        # CT stores global scales as divisors (1/actual_scale)
        stored_input_gs = layer.input_global_scale.data.to(torch.float32)
        stored_weight_gs = layer.weight_global_scale.data.to(torch.float32)

        # Input: use max of stored (= min of actual) for conservative quantization
        input_gs_inv = stored_input_gs.max()
        input_gs = (1.0 / input_gs_inv).to(torch.float32)

        # Weight: per-partition actual scale (actual = 1/stored)
        weight_gs_per_partition = 1.0 / stored_weight_gs

        # Compute per-partition alpha = input_gs * weight_gs[i]
        alpha_per_partition = input_gs * weight_gs_per_partition
        if (
            torch.unique(stored_weight_gs).numel() == 1
            and torch.unique(stored_input_gs).numel() == 1
        ):
            layer.alpha = Parameter(alpha_per_partition[0:1], requires_grad=False)
        else:
            alpha_per_column = alpha_per_partition.repeat_interleave(
                torch.tensor(layer.logical_widths, device=alpha_per_partition.device)
            )
            layer.alpha = Parameter(alpha_per_column, requires_grad=False)

        # Set scalar values for non-CUTLASS backends
        layer.input_global_scale = Parameter(input_gs, requires_grad=False)
        layer.weight_global_scale = Parameter(
            weight_gs_per_partition.max(), requires_grad=False
        )
        layer.input_global_scale_inv = Parameter(input_gs_inv, requires_grad=False)

        # Convert layer to NVFP4 linear kernel format
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer=layer, x=x, bias=bias)
