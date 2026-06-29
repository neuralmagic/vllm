# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.humming_utils import (
    prepare_humming_layer,
)
from vllm.platforms import current_platform

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)


class HummingNvFp4LinearKernel(NvFp4LinearKernel):
    """Humming GEMM Kernel for NVFP4."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "Humming only supported on CUDA"

        if not current_platform.has_device_capability(75):
            return False, "Humming only supported on SM75+"

        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Consume the canonical nvfp4 layout (packed fp4 weight + e4m3 group-16
        # scale + fp32 global scale) through humming's compressed-tensors nvfp4
        # loader -- the same canonical -> humming path the MoE oracle uses.
        # The native humming `group_tensor` schema mishandles a scalar global
        # scale (it folds weight_scale*global to bf16 but leaves the on-layer
        # tensor e4m3, tripping humming's dtype validation).
        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "type": "float",
            "num_bits": 4,
            "strategy": "group",
            "group_size": 16,
        }
        # humming's CT pack-quantized loader reads `weight_packed`; the nvfp4
        # linear scheme renames it to `weight`.
        if not hasattr(layer, "weight_packed"):
            layer.weight_packed = layer.weight
            del layer.weight
        # The CT nvfp4 linear scheme stores the global scale inverted (1/scale)
        # for its marlin/cutlass kernels; humming's loader expects the original
        # scale (the MoE path leaves it un-inverted on the layer). Restore it.
        layer.weight_global_scale = torch.nn.Parameter(
            1.0 / layer.weight_global_scale, requires_grad=False
        )
        prepare_humming_layer(layer, quant_config)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from humming.layer import HummingMethod

        flatten_inputs = x.view(-1, x.size(-1))
        output = HummingMethod.forward_layer(
            layer=layer,
            inputs=flatten_inputs,
            compute_config=layer.compute_config,
        )
        return output.view(*x.shape[:-1], output.size(-1))
