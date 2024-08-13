from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


class Sparsity24Config(QuantizationConfig):
    """Config class for 2:4 sparsity."""

    def __init__(self) -> None:
        return

    @classmethod
    def get_name(cls) -> str:
        return "sparsity_24"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Sparsity24Config":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            return Sparsity24LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class Sparsity24LinearMethod(LinearMethodBase):
    """Linear method for Sparsity24.
    Supports loading FP16/BF16 model checkpoints as dense weights.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Sparsity24Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # WEIGHT
        weight = Parameter(torch.empty(output_size_per_partition,
                                       input_size_per_partition,
                                       dtype=params_dtype),
                           requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            **extra_weight_attrs,
            "input_dim": 1,
            "output_dim": 0,
        })

    def process_weights_after_loading(self, layer: Module) -> None:
        from torch.sparse import to_sparse_semi_structured

        layer.weight = torch.nn.Parameter(to_sparse_semi_structured(
            layer.weight),
                                          requires_grad=False)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        return torch.nn.functional.linear(x, layer.weight, bias=bias)
