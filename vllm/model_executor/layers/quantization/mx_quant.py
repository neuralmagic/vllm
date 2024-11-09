from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.parameter import ModelWeightParameter

MXFP8_E5M2 = "fp8e5"
MXFP8_E4M3 = "fp8e4"
MXFP6_E3M2 = "fp6e3"
MXFP6_E2M3 = "fp6e2"
MXFP4_E2M1 = "fp4e2"

# Supported element dtypes
# TODO: add support for MXINT8
SUPPORTED_DTYPES = [
    MXFP8_E5M2,
    MXFP8_E4M3,
    MXFP6_E3M2,
    MXFP6_E2M3,
    MXFP4_E2M1,
]

SUPPORTED_MX_CONFIGS = [
    *[
        f"w{wdtype}_a{adtype}"
        for wdtype, adtype in zip(SUPPORTED_DTYPES, SUPPORTED_DTYPES)
    ],
    *[f"w{wdtype}" for wdtype in SUPPORTED_DTYPES],
]

# https://github.com/pytorch/ao/blob/71a442ae775e0ea5a541dcce637b128070d1243c/torchao/prototype/mx_formats/constants.py#L3-L18
DTYPE_MAP = {
    MXFP8_E5M2: torch.float8_e5m2,
    MXFP8_E4M3: torch.float8_e4m3fn,
    MXFP6_E3M2: "fp6_e3m2",
    MXFP6_E2M3: "fp6_e2m3",
    MXFP4_E2M1: "fp4_e2m1",
}


class MXConfig(QuantizationConfig):
    """MX Quantization Configuration."""

    def __init__(
        self,
        weight_dtype: str,
        act_dtype: Optional[str],
    ) -> None:
        if weight_dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported weight scheme {weight_dtype}")
        if act_dtype and act_dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported activation scheme {act_dtype}")

        self.weight_dtype = DTYPE_MAP[weight_dtype]
        self.act_dtype = DTYPE_MAP[act_dtype] if act_dtype else None
        # Hardcoded for the MX spec
        self.block_size = 32

    def get_name(self) -> str:
        return "mx_quant"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MXConfig":
        weight_dtype = cls.get_from_keys(config, ["weight_dtype"])
        act_dtype = cls.get_from_keys(config, ["act_dtype"])
        return cls(weight_dtype=weight_dtype, act_dtype=act_dtype)

    def get_quant_method(self, layer: Module,
                         prefix: str) -> Optional["MXLinearMethod"]:
        if isinstance(layer, LinearBase):
            return MXLinearMethod(self)
        return None


class MXLinearMethod(LinearMethodBase):
    """Linear method for MX quant. """

    def __init__(self, quant_config: MXConfig):
        try:
            import torchao  # noqa: F401
        except ImportError as err:
            raise ImportError("Please install torchao==0.6.1 via "
                              "`pip install torchao==0.6.1` to use "
                              "mx quantization.") from err
        self.quant_config = quant_config

    def create_weights(self, layer: Module, input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        del input_size, output_size  # Unused.
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=params_dtype),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: Module) -> None:
        from torchao.prototype.mx_formats.mx_tensor import MXTensor
        layer.weight_mx = MXTensor.to_mx(
            layer.weight.data.t().contiguous().to(torch.float32),
            self.quant_config.weight_dtype, self.quant_config.block_size)
        layer.weight = None

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        orig_dtype = x.dtype
        from torchao.prototype.mx_formats.mx_tensor import MXTensor
        if not self.quant_config.act_dtype:
            weight = layer.weight_mx.to_dtype(orig_dtype).t().contiguous()
            out = torch.nn.functional.linear(x, weight, bias)
        else:
            x = MXTensor.to_mx(x.to(torch.float32),
                               self.quant_config.act_dtype,
                               self.quant_config.block_size)
            out = torch.mm(x, layer.weight_mx)
            if bias:
                out += bias

        return out.to(orig_dtype)
