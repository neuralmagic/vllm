# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import Any, Optional

from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, FusedMoEMethodBase, FusedMoeWeightScaleSupported)
from vllm.triton_utils import HAS_TRITON

_config: Optional[dict[str, Any]] = None


@contextmanager
def override_config(config):
    global _config
    old_config = _config
    _config = config
    yield
    _config = old_config


def get_config() -> Optional[dict[str, Any]]:
    return _config


__all__ = [
    "FusedMoE",
    "FusedMoEMethodBase",
    "FusedMoeWeightScaleSupported",
    "override_config",
    "get_config",
]

if HAS_TRITON:
    # import to register the custom ops
    import vllm.model_executor.layers.fused_moe.fused_marlin_moe  # noqa
    import vllm.model_executor.layers.fused_moe.fused_moe  # noqa
    from vllm.model_executor.layers.fused_moe.cutlass_moe import (
        cutlass_moe_fp4, cutlass_moe_fp8)
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        TritonExperts, fused_experts, fused_moe, fused_topk,
        get_config_file_name, grouped_topk)

    __all__ += [
        "fused_moe",
        "fused_topk",
        "fused_experts",
        "get_config_file_name",
        "grouped_topk",
        "cutlass_moe_fp8",
        "cutlass_moe_fp4",
        "TritonExperts",
    ]
