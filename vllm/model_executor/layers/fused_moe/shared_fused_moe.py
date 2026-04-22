# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


# TODO(bnell): this will be deleted
def SharedFusedMoE(*args, **kwargs) -> MoERunner:
    return FusedMoE(*args, **kwargs)
