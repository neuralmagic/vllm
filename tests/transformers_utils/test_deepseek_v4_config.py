# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config as TransformersDeepseekV4Config,
)

from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config


@pytest.mark.parametrize("model_id", [
    "deepseek-ai/DeepSeek-V4-Pro",
    "nvidia/DeepSeek-V4-Flash-NVFP4",
    "inference-optimization/DeepSeek-V4-Pro-0.5B-A0.37B-NVFP4-FP8",
])
def test_config_loads(model_id):
    ref = TransformersDeepseekV4Config.from_pretrained(model_id)
    config = get_config(model_id, trust_remote_code=False)

    assert isinstance(config, DeepseekV4Config)

    # compress_ratios must be per-layer consistent with what transformers
    # derives from compress_rates + layer_types (truncated to num_hidden_layers;
    # the raw JSON may carry one extra MTP entry which attention.py ignores).
    expected_compress_ratios = [
        ref.compress_rates.get(lt, 1) for lt in ref.layer_types
    ]
    # Legacy configs use 0 for uncompressed layers; attention.py normalises with
    # max(1, ratio), so treat 0 and 1 as equivalent here.
    actual_ratios = [max(1, r) for r in config.compress_ratios[:ref.num_hidden_layers]]
    assert actual_ratios == expected_compress_ratios

    # mlp_layer_types drives hash-MoE routing; must match when present.
    if hasattr(config, "mlp_layer_types"):
        assert config.mlp_layer_types == ref.mlp_layer_types

    # qk_rope_head_dim determines the RoPE slice in every attention layer.
    assert config.qk_rope_head_dim == ref.qk_rope_head_dim
