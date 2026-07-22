# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import PretrainedConfig
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config as _TransformersDeepseekV4Config,
)


class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"

    def __init__(
        self,
        max_position_embeddings: int = 1048576,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        rope_theta: float = 10000.0,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        # New-style configs store per-branch rope params under 'compress'/'main'
        # sub-dicts. Flatten to the compress branch (the yarn config); rope.py
        # overwrites rope_theta per layer so the branch is selected at runtime.
        rp = rope_scaling or rope_parameters
        if rp and "compress" in rp:
            rp = dict(rp["compress"])
        self.rope_parameters = rp

        # Pop before super() to avoid PretrainedConfig rejecting "hash_moe";
        # re-attach and delegate to the V4-aware validator after.
        mlp_layer_types = kwargs.pop("mlp_layer_types", None)
        num_hash_layers = kwargs.pop("num_hash_layers", None)
        super().__init__(**kwargs)

        if mlp_layer_types is not None:
            self.mlp_layer_types = mlp_layer_types
            _TransformersDeepseekV4Config.validate_layer_type(self)

        if num_hash_layers is not None:
            self.num_hash_layers = num_hash_layers
        elif mlp_layer_types is not None:
            self.num_hash_layers = sum(1 for t in mlp_layer_types if t == "hash_moe")

        # New-style configs store compress_rates (dict keyed by attention type)
        # + layer_types (per-layer attention type list) instead of a legacy
        # compress_ratios list. Derive the list so downstream code can index by
        # layer_id uniformly.
        compress_rates = getattr(self, "compress_rates", None)
        layer_types = getattr(self, "layer_types", None)
        if not hasattr(self, "compress_ratios") and compress_rates and layer_types:
            self.compress_ratios = [compress_rates.get(lt, 1) for lt in layer_types]
