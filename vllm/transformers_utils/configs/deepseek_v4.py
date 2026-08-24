# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import PretrainedConfig


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
        self.rope_parameters = rope_scaling or rope_parameters
        super().__init__(**kwargs)
        if not hasattr(self, "num_hash_layers"):
            mlp_layer_types = getattr(self, "mlp_layer_types", None)
            if mlp_layer_types is not None:
                count = 0
                for t in mlp_layer_types:
                    if t == "hash_moe":
                        count += 1
                    else:
                        break
                self.num_hash_layers = count
            else:
                self.num_hash_layers = 0
