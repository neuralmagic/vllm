# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/deepseek_mtp.py
# Copyright 2025 Xiaomi Corporation.
# Copyright 2023 The vLLM team.
# Copyright 2024 DeepSeek-AI team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only MiMo-MTP model."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer
from vllm.sequence import IntermediateTensors

from .utils import maybe_prefix


class MiMoMultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()

        self.token_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        self.mtp_block = Qwen2DecoderLayer(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        # Position 0 tokens are not used by MTP; zero them out to avoid
        # leaking information across unrelated sequences in the batch.
        inputs_embeds[positions == 0] = 0
        inputs_embeds = self.token_layernorm(inputs_embeds)
        previous_hidden_states = self.hidden_layernorm(previous_hidden_states)

        hidden_states = self.input_proj(
            torch.cat([previous_hidden_states, inputs_embeds], dim=-1)
        )

        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        hidden_states = residual + hidden_states
        return self.final_layernorm(hidden_states)


class MiMoMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.mtp_layers = torch.nn.ModuleDict(
            {
                str(idx): MiMoMultiTokenPredictorLayer(
                    config,
                    f"{prefix}.layers.{idx}",
                    model_config=vllm_config.model_config,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )

        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        return self.mtp_layers[str(self.mtp_start_layer_idx + spec_step_idx)](
            inputs_embeds,
            positions,
            previous_hidden_states,
            spec_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.logits_processor(lm_head, hidden_states)


class MiMoMTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = MiMoMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        assert spec_step_idx == 0, "mimo_mtp only support predict one token now"
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, self.lm_head, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            name = self.map_model_name_to_mtp_param_name(name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mtp_layers" not in name:
                    break
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if "mtp_layers" not in name and (
                    "embed_tokens" not in name and "lm_head" not in name
                ):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        # embed_tokens and lm_head are shared from the verifier at load time;
        # mark them as loaded so the strict weight check in default_loader does
        # not raise. eagle.py replaces these modules with the verifier's copies.
        for param_name in params_dict:
            if "embed_tokens" in param_name or "lm_head" in param_name:
                loaded_params.add(param_name)
        return loaded_params

    def map_model_name_to_mtp_param_name(self, name: str) -> str:
        import regex as re

        # The checkpoint stores MTP layer indices starting at 0; our module
        # keys them from num_hidden_layers onward to avoid colliding with the
        # verifier's layer indices.
        pattern = r"(model\.mtp_layers\.)(\d+)(\.)"
        match = re.match(pattern, name)
        if match:
            original_num = int(match.group(2))
            new_num = original_num + self.config.num_hidden_layers
            name = name.replace(match.group(), f"{match.group(1)}{new_num}.")

        # These weights are direct attributes of the MTP layer, not inside
        # the inner transformer block, so no further renaming is needed.
        direct_mtp_weights = [
            "token_layernorm",
            "hidden_layernorm",
            "input_proj",
            "final_layernorm",
        ]
        if any(w in name for w in direct_mtp_weights):
            return name

        # All other weights belong to the inner Qwen2DecoderLayer, which is
        # stored under the .mtp_block attribute.
        pattern = r"(model\.mtp_layers\.\d+\.)"
        match = re.match(pattern, name)
        if match:
            name = name.replace(match.group(), match.group() + "mtp_block.")
        return name
