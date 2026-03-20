# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next MTP model."""

import itertools
from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextRMSNorm,
    QwenNextMixtureOfExperts,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import Qwen3NextConfig

from .utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)

logger = init_logger(__name__)

KVCache = tuple[torch.Tensor, torch.Tensor]

# Speculators FastMTP → Qwen3NextMultiTokenPredictor weight name mapping.
# Keys are suffixes after "mtp_layers.0.", values are full vLLM param names
# relative to Qwen3NextMultiTokenPredictor (without the outer "model." prefix).
_SPECULATORS_LAYER_REMAP: dict[str, str] = {
    "hidden_layernorm.": "pre_fc_norm_hidden.",
    "token_layernorm.": "pre_fc_norm_embedding.",
    "input_proj.": "fc.",
    "final_layernorm.": "norm.",
    "input_layernorm.": "layers.0.input_layernorm.",
    "post_attention_layernorm.": "layers.0.post_attention_layernorm.",
    "self_attn.": "layers.0.self_attn.",
    "mlp.": "layers.0.mlp.",
}


def _remap_speculators_weight(name: str) -> tuple[str, str] | None:
    """Map a speculators FastMTP weight name to a (target, inner_name) pair.

    Returns ("model", inner_name) for weights belonging to
    Qwen3NextMultiTokenPredictor, ("lm_head", name) for lm_head, or None to skip.
    """
    if name == "lm_head.weight":
        return "lm_head", name
    if name == "embed_tokens.weight":
        return "model", name
    if name.startswith("mtp_layers.0."):
        inner = name[len("mtp_layers.0."):]
        for prefix, mapped in _SPECULATORS_LAYER_REMAP.items():
            if inner.startswith(prefix):
                return "model", mapped + inner[len(prefix):]
    return None


@support_torch_compile
class Qwen3NextMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config

        config: Qwen3NextConfig = model_config.hf_config

        self.config = config

        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        self.fc = ColumnParallelLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            gather_output=True,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc",
        )

        self.layers = torch.nn.ModuleList(
            Qwen3NextDecoderLayer(
                vllm_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_embedding = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
            hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
            hidden_states = self.fc(hidden_states)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        hidden_states, residual = self.layers[current_step_idx](
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


@support_torch_compile
class Qwen3NextMTP(nn.Module, QwenNextMixtureOfExperts):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["up_proj", "down_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3NextMTP currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen3NextMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Auto-detect checkpoint format from weight key names.
        # Standalone speculators checkpoints use "mtp_layers.N.*" keys;
        # base-model-embedded checkpoints use "mtp.*" keys.
        # Peek at the first few keys — standalone format always has an
        # "mtp_layers." key within the first handful of entries.
        it = iter(weights)
        peek = list(itertools.islice(it, 10))
        weights = itertools.chain(peek, it)
        # Base-model checkpoints always have "model.*"-prefixed keys for the
        # transformer layers (e.g. "model.embed_tokens.weight" is key #1).
        # Standalone speculators checkpoints never have "model.*" keys.
        is_standalone = not any(n.startswith("model.") for n, _ in peek)

        if not is_standalone:
            # Base-model-embedded format: MTP weights carry the "mtp." prefix.
            shared_weight_names = ["embed_tokens", "lm_head"]

            def remap_weight_names(w):
                for name, weight in w:
                    if name.startswith("mtp."):
                        name = name.replace("mtp.", "model.")
                    elif not any(key in name for key in shared_weight_names):
                        continue
                    yield name, weight

            loader = AutoWeightsLoader(self)
            return loader.load_weights(remap_weight_names(weights))

        # Standalone speculators checkpoint: "mtp_layers.0.*" naming.
        # embed_tokens and lm_head are identical to the verifier's — skip them.
        # Only remap "mtp_layers.0.*" keys into Qwen3NextMultiTokenPredictor
        # namespace and delegate to its load_weights(), which correctly handles
        # QKV fusion and FusedMoE experts via stacked/expert_params_mapping.
        model_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if not name.startswith("mtp_layers."):
                continue
            remapped = _remap_speculators_weight(name)
            if remapped is None:
                continue
            _, inner_name = remapped
            model_weights.append((inner_name, weight))

        # self.model.load_weights() returns names relative to
        # Qwen3NextMultiTokenPredictor; prefix with "model." to match
        # Qwen3NextMTP.named_parameters() which default_loader checks against.
        loaded = {
            f"model.{name}"
            for name in self.model.load_weights(iter(model_weights))
        }
        # embed_tokens and lm_head are skipped above; eagle.py's
        # _maybe_share_embeddings/_maybe_share_lm_head will assign them from
        # the verifier. Mark them as loaded so default_loader doesn't error.
        loaded.add("model.embed_tokens.weight")
        loaded.add("lm_head.weight")
        return loaded
