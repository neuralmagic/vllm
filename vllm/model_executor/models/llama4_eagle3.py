# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
# All rights reserved.
#
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
"""
Llama4 Eagle3 implementation for speculative decoding.

This module implements Eagle3 speculative decoding for Llama4 models,
enabling efficient draft token generation using auxiliary hidden states
from multiple layers of the target model.

Key features:
- Single-layer draft model with Llama4 decoder architecture
- Auxiliary hidden state combination from target model
- Draft-to-target vocabulary mapping support
- Advanced quantization and multimodal compatibility
"""

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.torchao import TorchAOConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama4 import (Llama4DecoderLayer,
                                               Llama4ForCausalLM)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.multimodal.inputs import NestedTensors
from vllm.v1.sample.metadata import SamplingMetadata

from .utils import AutoWeightsLoader, maybe_prefix, merge_multimodal_embeddings

logger = init_logger(__name__)


@support_torch_compile
class LlamaModel(nn.Module):
    """
    Eagle3 draft model implementation using Llama4 architecture.
    
    This model implements the Eagle3 speculation pattern with a single
    Llama4 decoder layer that processes input embeddings combined with
    auxiliary hidden states from the target model.
    
    Key architectural features:
    - Single Llama4DecoderLayer for efficient draft generation
    - Linear combination layer for auxiliary hidden states
    - Quantization support via TorchAO
    - Pipeline parallelism awareness
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        start_layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        """
        Initialize the Eagle3 draft model.
        
        Args:
            vllm_config: Global vLLM configuration
            prefix: Module name prefix for parameter naming
            start_layer_id: Layer offset based on target model depth
            quant_config: Quantization configuration for the draft model
        """
        super().__init__()
        self.config = (
            vllm_config.speculative_config.draft_model_config.hf_config)
        self._validate_and_update_config(start_layer_id, quant_config)
        self.vocab_size = self.config.vocab_size

        # Embedding layer for draft model vocabulary
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        # Single decoder layer following Eagle3 pattern
        # The layer ID is offset by target model depth to maintain
        # correct parameter naming and quantization mappings
        self.layer = nn.ModuleList([
            Llama4DecoderLayer(
                self.config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, f"layers.{start_layer_id}"),
            )
        ])

        # Eagle3 auxiliary hidden state combination layer
        # Typically combines 3 auxiliary states from target model layers
        aux_state_multiplier = getattr(self.config, "aux_state_multiplier", 3)
        if hasattr(self.config, "target_hidden_size"):
            # Handle different hidden sizes between target and draft models
            input_dim = self.config.target_hidden_size * aux_state_multiplier
        else:
            # Same hidden size assumption
            input_dim = self.config.hidden_size * aux_state_multiplier

        self.fc = nn.Linear(input_dim, self.config.hidden_size, bias=False)

        # Final layer normalization
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings for the given token IDs."""
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Eagle3 draft generation.
        
        Args:
            input_ids: Input token IDs for draft generation
            positions: Position indices for rotary embeddings
            hidden_states: Auxiliary hidden states from target model
            inputs_embeds: Pre-computed input embeddings (optional)
            multimodal_embeddings: Multimodal embeddings (optional)
            
        Returns:
            Tuple of (hidden_states, hidden_states) following vLLM convention
        """
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)

            # Apply multimodal embeddings if provided
            if multimodal_embeddings is not None:
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids,
                    inputs_embeds,
                    multimodal_embeddings,
                    getattr(self.config, "image_token_index", None),
                )

        # Eagle3 pattern: auxiliary hidden states have same dimension as embeddings
        # This assertion ensures compatibility for the single decoder layer
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1], (
            f"Hidden states dimension {hidden_states.shape[-1]} must match "
            f"input embeddings dimension {inputs_embeds.shape[-1]}")

        # Pass through single Llama4 decoder layer
        # The layer processes concatenated embeddings and hidden states
        residual = None
        hidden_states, residual = self.layers[0](
            positions,
            inputs_embeds,
            hidden_states,
            residual,
        )

        # Final normalization and return
        hidden_states, hidden_prenorm = self.norm(hidden_states, residual)
        return hidden_states, hidden_prenorm

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        """
        Load model weights with Eagle3-specific mappings.
        
        This method handles the specific weight naming conventions used by
        Eagle3 models, including layer remapping and parameter stacking.
        
        Args:
            weights: Iterable of (parameter_name, tensor) pairs
            
        Returns:
            Set of loaded parameter names
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Eagle3 specific: convert midlayer naming to standard layer naming
            if 'midlayer.' in name:
                name = name.replace('midlayer.', 'layers.0.')

            # Handle stacked parameters (QKV and gate/up projections)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle embedding sharing in pipeline parallelism
                if (get_pp_group().world_size == 1
                        and "embed_tokens." in name):
                    # Skip embed_tokens when PP is disabled and shared with target
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # Validate all required parameters were loaded
        for name in params_dict:
            if (get_pp_group().world_size == 1 and "embed_tokens." in name):
                continue
            assert name in loaded_params, f"Parameter {name} was not loaded!"

        return loaded_params

    def _validate_and_update_config(
            self,
            start_layer_id: int,
            quant_config: Optional[QuantizationConfig] = None) -> None:
        """
        Validate and update model configuration for Eagle3 compatibility.
        
        Eagle3 draft models have specific limitations and requirements
        that must be enforced during initialization.
        
        Args:
            start_layer_id: Layer offset for parameter naming
            quant_config: Quantization configuration to update
        """
        # Eagle3 draft models don't support advanced Llama4 features yet
        assert getattr(self.config, 'yoco_global_kv_layer', None) is None, (
            "YOCO global KV layers are not supported in Eagle3 draft models")
        assert getattr(self.config, 'yoco_local_kv_layer', None) is None, (
            "YOCO local KV layers are not supported in Eagle3 draft models")
        assert len(getattr(self.config, 'moe_layers', [])) == 0, (
            "Mixture of Experts layers are not supported in Eagle3 draft models"
        )

        # Pad layer-specific configurations for start_layer_id offset
        # This ensures correct behavior when draft layers have offset indices
        self.config.no_rope_layers = (
            [0] * start_layer_id + getattr(self.config, 'no_rope_layers', []))

        # Update quantization configuration for layer offset
        if isinstance(quant_config, TorchAOConfig):

            def pad_layer_name(layer: str) -> str:
                """Add start_layer_id offset to layer names in quantization config."""
                layer_index = extract_layer_index(layer)
                return layer.replace(str(layer_index),
                                     str(layer_index + start_layer_id))

            # Update module FQN mappings for correct quantization application
            quant_config.torchao_config.module_fqn_to_config = {
                pad_layer_name(layer): quantization
                for layer, quantization in
                quant_config.torchao_config.module_fqn_to_config.items()
            }


class Eagle3Llama4ForCausalLM(Llama4ForCausalLM):
    """
    Eagle3 speculative decoding implementation for Llama4 models.
    
    This class implements the Eagle3 speculation algorithm that uses auxiliary
    hidden states from multiple layers of the target model to generate draft
    tokens efficiently. It supports vocabulary mapping between draft and target
    models and maintains compatibility with Llama4's advanced features.
    
    Architecture highlights:
    - Single-layer draft model with auxiliary state combination
    - Draft-to-target vocabulary mapping for different vocabularies
    - Advanced quantization support (TorchAO)
    - Multimodal input compatibility
    - Pipeline parallelism awareness
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """
        Initialize Eagle3 Llama4 model for speculative decoding.
        
        Args:
            vllm_config: Global vLLM configuration containing model and
                        speculative decoding settings
            prefix: Module name prefix for parameter organization
        """
        # Initialize as nn.Module directly, bypassing parent __init__
        # This is the standard Eagle3 pattern to avoid conflicts
        nn.Module.__init__(self)

        self.config = (
            vllm_config.speculative_config.draft_model_config.hf_config)

        # Calculate layer offset based on target model depth
        # Eagle3 draft layers are conceptually "after" target layers
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config)

        # Draft model may have different quantization settings than target
        quant_config = VllmConfig.get_quantization_config(
            vllm_config.speculative_config.draft_model_config,
            vllm_config.load_config)

        # Initialize the core draft model
        self.model = LlamaModel(vllm_config=vllm_config,
                                prefix="model",
                                start_layer_id=target_layer_num,
                                quant_config=quant_config)

        # Logits processing with optional scaling
        logit_scale = getattr(self.config, "logit_scale", 1.0)

        # Eagle3 uses separate vocabulary for draft model
        # This enables vocabulary-independent speculation
        draft_vocab_size = getattr(self.config, "draft_vocab_size",
                                   self.config.vocab_size)

        self.lm_head = ParallelLMHead(draft_vocab_size,
                                      self.config.hidden_size,
                                      org_num_embeddings=draft_vocab_size,
                                      padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                                      prefix="")

        self.logits_processor = LogitsProcessor(draft_vocab_size,
                                                scale=logit_scale)

        # Draft-to-target vocabulary mapping for Eagle3
        # This tensor maps draft vocabulary indices to target vocabulary
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Eagle3 draft token generation.
        
        Args:
            input_ids: Input token IDs for current speculation step
            positions: Position indices for rotary embeddings
            hidden_states: Auxiliary hidden states from target model
            inputs_embeds: Pre-computed input embeddings (optional)
            
        Returns:
            Tuple of (hidden_states, hidden_states) for vLLM compatibility
        """
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """
        Compute logits with draft-to-target vocabulary mapping.
        
        This method transforms draft model logits to the target model's
        vocabulary space, enabling seamless speculation across different
        vocabularies.
        
        Args:
            hidden_states: Final hidden states from draft model
            sampling_metadata: Sampling configuration and metadata
            
        Returns:
            Logits tensor in target vocabulary space, or None if no sampling
        """
        # Generate logits in draft vocabulary space
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)

        # Handle vocabulary mapping if draft and target differ
        if self.draft_id_to_target_id is None:
            # No vocabulary mapping needed - draft and target use same vocabulary
            target_vocab_size = getattr(self.config, "vocab_size",
                                        self.config.draft_vocab_size)
            assert logits.shape[1] == target_vocab_size, (
                f"Expected logits to have shape (*, {target_vocab_size}), "
                f"but got {logits.shape}")
            return logits

        # Apply draft-to-target vocabulary mapping
        # Create base indices for draft vocabulary
        base_indices = torch.arange(self.config.draft_vocab_size,
                                    device=logits.device)

        # Map to target vocabulary indices
        target_indices = base_indices + self.draft_id_to_target_id

        # Create expanded logits tensor in target vocabulary space
        target_vocab_size = getattr(self.config, "vocab_size",
                                    max(target_indices) + 1)
        expanded_logits = logits.new_full((logits.shape[0], target_vocab_size),
                                          float('-inf'))

        # Map draft logits to target positions
        expanded_logits[:, target_indices] = logits
        return expanded_logits

    def combine_hidden_states(self,
                              hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Combine auxiliary hidden states from target model.
        
        Eagle3 receives auxiliary hidden states from multiple layers of the
        target model. This method combines them into a single representation
        suitable for draft token generation.
        
        Args:
            hidden_states: Concatenated auxiliary hidden states from target
                          model layers, typically from early, middle, and
                          late layers
                          
        Returns:
            Combined hidden state representation for draft generation
        """
        return self.model.fc(hidden_states)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        """
        Get input embeddings with optional multimodal support.
        
        This method provides embeddings for input tokens with support for
        multimodal inputs when available. Currently, multimodal support
        is limited but the interface is provided for future compatibility.
        
        Args:
            input_ids: Input token IDs
            multimodal_embeddings: Optional multimodal embeddings
            
        Returns:
            Input embeddings tensor
        """
        inputs_embeds = self.model.get_input_embeddings(input_ids)

        if multimodal_embeddings is not None:
            # Merge multimodal embeddings if available
            # This maintains compatibility with Llama4's multimodal capabilities
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                getattr(self.config, "image_token_index", None),
            )

        return inputs_embeds

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> None:
        """
        Load Eagle3 model weights with specific mapping rules.
        
        This method handles the complex weight loading requirements for Eagle3
        models, including vocabulary mapping tensors and layer name translations.
        
        Args:
            weights: Iterable of (parameter_name, tensor) pairs from checkpoint
        """
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False

        # Process and filter weights according to Eagle3 conventions
        for name, loaded_weight in weights:
            # Skip target-to-draft mappings (not used in current implementation)
            if "t2d" in name:
                continue

            # Handle draft-to-target vocabulary mapping
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                # Prefix non-lm_head weights with "model."
                name = "model." + name

            if "embed_tokens" in name:
                includes_embed_tokens = True

            model_weights[name] = loaded_weight

        # Configure weight loader with conditional skipping
        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")

        # Use AutoWeightsLoader for robust weight loading
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
