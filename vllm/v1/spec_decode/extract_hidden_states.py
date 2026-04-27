# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


class ExtractHiddenStatesProposer:
    """Proposer that extracts hidden states and passes them directly to the
    KV connector, bypassing the KV cache entirely.

    Instead of loading a draft model with CacheOnly attention layers and
    routing hidden states through the KV cache pipeline, this proposer
    stacks the target model's intermediate hidden states and hands them
    to the connector's ``save_hidden_states()`` method.  This avoids
    any interaction with the KV cache group/coordinator machinery, making
    it compatible with hybrid models (e.g. Qwen3.5) without modifications
    to core KV cache code.
    """

    def __init__(self, vllm_config: VllmConfig, device):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.num_speculative_tokens == 1

        self.vllm_config = vllm_config
        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        self.hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        layer_ids = getattr(self.hf_config, "eagle_aux_hidden_state_layer_ids", None)
        if not layer_ids:
            raise ValueError(
                "eagle_aux_hidden_state_layer_ids must be set in the draft "
                "model config for extract_hidden_states method"
            )
        self.num_hidden_states = len(layer_ids)
        self.hidden_size = vllm_config.model_config.get_hidden_size()

        # No draft model, no attention metadata, no CUDAGraphs, no buffers.
        self.model = None

    def propose(
        self,
        sampled_token_ids: torch.Tensor,
        target_hidden_states: list[torch.Tensor],
        common_attn_metadata: CommonAttentionMetadata,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        """Stack hidden states and pass them to the KV connector directly."""
        assert isinstance(target_hidden_states, list)

        # [num_tokens, num_hidden_states, hidden_size]
        stacked = torch.stack(target_hidden_states, dim=1)

        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )

        if has_kv_transfer_group():
            connector = get_kv_transfer_group()
            if hasattr(connector, "save_hidden_states"):
                qsl = common_attn_metadata.query_start_loc_cpu
                tokens_per_req = (qsl[1:] - qsl[:-1]).tolist()
                connector.save_hidden_states(stacked, tokens_per_req)

        return sampled_token_ids[:, :1]

    def load_model(self, target_model) -> None:
        """No draft model to load in bypass mode."""

    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """No draft model to warm up."""

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        """No CUDAGraphs for the proposer."""

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """No KV cache groups to validate."""

    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare next token IDs for speculative decoding.

        Since num_speculative_tokens == 1, sampled_token_ids has shape
        (batch_size, 1). For each request we either use the sampled token
        (if valid and not discarded) or a backup token from the request state.
        """
        num_reqs = gpu_input_batch.num_reqs
        device = sampled_token_ids.device

        seq_lens_list = (gpu_input_batch.num_tokens_no_spec[:num_reqs] - 1).tolist()
        backup_tokens_gpu = torch.tensor(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(seq_lens_list[i])
                for i in range(num_reqs)
            ],
            dtype=torch.int32,
            device=device,
        )

        assert discard_request_mask.dtype == torch.bool

        sampled = sampled_token_ids[:, 0]
        is_valid = (sampled >= 0) & (sampled < gpu_input_batch.vocab_size)
        valid_sampled_tokens_count = is_valid.to(torch.int32)

        use_sampled = is_valid & ~discard_request_mask[:num_reqs]
        next_token_ids = torch.where(
            use_sampled, sampled.to(torch.int32), backup_tokens_gpu
        )

        return next_token_ids, valid_sampled_tokens_count
