# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _create_side_stream(device: torch.device) -> torch.Stream:
    return torch.Stream(device=device)


def _create_event() -> torch.Event:
    return torch.Event()


@dataclass
class SparseMLAIndexGroup:
    """Layers that consume one sparse-indexer result."""

    logical_topk_indices: torch.Tensor
    physical_topk_indices: torch.Tensor
    valid_topk_counts: torch.Tensor
    side_stream: torch.Stream
    logical_topk_ready: torch.Event
    physical_topk_ready: torch.Event
    has_indexer: bool
    num_layers: int = 0
    hisparse_group: Any | None = None
    physical_layout_key: tuple[int, int, int] | None = None


class SparseMLAIndexGroupBuilder:
    """Assign sparse MLA layers to their index-producing layer."""

    def __init__(
        self, logical_topk_indices: torch.Tensor, max_decode_rows: int | None = None
    ) -> None:
        self.logical_topk_indices = logical_topk_indices
        self.max_decode_rows = (
            logical_topk_indices.shape[0]
            if max_decode_rows is None
            else max_decode_rows
        )
        self.current_group: SparseMLAIndexGroup | None = None

    def register_layer(
        self, is_index_producing_layer: bool
    ) -> tuple[SparseMLAIndexGroup, int]:
        if is_index_producing_layer or self.current_group is None:
            physical_topk_indices = torch.empty(
                (self.max_decode_rows, self.logical_topk_indices.shape[1]),
                dtype=self.logical_topk_indices.dtype,
                device=self.logical_topk_indices.device,
            )
            self.current_group = SparseMLAIndexGroup(
                logical_topk_indices=self.logical_topk_indices,
                physical_topk_indices=physical_topk_indices,
                valid_topk_counts=torch.empty(
                    self.max_decode_rows,
                    dtype=torch.int32,
                    device=self.logical_topk_indices.device,
                ),
                side_stream=_create_side_stream(self.logical_topk_indices.device),
                logical_topk_ready=_create_event(),
                physical_topk_ready=_create_event(),
                has_indexer=is_index_producing_layer,
            )
        group = self.current_group
        group_index = group.num_layers
        group.num_layers += 1
        return group, group_index


def get_sparse_mla_index_group_max_rows(vllm_config: VllmConfig) -> int:
    max_query_len = 1
    speculative_config = vllm_config.speculative_config
    if (
        speculative_config is not None
        and speculative_config.num_speculative_tokens is not None
    ):
        max_query_len += speculative_config.num_speculative_tokens * (
            2 if speculative_config.parallel_drafting else 1
        )
    scheduler_config = vllm_config.scheduler_config
    return min(
        scheduler_config.max_num_batched_tokens,
        scheduler_config.max_num_seqs * max_query_len,
    )
