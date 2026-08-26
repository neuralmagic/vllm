# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

_PrecomputeCall = tuple[
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
    bool,
]


class _DraftModel:
    def __init__(self) -> None:
        self.precompute_call: _PrecomputeCall | None = None

    @staticmethod
    def get_draft_kv_cache_layer_names() -> list[str]:
        return ["draft.0", "draft.1"]

    @staticmethod
    def combine_hidden_states(states: torch.Tensor) -> torch.Tensor:
        return states + 1

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        positions: torch.Tensor,
        slot_mappings: list[torch.Tensor],
        *,
        publish_to_pcp: bool,
    ) -> None:
        self.precompute_call = (
            context_states.clone(),
            positions.clone(),
            [mapping.clone() for mapping in slot_mappings],
            publish_to_pcp,
        )


class _Qwen3Model:
    def __init__(self) -> None:
        self.publish_to_pcp: bool | None = None

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor,
        *,
        publish_to_pcp: bool,
    ) -> None:
        self.publish_to_pcp = publish_to_pcp


def test_qwen3_dflash_forwards_pcp_publication() -> None:
    model = SimpleNamespace(model=_Qwen3Model())
    DFlashQwen3ForCausalLM.precompute_and_store_context_kv(
        model,
        torch.empty(1, 1),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        publish_to_pcp=True,
    )
    assert model.model.publish_to_pcp is True


def test_precompute_pcp_context_kv_uses_local_rows_and_marks_step() -> None:
    speculator = object.__new__(DFlashSpeculator)
    speculator._pcp_context_kv_precomputed = False
    speculator.model = _DraftModel()
    input_batch = SimpleNamespace(
        num_tokens=2,
        positions=torch.tensor([7, 8, 99]),
    )
    aux_hidden_states = [
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[3.0], [4.0]]),
    ]
    slot_mappings = {
        "draft.0": torch.tensor([10, 11, 12]),
        "draft.1": torch.tensor([20, 21, 22]),
    }

    speculator.precompute_pcp_context_kv(input_batch, aux_hidden_states, slot_mappings)

    precompute_call = speculator.model.precompute_call
    assert precompute_call is not None
    context, positions, mappings, publish = precompute_call
    torch.testing.assert_close(context, torch.tensor([[2.0, 4.0], [3.0, 5.0]]))
    assert torch.equal(positions, torch.tensor([7, 8]))
    assert [mapping.tolist() for mapping in mappings] == [[10, 11], [20, 21]]
    assert publish is True
    assert speculator._pcp_context_kv_precomputed is True

    with pytest.raises(RuntimeError, match="already precomputed"):
        speculator.precompute_pcp_context_kv(
            input_batch, aux_hidden_states, slot_mappings
        )
