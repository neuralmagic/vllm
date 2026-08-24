# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from unittest.mock import Mock
from vllm.v1.worker.gpu import cudagraph_utils


def _copy_to_cpu(value, out=None, device=None):
    tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    if out is not None:
        return out.copy_(tensor)
    return tensor


def test_replicated_decode_piecewise_graph_padding(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)

    segments_by_rank, per_rank_num_tokens = manager._build_batch_layout(
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        num_computed_tokens=np.full(3, 16, dtype=np.int32),
        is_prefilling=np.zeros(3, dtype=np.bool_),
        query_start_loc_np=np.arange(4, dtype=np.int32),
        padded_num_tokens=4,
    )

    assert per_rank_num_tokens == [3, 3]
    request_indices = [
        [segment.global_batch_req_idx for segment in rank] for rank in segments_by_rank
    ]
    assert request_indices == [[0, 1, 2], [0, 1, 2]]
    assert torch.equal(manager._hidden_restore_idx, torch.tensor([0, 1, 2]))
    assert torch.equal(
        manager._padded_gather_idx,
        torch.tensor([0, 1, 2, 0, 0, 1, 2, 0]),
    )
    assert torch.equal(
        manager._gathered_kv_write_mask,
        torch.tensor([True, True, True, False, False, False, False, False]),
    )


def test_input_buffers_are_exposed_for_cudagraph_capture():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=8,
    )

    assert manager.input_buffers is manager._input_buffers
    assert manager.input_buffers.input_ids.shape == (8,)
    assert manager.input_buffers.positions.shape == (8,)
    assert manager.input_buffers.is_padding.shape == (8,)


@pytest.mark.parametrize(
    ("pcp_world_size", "num_scheduled_tokens", "is_prefilling", "expected"),
    [
        (2, [8], [True], 4),
        (2, [7], [True], 4),
        (2, [3], [False], 3),
        (2, [3, 8], [False, True], 7),
        (4, [2, 9], [False, True], 5),
    ],
)
def test_num_tokens_for_dispatch_uses_largest_pcp_rank(
    pcp_world_size, num_scheduled_tokens, is_prefilling, expected
):
    manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=0,
        device=torch.device("cpu"),
    )

    actual = manager.get_num_tokens_for_dispatch(
        np.asarray(num_scheduled_tokens, dtype=np.int32),
        np.asarray(is_prefilling, dtype=np.bool_),
    )

    assert actual == expected


def test_graph_padding_cannot_be_smaller_than_largest_pcp_rank(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)

    with pytest.raises(ValueError, match="smaller than the largest rank-local batch"):
        manager._build_batch_layout(
            num_scheduled_tokens=np.ones(3, dtype=np.int32),
            num_computed_tokens=np.full(3, 16, dtype=np.int32),
            is_prefilling=np.zeros(3, dtype=np.bool_),
            query_start_loc_np=np.arange(4, dtype=np.int32),
            padded_num_tokens=2,
        )


def test_sparse_mla_accepts_piecewise_cudagraphs():
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(index_topk=2048),
        ),
        lora_config=None,
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE),
    )

    PCPManager.validate_config(vllm_config, supports_mm_inputs=False)


def test_cudagraph_capture_uses_pcp_dummy_attention(monkeypatch):
    input_batch = SimpleNamespace(dcp_local_seq_lens=None)
    input_block_tables = (torch.empty(0),)
    slot_mappings = torch.empty(2, 16, dtype=torch.int64)
    prepare_dummy_attn = Mock(return_value=(input_block_tables, slot_mappings))
    model_state = SimpleNamespace(prepare_attn=Mock(return_value={"layer": "metadata"}))
    kv_cache_config = SimpleNamespace()

    monkeypatch.setattr(
        cudagraph_utils.InputBatch,
        "make_dummy",
        staticmethod(lambda *args, **kwargs: input_batch),
    )
    monkeypatch.setattr(
        cudagraph_utils,
        "build_slot_mappings_by_layer",
        lambda mappings, config: {"layer": mappings},
    )

    state = cudagraph_utils.prepare_inputs_to_capture(
        num_reqs=8,
        num_tokens=16,
        model_state=model_state,
        input_buffers=SimpleNamespace(),
        block_tables=SimpleNamespace(cp_size=1),
        attn_groups=[],
        kv_cache_config=kv_cache_config,
        full_cudagraph=False,
        prepare_dummy_attn=prepare_dummy_attn,
    )

    prepare_dummy_attn.assert_called_once_with(input_batch)
    assert state.slot_mappings["layer"] is slot_mappings
    model_state.prepare_attn.assert_called_once_with(
        input_batch,
        CUDAGraphMode.NONE,
        input_block_tables,
        slot_mappings,
        [],
        kv_cache_config,
        for_capture=False,
    )
