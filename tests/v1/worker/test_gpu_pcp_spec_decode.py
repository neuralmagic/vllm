# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.pcp_manager import PCPManager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_partition_reuses_gpu_cursor_for_replicated_spec_decode():
    device = torch.device("cuda")
    global_buffers = InputBuffers(max_num_reqs=1, max_num_tokens=4, device=device)
    global_batch = InputBatch.make_dummy(
        num_reqs=1,
        num_tokens=4,
        input_buffers=global_buffers,
    )
    global_batch.num_draft_tokens = 3
    global_batch.num_draft_tokens_per_req = np.array([3], dtype=np.int32)
    global_batch.num_computed_tokens_np[:] = 20
    global_batch.prefill_len_np[:] = 8
    global_batch.num_computed_prefill_tokens_np[:] = 8
    global_batch.positions.copy_(torch.arange(10, 14, device=device))
    global_batch.seq_lens.fill_(14)

    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=device,
        req_states=SimpleNamespace(),
        max_num_reqs=1,
        max_num_tokens=4,
    )
    local_batch = manager.partition_batch(global_batch)

    assert local_batch.num_reqs == 1
    assert local_batch.num_scheduled_tokens.tolist() == [4]
    torch.testing.assert_close(
        local_batch.positions,
        torch.arange(10, 14, device=device),
    )
    torch.testing.assert_close(
        local_batch.seq_lens,
        torch.tensor([14], dtype=torch.int32, device=device),
    )
    assert local_batch.num_computed_tokens_np.tolist() == [20]


def test_replicated_verification_skips_pcp_restore(monkeypatch):
    device = torch.device("cpu")
    global_buffers = InputBuffers(max_num_reqs=1, max_num_tokens=8, device=device)
    global_batch = InputBatch.make_dummy(
        num_reqs=1,
        num_tokens=4,
        input_buffers=global_buffers,
    )
    global_batch.num_draft_tokens = 3
    global_batch.num_draft_tokens_per_req = np.array([3], dtype=np.int32)
    global_batch.num_computed_tokens_np[:] = 20
    global_batch.prefill_len_np[:] = 8
    global_batch.num_computed_prefill_tokens_np[:] = 8
    global_batch.positions.copy_(torch.arange(10, 14, device=device))
    global_batch.seq_lens.fill_(14)

    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=device,
        req_states=SimpleNamespace(),
        max_num_reqs=1,
        max_num_tokens=8,
    )
    local_batch = manager.partition_batch(global_batch, padded_num_tokens=8)

    assert manager._replicated_verification
    assert local_batch.num_draft_tokens == 3
    assert local_batch.num_tokens_after_padding == 8
    torch.testing.assert_close(
        local_batch.positions[:4], torch.arange(10, 14, device=device)
    )
    torch.testing.assert_close(
        local_batch.positions[4:], torch.zeros(4, dtype=torch.int64, device=device)
    )
    assert local_batch.is_padding.tolist() == [False] * 4 + [True] * 4

    def fail_if_called():
        raise AssertionError("replicated verification must not use PCP all-gather")

    monkeypatch.setattr(pcp_manager_module, "get_pcp_group", fail_if_called)
    hidden_states = torch.randn(8, 4, device=device)
    assert manager.restore_hidden_states(hidden_states) is hidden_states
    restored_buffer = manager.restore_hidden_state_buffer(hidden_states)
    assert restored_buffer.shape == hidden_states.shape
    torch.testing.assert_close(restored_buffer, hidden_states)

    expected_slots = torch.arange(8, dtype=torch.int64).reshape(1, 8)

    def compute_slot_mappings(*_args, out, **_kwargs):
        out[:, :8].copy_(expected_slots)
        return out[:, :8]

    local_block_tables = (torch.empty(1, 1, dtype=torch.int32),)
    manager._block_tables = SimpleNamespace(
        gather_block_tables=lambda *_args, **_kwargs: local_block_tables,
        compute_slot_mappings=compute_slot_mappings,
    )
    manager._local_block_tables = local_block_tables
    manager._local_block_table_ptrs = torch.empty(1, dtype=torch.uint64)
    manager._global_batch_slot_mappings = torch.empty(1, 8, dtype=torch.int64)
    manager._gathered_kv_slot_mappings = torch.empty(1, 32, dtype=torch.int64)
    monkeypatch.setattr(
        PCPManager,
        "direct_kv_enabled",
        property(lambda _self: True),
    )

    # Graph capture obtains this PCP buffer from get_dummy_slot_mappings().
    captured_slot_mappings = manager.get_dummy_slot_mappings(8)
    captured_ptr = captured_slot_mappings.data_ptr()
    _, runtime_slot_mappings = manager.prepare_attn(local_batch)

    # Replicated verification must update the captured allocation rather than
    # return the otherwise-correct global scratch buffer at a new address.
    assert runtime_slot_mappings.data_ptr() == captured_ptr
    torch.testing.assert_close(runtime_slot_mappings, expected_slots)


def test_restore_hidden_states_appends_zero_graph_padding(monkeypatch):
    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
        use_mtp=True,
    )
    manager._global_batch = SimpleNamespace(
        num_tokens=5,
        num_tokens_after_padding=8,
    )
    restored = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    manager._hidden_restore_idx = torch.arange(5)
    monkeypatch.setattr(
        pcp_manager_module,
        "get_pcp_group",
        lambda: SimpleNamespace(all_gather=lambda *_args, **_kwargs: restored),
    )

    actual = manager.restore_hidden_states(torch.empty(0))

    assert actual.shape == (8, 2)
    torch.testing.assert_close(actual[:5], restored)
    torch.testing.assert_close(actual[5:], torch.zeros(3, 2))
