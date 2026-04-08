# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform

FLASHINFER_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="FlashInfer MLA Requires compute capability of 10 or above.",
        allow_module_level=True,
    )
else:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla


_FP8_DTYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
}


def _finite_check_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float16) if t.dtype in _FP8_DTYPES else t


@pytest.mark.parametrize("max_seq_len", [128, 256, 512, 1024, 2048, 4096])
def test_flashinfer_mla_decode_padding_rows_not_updated(max_seq_len: int):
    """Regression test: kernel must not write into padding rows."""
    torch.set_default_device("cuda")
    torch.manual_seed(42)

    dtype = torch.float8_e4m3fn
    block_size = 64
    num_heads = 128
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = kv_lora_rank + qk_rope_head_dim

    bs = 8
    q_len_per_request = 1
    seq_lens_tensor = torch.tensor([3, 3, 3, 3, 3, 0, 0, 0], dtype=torch.int32)

    # Build a realistic block-table layout:
    # - table width follows max_seq_len,
    # - active rows get unique page IDs for required blocks,
    # - unused slots stay -1.
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = torch.full((bs, max_blocks_per_seq), -1, dtype=torch.int32)
    num_pages = bs * max_blocks_per_seq
    page_ids = torch.randperm(num_pages, dtype=torch.int32)
    cursor = 0
    for req_idx, seq_len in enumerate(seq_lens_tensor.tolist()):
        num_blocks = (seq_len + block_size - 1) // block_size
        if num_blocks > 0:
            block_tables[req_idx, :num_blocks] = page_ids[cursor : cursor + num_blocks]
            cursor += num_blocks

    kv_cache = torch.randn(num_pages, block_size, qk_head_dim).to(dtype)
    q = torch.randn(bs, q_len_per_request, num_heads, qk_head_dim).to(dtype)
    assert torch.isfinite(_finite_check_tensor(kv_cache)).all(), (
        "kv_cache contains NaN/Inf before test."
    )
    assert torch.isfinite(_finite_check_tensor(q)).all(), (
        "q contains NaN/Inf before test."
    )

    workspace_buffer = torch.zeros(
        FLASHINFER_WORKSPACE_BUFFER_SIZE,
        dtype=torch.uint8,
        device=q.device,
    )

    out = torch.zeros(
        (bs, num_heads, kv_lora_rank),
        dtype=torch.bfloat16,
        device=q.device,
    )
    padding_rows = seq_lens_tensor == 0
    padding_expected = torch.ones_like(out[padding_rows])

    for i in range(1000):
        out[padding_rows] = 1
        out_ans = trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv_cache.unsqueeze(1),
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_tensor,
            max_seq_len=max_seq_len,
            out=out,
        )
        assert torch.equal(out_ans[padding_rows], padding_expected), (
            f"Kernel updated padding rows (seq_lens == 0) at iteration {i}."
        )
