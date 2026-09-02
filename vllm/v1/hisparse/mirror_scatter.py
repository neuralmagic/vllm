# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-resolved host-mirror scatter for async speculative decoding.

Under async scheduling the CPU cannot know a request's actual start position
when mirror work is enqueued (the previous step's rejection sampling may not
have run yet), so destination rows must be resolved on the GPU at execution
time from the per-token ``positions`` tensor, which is exact in stream order.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _mirror_scatter_kernel(
    staging_ptr,  # [num_rows, row_bytes] int8, this forward's rows in order
    host_ptr,  # [host_rows, row_bytes] int8 UVA view of the pinned pool
    positions_ptr,  # [num_rows] int64, actual token position per row
    req_of_row_ptr,  # [num_rows] int32, batch request index per row
    window_starts_ptr,  # [num_reqs] int64, first token position of the window
    window_offsets_ptr,  # [num_reqs + 1] int64, prefix sum of window rows
    dst_rows_ptr,  # [total_window_rows] int64, host row per window position
    row_bytes,
    staging_stride,
    host_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.load(req_of_row_ptr + row)
    position = tl.load(positions_ptr + row)
    window_start = tl.load(window_starts_ptr + req)
    rel = position - window_start
    win_begin = tl.load(window_offsets_ptr + req)
    win_end = tl.load(window_offsets_ptr + req + 1)
    in_window = (rel >= 0) & (win_begin + rel < win_end)
    dst_row = tl.load(dst_rows_ptr + win_begin + rel, mask=in_window, other=-1)
    if dst_row >= 0:
        src_base = staging_ptr + row.to(tl.int64) * staging_stride
        dst_base = host_ptr + dst_row * host_stride
        for start in range(0, row_bytes, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            mask = offs < row_bytes
            vals = tl.load(src_base + offs, mask=mask)
            tl.store(dst_base + offs, vals, mask=mask)


def scatter_mirror_rows(
    staging: torch.Tensor,
    host_uva: torch.Tensor,
    positions: torch.Tensor,
    req_of_row: torch.Tensor,
    window_starts: torch.Tensor,
    window_offsets: torch.Tensor,
    dst_rows: torch.Tensor,
    num_rows: int,
) -> None:
    """Copy ``num_rows`` staged rows to their actual-position host rows.

    Rows whose position falls outside the request's mirror window (or maps to
    a -1 destination) are skipped; a later step re-mirrors them once the
    window catches up.
    """
    if num_rows == 0:
        return
    row_bytes = staging.shape[-1] * staging.element_size()
    assert host_uva.shape[-1] * host_uva.element_size() == row_bytes
    _mirror_scatter_kernel[(num_rows,)](
        staging.view(torch.int8),
        host_uva.view(torch.int8),
        positions,
        req_of_row,
        window_starts,
        window_offsets,
        dst_rows,
        row_bytes,
        staging.stride(0) * staging.element_size(),
        host_uva.stride(0) * host_uva.element_size(),
        BLOCK_SIZE=1024,
    )
