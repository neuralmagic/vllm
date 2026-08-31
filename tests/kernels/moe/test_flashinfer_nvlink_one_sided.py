# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import patch

from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    flashinfer_nvlink_one_sided,
)


def test_get_local_sizes_without_dp_metadata():
    context = SimpleNamespace(dp_metadata=None)
    with patch(
        "vllm.model_executor.layers.fused_moe.prepare_finalize."
        "flashinfer_nvlink_one_sided.get_forward_context",
        return_value=context,
    ):
        assert flashinfer_nvlink_one_sided.get_local_sizes() is None


def test_get_local_sizes_with_dp_metadata():
    metadata = SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: [3, 5])
    context = SimpleNamespace(dp_metadata=metadata)
    with patch(
        "vllm.model_executor.layers.fused_moe.prepare_finalize."
        "flashinfer_nvlink_one_sided.get_forward_context",
        return_value=context,
    ):
        assert flashinfer_nvlink_one_sided.get_local_sizes() == [3, 5]


def test_runtime_max_tokens_covers_pcp_padding():
    get_max = flashinfer_nvlink_one_sided._runtime_max_tokens_per_rank
    assert get_max(7, [3, 5]) == 7
    assert get_max(3, [3, 5]) == 5
    assert get_max(7, None) == 7
