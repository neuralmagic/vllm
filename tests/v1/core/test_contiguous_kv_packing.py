# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for contiguous KV cache packing in _get_kv_cache_config_deepseek_v4."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.core.kv_cache_utils import _get_kv_cache_config_deepseek_v4
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)


def _make_mla_spec(page_size: int, block_size: int = 256) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        page_size_padded=page_size,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
        alignment=576,
    )


def _make_swa_spec(page_size: int, block_size: int = 64) -> SlidingWindowMLASpec:
    spec = SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        page_size_padded=page_size,
        sliding_window=128,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
        alignment=576,
    )
    object.__setattr__(spec, "page_size_padded", page_size)
    return spec


def _make_groups(n_c4, n_c128, n_swa):
    PS_C4_MLA = 37440
    PS_C4_IDX = 8640
    PS_C128 = 1728
    PS_SWA = 149760

    mla_specs = {}
    for i in range(n_c4):
        mla_specs[f"c4_mla.{i}"] = _make_mla_spec(PS_C4_MLA)
        mla_specs[f"c4_idx.{i}"] = _make_mla_spec(PS_C4_IDX)
    for i in range(n_c128):
        mla_specs[f"c128_mla.{i}"] = _make_mla_spec(PS_C128)

    mla_group = KVCacheGroupSpec(
        layer_names=list(mla_specs.keys()),
        kv_cache_spec=UniformTypeKVCacheSpecs(block_size=256, kv_cache_specs=mla_specs),
    )

    swa_specs = {}
    for i in range(n_swa):
        swa_specs[f"swa.{i}"] = _make_swa_spec(PS_SWA)

    swa_group = KVCacheGroupSpec(
        layer_names=list(swa_specs.keys()),
        kv_cache_spec=UniformTypeKVCacheSpecs(block_size=256, kv_cache_specs=swa_specs),
    )

    return [mla_group, swa_group]


def _mock_vllm_config():
    config = MagicMock()
    config.cache_config.num_gpu_blocks_override = None
    config.scheduler_config.max_num_batched_tokens = 64
    config.scheduler_config.max_num_seqs = 4
    config.model_config.max_model_len = 1024
    return config


def _run(n_c4=3, n_c128=2, n_swa=5, mem=100 * 1024 * 1024):
    groups = _make_groups(n_c4, n_c128, n_swa)
    return _get_kv_cache_config_deepseek_v4(_mock_vllm_config(), groups, mem)


def _specs_by_layer(groups):
    specs = {}
    for group in groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs.update(group.kv_cache_spec.kv_cache_specs)
    return specs


class TestInterleavedPacking:
    def test_all_tensors_have_block_stride(self):
        _, tensors = _run()
        for t in tensors:
            assert t.block_stride > 0

    def test_tensors_share_size_within_backing(self):
        _, tensors = _run()
        sizes_by_backing: dict[int, set[int]] = {}
        for t in tensors:
            sizes_by_backing.setdefault(t.backing_id, set()).add(t.size)

        assert len(sizes_by_backing) == 1
        for sizes in sizes_by_backing.values():
            assert len(sizes) == 1
            assert sizes.pop() > 0

    def test_offsets_within_one_block(self):
        _, tensors = _run()
        for t in tensors:
            assert t.offset < t.block_stride

    def test_all_layers_accounted_for(self):
        n_c4, n_c128, n_swa = 5, 4, 7
        _, tensors = _run(n_c4=n_c4, n_c128=n_c128, n_swa=n_swa)
        all_names = set()
        for t in tensors:
            all_names.update(t.shared_by)
        expected = n_c4 * 2 + n_c128 + n_swa
        assert len(all_names) == expected

    def test_mla_and_swa_share_hma_tuple_storage(self):
        _, tensors = _run(n_c4=3, n_swa=3)
        backing_by_layer = {}
        shared_slots = []
        for t in tensors:
            if any(name.startswith("swa.") for name in t.shared_by) and any(
                name.startswith(("c4_", "c128_")) for name in t.shared_by
            ):
                shared_slots.append(t.shared_by)
            for name in t.shared_by:
                backing_by_layer[name] = t.backing_id

        linear_backings = {
            backing
            for name, backing in backing_by_layer.items()
            if name.startswith(("c4_", "c128_"))
        }
        swa_backings = {
            backing
            for name, backing in backing_by_layer.items()
            if name.startswith("swa.")
        }
        assert len(linear_backings) == 1
        assert len(swa_backings) == 1
        assert linear_backings == swa_backings
        assert shared_slots

    def test_packed_backing_uses_shared_hma_blocks(self):
        config = _mock_vllm_config()
        groups = _make_groups(n_c4=3, n_c128=2, n_swa=5)
        num_blocks, tensors = _get_kv_cache_config_deepseek_v4(
            config, groups, 100 * 1024 * 1024
        )

        rows_by_backing = {t.backing_id: t.size // t.block_stride for t in tensors}

        assert set(rows_by_backing.values()) == {num_blocks}

    def test_strided_views_are_independent(self):
        groups = _make_groups(3, 2, 5)
        specs = _specs_by_layer(groups)
        num_blocks, tensors = _get_kv_cache_config_deepseek_v4(
            _mock_vllm_config(), groups, 100 * 1024 * 1024
        )
        backings = {
            t.backing_id: torch.zeros(t.size, dtype=torch.uint8) for t in tensors
        }
        views = []
        for t in tensors:
            page_size = specs[t.shared_by[0]].page_size_bytes
            backing_blocks = t.size // t.block_stride
            v = torch.as_strided(
                backings[t.backing_id],
                size=(backing_blocks, page_size),
                stride=(t.block_stride, 1),
                storage_offset=t.offset,
            )
            views.append(v)

        for i, v in enumerate(views):
            v.fill_(i + 1)

        for i, v in enumerate(views):
            assert (v == i + 1).all(), f"View {i} was corrupted"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
