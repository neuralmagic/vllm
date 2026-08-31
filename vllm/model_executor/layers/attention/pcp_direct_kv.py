# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP direct-final KV publication through PyTorch symmetric memory."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed.device_communicators.symm_mem import (
    SymmMemPeerAllocation,
    allocate_symm_mem_peer,
    symm_mem_available,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_MAX_FENCE_SPINS = 100_000_000


@triton.jit
def _gather_sharded_peer_cache_kernel(
    peer_ptrs,
    dst,
    block_table,
    cu_seq_lens,
    token_to_seq,
    seq_starts,
    scale,
    block_table_stride: tl.constexpr,
    cache_block_size: tl.constexpr,
    cache_block_stride_bytes: tl.constexpr,
    cache_token_stride_bytes: tl.constexpr,
    row_dim: tl.constexpr,
    world_size: tl.constexpr,
    interleave_size: tl.constexpr,
    packed_ds_mla: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    req_idx = tl.load(token_to_seq + token_idx)
    chunk_offset = token_idx - tl.load(cu_seq_lens + req_idx)
    global_pos = tl.load(seq_starts + req_idx) + chunk_offset

    owner = (global_pos // interleave_size) % world_size
    virtual_block_size = world_size * interleave_size
    local_pos = (global_pos // virtual_block_size) * interleave_size + (
        global_pos % interleave_size
    )
    block_idx = local_pos // cache_block_size
    block_number = tl.load(block_table + req_idx * block_table_stride + block_idx)
    block_offset = local_pos % cache_block_size

    peer_base = tl.load(peer_ptrs + owner).to(tl.pointer_type(tl.uint8))
    mask = cols < row_dim
    entry_base = (
        peer_base
        + block_number * cache_block_stride_bytes
        + block_offset * cache_token_stride_bytes
    )
    if packed_ds_mla:
        nope_mask = mask & (cols < 512)
        raw = tl.load(entry_base + cols, mask=nope_mask, other=0)
        value = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_ptr = (entry_base + 512).to(tl.pointer_type(tl.float32))
        tile_scale = tl.load(scale_ptr + cols // 128, mask=nope_mask, other=1.0)
        value *= tile_scale

        rope_offset = cols - 512
        rope_mask = mask & (cols >= 512)
        rope_ptr = (entry_base + 528).to(tl.pointer_type(tl.bfloat16))
        rope = tl.load(rope_ptr + rope_offset, mask=rope_mask, other=0.0)
        value = tl.where(rope_mask, rope.to(tl.float32), value)
    else:
        raw = tl.load(entry_base + cols, mask=mask, other=0)
        value = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        value *= tl.load(scale).to(tl.float32)
    tl.store(dst + token_idx * row_dim + cols, value, mask=mask)


@triton.jit
def _trap_if_nonzero(value):
    # Unconditional PTX trap. tl.device_assert is a no-op unless TRITON_DEBUG=1.
    tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.ne.s32 %p0, $1, 0;
            @%p0 trap;
        }
        """,
        "=r, r",
        [value.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _peer_cache_fence_kernel(
    peer_ptrs,
    local_signal_ptr,
    epoch_ptr,
    source_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_SPINS: tl.constexpr,
):
    # Keep the epoch on device so CUDA graph replay advances it on every launch.
    epoch = tl.atomic_add(epoch_ptr, 1, sem="relaxed", scope="gpu") + 1
    parity = epoch & 1

    # System-scope release publishes this rank's preceding peer-cache writes.
    for destination_rank in tl.static_range(0, world_size):
        dest_base = tl.load(peer_ptrs + destination_rank).to(tl.pointer_type(tl.int32))
        tl.atomic_xchg(
            dest_base + parity * world_size + source_rank,
            epoch,
            sem="release",
            scope="sys",
        )

    # System-scope acquire waits until every peer has published the same epoch.
    peer_rank = tl.arange(0, BLOCK_SIZE)
    mask = peer_rank < world_size
    signal_ptr = local_signal_ptr + parity * world_size + peer_rank
    observed = tl.atomic_add(signal_ptr, 0, mask=mask, sem="acquire", scope="sys")
    pending = tl.max(tl.where(mask & (observed != epoch), 1, 0))
    spins = 0
    while (pending != 0) & (spins < MAX_SPINS):
        observed = tl.atomic_add(signal_ptr, 0, mask=mask, sem="acquire", scope="sys")
        pending = tl.max(tl.where(mask & (observed != epoch), 1, 0))
        spins += 1
    _trap_if_nonzero(pending)


@triton.jit
def _copy_cache_rows_to_peers_kernel(
    peer_ptrs,
    slot_mapping,
    row_nbytes,
    source_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    byte_offset = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    slot = tl.load(slot_mapping + token_idx)
    mask = (slot >= 0) & (byte_offset < row_nbytes)
    source_base = tl.load(peer_ptrs + source_rank).to(tl.pointer_type(tl.uint8))
    source = source_base + slot * row_nbytes + byte_offset
    value = tl.load(source, mask=mask, other=0)
    for peer in tl.static_range(0, world_size):
        if peer != source_rank:
            destination_base = tl.load(peer_ptrs + peer).to(tl.pointer_type(tl.uint8))
            destination = destination_base + slot * row_nbytes + byte_offset
            tl.store(destination, value, mask=mask)


class PCPPeerCacheFence:
    """Device-epoch release/acquire publication for one PCP group."""

    def __init__(self, group: ProcessGroup, device: torch.device) -> None:
        self._group = group
        self._world_size = group.size()
        self._rank = group.rank()
        self._epoch = torch.zeros((1,), dtype=torch.int32, device=device)
        self._allocation = allocate_symm_mem_peer(
            (2, self._world_size),
            dtype=torch.int32,
            device=device,
            group=group,
        )
        self._allocation.storage.zero_()
        torch.accelerator.synchronize()
        dist.barrier(group=group)

    def __call__(self) -> None:
        _peer_cache_fence_kernel[(1,)](
            self._allocation.peer_ptrs,
            self._allocation.storage,
            self._epoch,
            source_rank=self._rank,
            world_size=self._world_size,
            BLOCK_SIZE=triton.next_power_of_2(self._world_size),
            MAX_SPINS=_MAX_FENCE_SPINS,
        )

    def reset(self) -> None:
        self._epoch.zero_()
        self._allocation.storage.zero_()
        torch.accelerator.synchronize()
        dist.barrier(group=self._group)

    def close(self) -> None:
        torch.accelerator.synchronize(self._allocation.storage.device)
        # Cleanup must remain non-collective: a peer may already have failed
        # while another worker is unwinding. A process-group barrier here can
        # deadlock the entire executor on any asynchronous worker failure.
        self._allocation.close()


@dataclass
class PCPDirectKVState:
    enabled: bool = False
    sharded: bool = False
    interleave_size: int = 1
    world_size: int = 1
    rank: int = 0
    allocations: list[SymmMemPeerAllocation] = field(default_factory=list)
    layer_peer_ptrs: dict[str, torch.Tensor] = field(default_factory=dict)
    tensor_peer_ptrs: dict[int, torch.Tensor] = field(default_factory=dict)
    fence: PCPPeerCacheFence | None = None

    def close(self) -> None:
        if self.fence is not None:
            self.fence.close()
            self.fence = None
        for allocation in self.allocations:
            allocation.close()
        self.allocations.clear()
        self.layer_peer_ptrs.clear()
        self.tensor_peer_ptrs.clear()
        self.enabled = False
        self.sharded = False
        self.interleave_size = 1


_STATE = PCPDirectKVState()


def pcp_direct_kv_requested() -> bool:
    return bool(envs.VLLM_USE_PCP_DIRECT_KV)


def pcp_direct_kv_active() -> bool:
    """Whether PCP direct-KV is replicating every cache row to every peer."""
    return _STATE.enabled and not _STATE.sharded


def pcp_peer_kv_active() -> bool:
    """Whether either replicated or sharded PCP peer cache views are active."""
    return _STATE.enabled


def pcp_sharded_peer_kv_active() -> bool:
    """Whether PCP peer views expose DCP-sharded cache rows in place."""
    return _STATE.enabled and _STATE.sharded


def get_pcp_direct_kv_state() -> PCPDirectKVState:
    return _STATE


def get_layer_peer_ptrs(layer_name: str) -> torch.Tensor | None:
    if not _STATE.enabled:
        return None
    return _STATE.layer_peer_ptrs.get(layer_name)


def gather_pcp_sharded_peer_cache(
    cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    seq_starts: torch.Tensor,
    num_tokens: int,
    scale: torch.Tensor,
    cache_block_size: int,
    packed_ds_mla: bool,
) -> None:
    """Gather a logical context directly from DCP-sharded PCP peer caches.

    Every PCP rank follows its own query/context schedule, so this deliberately
    performs peer loads without a collective. The physical cache address is
    derived from the DCP interleave owner and that owner's compressed position.
    """
    if not pcp_sharded_peer_kv_active():
        raise RuntimeError("PCP sharded peer cache is not active")
    peer_ptrs = _STATE.tensor_peer_ptrs.get(cache.data_ptr())
    if peer_ptrs is None:
        raise RuntimeError("No PCP peer cache pointers registered for cache tensor")
    if num_tokens == 0:
        return
    if dst.ndim != 2 or not dst.is_contiguous():
        raise ValueError("PCP sharded peer gather requires contiguous [T, D] output")
    if dst.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"Unsupported PCP peer gather output dtype: {dst.dtype}")
    row_dim = dst.shape[1]
    cache_row_bytes = cache.shape[-1] * cache.element_size()
    cache_block_stride_bytes = cache.stride(0) * cache.element_size()
    cache_token_stride_bytes = cache.stride(1) * cache.element_size()
    if packed_ds_mla and (row_dim != 576 or cache_row_bytes != 656):
        raise ValueError(
            "fp8_ds_mla peer gather requires 576 output elements and a "
            f"656-byte cache entry; got {row_dim} and {cache_row_bytes}"
        )
    if dst.shape[0] < num_tokens:
        raise ValueError(
            f"PCP peer gather output has {dst.shape[0]} rows, need {num_tokens}"
        )
    if block_table.ndim != 2:
        raise ValueError("PCP sharded peer gather requires a 2-D block table")
    block = 256
    _gather_sharded_peer_cache_kernel[(num_tokens, triton.cdiv(row_dim, block))](
        peer_ptrs,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        seq_starts,
        scale,
        block_table_stride=block_table.stride(0),
        cache_block_size=cache_block_size,
        cache_block_stride_bytes=cache_block_stride_bytes,
        cache_token_stride_bytes=cache_token_stride_bytes,
        row_dim=row_dim,
        world_size=peer_ptrs.numel(),
        interleave_size=_STATE.interleave_size,
        packed_ds_mla=packed_ds_mla,
        BLOCK_SIZE=block,
    )


def copy_pcp_cache_rows_to_peers(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    peer_ptrs: torch.Tensor,
    source_rank: int,
    block_size: int,
) -> None:
    """Copy locally produced physical cache rows to every PCP peer.

    The cache must be a contiguous paged tensor whose first two logical
    dimensions are ``[num_blocks, block_size]``. The copy is byte-oriented,
    so it preserves BF16, FP8 and packed cache row layouts unchanged.
    """
    if not cache.is_cuda or not slot_mapping.is_cuda or not peer_ptrs.is_cuda:
        raise ValueError("PCP cache-row publication requires CUDA tensors")
    if not cache.is_contiguous():
        raise ValueError("PCP cache-row publication requires a contiguous cache")
    if cache.ndim < 2 or cache.shape[1] != block_size:
        raise ValueError(
            "Expected paged cache shape [num_blocks, block_size, ...], got "
            f"{tuple(cache.shape)} with block_size={block_size}"
        )
    if slot_mapping.ndim != 1:
        raise ValueError("PCP cache-row slot mapping must be one-dimensional")
    world_size = peer_ptrs.numel()
    if world_size <= 1 or slot_mapping.numel() == 0:
        return
    if not 0 <= source_rank < world_size:
        raise ValueError(
            f"PCP source rank {source_rank} is outside world size {world_size}"
        )
    num_physical_rows = cache.shape[0] * block_size
    cache_nbytes = cache.numel() * cache.element_size()
    if cache_nbytes % num_physical_rows != 0:
        raise ValueError("Cache byte size is not divisible by its physical rows")
    row_nbytes = cache_nbytes // num_physical_rows
    block = 256
    _copy_cache_rows_to_peers_kernel[
        (slot_mapping.numel(), triton.cdiv(row_nbytes, block))
    ](
        peer_ptrs,
        slot_mapping,
        row_nbytes,
        source_rank=source_rank,
        world_size=world_size,
        BLOCK_SIZE=block,
    )


def publish_pcp_cache_rows(
    layer_name: str,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    if not _STATE.enabled:
        raise RuntimeError("PCP direct-KV is not active")
    peer_ptrs = _STATE.layer_peer_ptrs.get(layer_name)
    if peer_ptrs is None:
        raise RuntimeError(f"No PCP peer cache pointers registered for {layer_name}")
    copy_pcp_cache_rows_to_peers(
        cache, slot_mapping, peer_ptrs, _STATE.rank, block_size
    )


def publish_pcp_direct_kv() -> None:
    if pcp_direct_kv_active() and _STATE.fence is not None:
        _STATE.fence()


def publish_pcp_sharded_peer_kv() -> None:
    """Publish local DCP cache writes before asynchronous peer reads."""
    if pcp_sharded_peer_kv_active() and _STATE.fence is not None:
        _STATE.fence()


def reset_pcp_peer_cache_fence() -> None:
    if _STATE.fence is not None:
        _STATE.fence.reset()


def should_allocate_pcp_direct_kv(vllm_config) -> bool:
    if not pcp_direct_kv_requested():
        return False
    if not current_platform.is_cuda() or not symm_mem_available:
        raise RuntimeError(
            "VLLM_USE_PCP_DIRECT_KV=1 requires CUDA torch.distributed._symmetric_memory"
        )
    parallel_config = vllm_config.parallel_config
    if parallel_config.prefill_context_parallel_size <= 1:
        raise RuntimeError(
            "VLLM_USE_PCP_DIRECT_KV=1 requires prefill_context_parallel_size > 1"
        )
    dcp_size = parallel_config.decode_context_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    if dcp_size not in (1, pcp_size):
        raise RuntimeError(
            "VLLM_USE_PCP_DIRECT_KV requires DCP=1 or DCP=PCP; got "
            f"DCP={dcp_size}, PCP={pcp_size}"
        )
    if parallel_config.data_parallel_size != 1:
        raise RuntimeError("VLLM_USE_PCP_DIRECT_KV requires data_parallel_size=1")
    return True


def allocate_pcp_direct_backing(
    nbytes: int, device: torch.device, group: ProcessGroup
) -> SymmMemPeerAllocation:
    allocation = allocate_symm_mem_peer((nbytes,), torch.int8, device, group)
    _STATE.allocations.append(allocation)
    return allocation


def bind_pcp_direct_layer_views(
    kv_caches: dict[str, object],
    group: ProcessGroup,
    device: torch.device,
) -> None:
    if not _STATE.allocations:
        raise RuntimeError(
            "VLLM_USE_PCP_DIRECT_KV=1 requires every KV buffer to be allocated "
            "with PyTorch symmetric memory"
        )
    layer_peer_ptrs: dict[str, torch.Tensor] = {}
    tensor_peer_ptrs: dict[int, torch.Tensor] = {}
    missing: list[str] = []
    for layer_name, cache in kv_caches.items():
        tensor = _as_cache_tensor(cache)
        if tensor is None:
            continue
        allocation = _allocation_for_tensor(tensor)
        if allocation is None:
            missing.append(layer_name)
            continue
        peer_ptrs = allocation.peer_ptrs_for_view(tensor)
        layer_peer_ptrs[layer_name] = peer_ptrs
        tensor_peer_ptrs[tensor.data_ptr()] = peer_ptrs
    if missing:
        raise RuntimeError(
            "VLLM_USE_PCP_DIRECT_KV=1: cache layers not on symmetric-memory "
            f"backing: {', '.join(missing)}"
        )
    if not layer_peer_ptrs:
        raise RuntimeError("VLLM_USE_PCP_DIRECT_KV=1: no bindable KV cache tensors")
    _STATE.layer_peer_ptrs = layer_peer_ptrs
    _STATE.tensor_peer_ptrs = tensor_peer_ptrs
    _STATE.world_size = group.size()
    _STATE.rank = group.rank()
    parallel_config = get_current_vllm_config().parallel_config
    _STATE.sharded = parallel_config.decode_context_parallel_size > 1
    _STATE.interleave_size = parallel_config.cp_kv_cache_interleave_size
    _STATE.fence = PCPPeerCacheFence(group, device)
    _STATE.enabled = True
    logger.info(
        "PCP direct-KV enabled: world_size=%d layers=%d allocations=%d sharded=%s",
        _STATE.world_size,
        len(layer_peer_ptrs),
        len(_STATE.allocations),
        _STATE.sharded,
    )


def close_pcp_direct_kv() -> None:
    _STATE.close()


def _as_cache_tensor(cache: object) -> torch.Tensor | None:
    if isinstance(cache, torch.Tensor):
        return cache
    kv_cache = getattr(cache, "kv_cache", None)
    if isinstance(kv_cache, torch.Tensor):
        return kv_cache
    return None


def _allocation_for_tensor(tensor: torch.Tensor) -> SymmMemPeerAllocation | None:
    storage_ptr = tensor.untyped_storage().data_ptr()
    for allocation in _STATE.allocations:
        if allocation.storage.untyped_storage().data_ptr() == storage_ptr:
            return allocation
    return None


def pcp_group_is_single_node(group) -> bool:
    try:
        return all(in_the_same_node_as(group.cpu_group, source_rank=0))
    except Exception:
        return False
