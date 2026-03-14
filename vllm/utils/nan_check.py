# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NaN/Inf detection for debugging numerical issues under CUDA graphs.

Controlled by the VLLM_NAN_CHECK=1 environment variable.

Each instrumented decoder layer registers backing tensors (as nn.Buffers)
that capture full hidden_states at four checkpoints: before/after attention
and before/after MLP.  Because the buffers are registered on the module,
Dynamo tracks the mutations and the .copy_() calls are compiled into the
CUDA graph.

After the full model forward returns in gpu_model_runner.execute_model,
the registry iterates over every registered backing tensor and inspects
it for NaN/Inf, reporting real-token and padded-token regions separately.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.library

from vllm.logger import init_logger

logger = init_logger(__name__)

_file_handler: logging.FileHandler | None = None


def _ensure_file_logger() -> None:
    """Add a file handler so NaN check messages are also written to disk."""
    global _file_handler
    if _file_handler is not None:
        return
    log_dir = os.getenv("VLLM_NAN_CHECK_LOG_DIR", "/tmp")
    rank = "unknown"
    try:
        if torch.distributed.is_initialized():
            rank = str(torch.distributed.get_rank())
    except Exception:
        pass
    log_path = os.path.join(log_dir, f"nan_check_rank{rank}.log")
    _file_handler = logging.FileHandler(log_path, mode="w")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_file_handler)
    logger.warning("[NaN Check] Logging to file: %s", log_path)


# ---------------------------------------------------------------------------
# Custom ops — opaque to Inductor so copies/detections survive compilation
# ---------------------------------------------------------------------------

@torch.library.custom_op("vllm::nan_check_copy", mutates_args=("dst",))
def nan_check_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy src into dst, handling mismatched first dimensions.
    Registered as a custom op so that Inductor cannot dead-code-eliminate
    the write (it sees the mutation via ``mutates_args``).
    The size clamping is inside the op so Inductor cannot optimize it away."""
    n = min(dst.shape[0], src.shape[0])
    dst[:n].copy_(src[:n])


@nan_check_copy.register_fake
def _nan_check_copy_fake(dst: torch.Tensor, src: torch.Tensor) -> None:
    pass


@torch.library.custom_op("vllm::nan_check_detect", mutates_args=("flags",))
def nan_check_detect(flags: torch.Tensor, src: torch.Tensor) -> None:
    """Per-token NaN/Inf detection: flags[i] = 1 if src[i] has NaN or Inf.
    ``flags`` shape is (max_tokens,) int8; ``src`` can be any (tokens, ...).
    Size clamping is inside the op so Inductor cannot optimize it away."""
    n = min(flags.shape[0], src.shape[0])
    src_flat = src[:n].reshape(n, -1)
    has_bad = (torch.isnan(src_flat).any(dim=1)
               | torch.isinf(src_flat).any(dim=1))
    flags[:n] = has_bad.to(flags.dtype)


@nan_check_detect.register_fake
def _nan_check_detect_fake(flags: torch.Tensor, src: torch.Tensor) -> None:
    pass


@torch.library.custom_op("vllm::nan_check_kv_written",
                         mutates_args=("flags",))
def nan_check_kv_written(flags: torch.Tensor, kv_cache: torch.Tensor,
                         slot_mapping: torch.Tensor) -> None:
    """Check just-written KV cache entries for NaN/Inf.
    kv_cache: (num_blocks, block_size, head_size)
    slot_mapping: (num_tokens,) flat slot indices
    flags: (max_tokens,) int8 output
    Fully GPU-resident — no host sync, safe during CUDA graph capture."""
    n = min(flags.shape[0], slot_mapping.shape[0])
    slots = slot_mapping[:n]
    kv_flat = kv_cache.view(-1, kv_cache.shape[-1])
    num_kv_slots = kv_flat.shape[0]
    clamped = slots.clamp(0, num_kv_slots - 1)
    entries = kv_flat[clamped]
    if entries.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        entries = entries.float()
    has_bad = (torch.isnan(entries).any(dim=1)
               | torch.isinf(entries).any(dim=1))
    valid = (slots >= 0) & (slots < num_kv_slots)
    flags[:n] = (has_bad & valid).to(flags.dtype)


@nan_check_kv_written.register_fake
def _nan_check_kv_written_fake(flags: torch.Tensor, kv_cache: torch.Tensor,
                               slot_mapping: torch.Tensor) -> None:
    pass


@torch.library.custom_op("vllm::fp4_pad_nan_check", mutates_args=("flag",))
def fp4_pad_nan_check(flag: torch.Tensor, scales: torch.Tensor,
                      m: int, rounded_m: int) -> None:
    """Check padding rows of FP4 swizzled scales for NaN/Inf.
    flag: (1,) int32 — set to 1 if any padding row has NaN/Inf.
    scales: the full output_scale tensor.
    Fully GPU-resident — no scalar indexing, safe during CUDA graph capture."""
    if rounded_m <= m:
        return
    pad = scales[m:rounded_m]
    pad_f8 = pad.reshape(-1).view(torch.float8_e4m3fn).float()
    has_bad = torch.isnan(pad_f8).any() | torch.isinf(pad_f8).any()
    # OR-accumulate into flag using bitwise_or_ (no scalar indexing)
    flag.bitwise_or_(has_bad.to(flag.dtype).unsqueeze(0))


@fp4_pad_nan_check.register_fake
def _fp4_pad_nan_check_fake(flag: torch.Tensor, scales: torch.Tensor,
                            m: int, rounded_m: int) -> None:
    pass


_fp4_pad_flag: torch.Tensor | None = None


def get_fp4_pad_flag(device: torch.device) -> torch.Tensor:
    """Lazily allocate a persistent (1,) int32 flag on the given device."""
    global _fp4_pad_flag
    if _fp4_pad_flag is None or _fp4_pad_flag.device != device:
        _fp4_pad_flag = torch.zeros(1, device=device, dtype=torch.int32)
    return _fp4_pad_flag


def check_and_reset_fp4_pad_flag() -> None:
    """Called after forward — check the flag and log if set."""
    global _fp4_pad_flag
    if _fp4_pad_flag is None:
        return
    has_bad = _fp4_pad_flag.item()
    if has_bad:
        _ensure_file_logger()
        rank = (torch.distributed.get_rank()
                if torch.distributed.is_initialized() else 0)
        logger.error(
            "[FP4 Scale Check] NaN/Inf detected in PADDING rows of "
            "output_scale (swizzled layout) | rank=%d", rank,
        )
    _fp4_pad_flag.zero_()


# ---------------------------------------------------------------------------
# Feature gate
# ---------------------------------------------------------------------------

_NAN_CHECK_ENABLED: bool | None = None


def nan_check_enabled() -> bool:
    global _NAN_CHECK_ENABLED
    if _NAN_CHECK_ENABLED is None:
        _NAN_CHECK_ENABLED = bool(int(os.getenv("VLLM_NAN_CHECK", "0")))
        if _NAN_CHECK_ENABLED:
            logger.warning("VLLM_NAN_CHECK is enabled – "
                           "full-tensor NaN/Inf checks are active. "
                           "This adds memory overhead and should only be "
                           "used for debugging.")
    return _NAN_CHECK_ENABLED


# ---------------------------------------------------------------------------
# Registry data structures
# ---------------------------------------------------------------------------

CHECKPOINT_LABELS: list[str] = [
    "before_attention",
    "after_attention",
    "before_mlp",
    "after_mlp",
]
NUM_CHECKPOINTS = len(CHECKPOINT_LABELS)


@dataclass
class NanCheckEntry:
    """One tensor being tracked for NaN/Inf."""
    label: str
    backing: torch.Tensor
    is_flag: bool = False


@dataclass
class NanCheckGroup:
    """A group of entries for one decoder layer."""
    layer_name: str
    entries: list[NanCheckEntry] = field(default_factory=list)
    metadata: dict[str, torch.Tensor] = field(default_factory=dict)

    def add(self, label: str, backing: torch.Tensor,
            is_flag: bool = False) -> None:
        self.entries.append(NanCheckEntry(label=label, backing=backing,
                                         is_flag=is_flag))

    def add_metadata(self, label: str, backing: torch.Tensor) -> None:
        self.metadata[label] = backing


class NanCheckRegistry:
    """Global singleton that tracks all NaN-check backing tensors."""
    _instance: NanCheckRegistry | None = None

    @classmethod
    def get(cls) -> NanCheckRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.groups: list[NanCheckGroup] = []
        self._check_count: int = 0

    def new_group(self, layer_name: str) -> NanCheckGroup:
        group = NanCheckGroup(layer_name=layer_name)
        self.groups.append(group)
        return group

    def run_checks(self, num_tokens: int,
                   num_tokens_padded: int) -> None:
        if not nan_check_enabled():
            return

        _ensure_file_logger()

        logger.warning_once(
            "[NaN Check] run_checks is active — checking %d layers "
            "(%d total entries)",
            len(self.groups),
            sum(len(g.entries) for g in self.groups),
        )

        if len(self.groups) == 0:
            logger.error("[NaN Check] No groups to check")

        rank = (torch.distributed.get_rank()
                if torch.distributed.is_initialized() else 0)

        self._check_count += 1
        if self._check_count % 1000 == 1:
            group_names = [g.layer_name for g in self.groups]
            logger.warning(
                "[NaN Check] check #%d | rank=%d num_tokens=%d "
                "num_tokens_padded=%d groups=%s",
                self._check_count, rank, num_tokens,
                num_tokens_padded, group_names,
            )

        for group in self.groups:
            if len(group.entries) == 0:
                logger.error("[NaN Check] No entries to check for "
                             "group %s", group.layer_name)
                continue

            for entry in group.entries:
                n_real = min(num_tokens, entry.backing.shape[0])
                n_pad = min(num_tokens_padded, entry.backing.shape[0])

                if entry.is_flag:
                    _report_flags(entry.backing[:n_real],
                                  group.layer_name, entry.label,
                                  "real_tokens", rank, n_real,
                                  group.metadata)
                    if n_pad > n_real:
                        _report_flags(entry.backing[n_real:n_pad],
                                      group.layer_name, entry.label,
                                      "padded_tokens", rank,
                                      n_pad - n_real,
                                      group.metadata)
                else:
                    _report(entry.backing[:n_real], group.layer_name,
                            entry.label, "real_tokens", rank, n_real)

                    if n_pad > n_real:
                        _report(entry.backing[n_real:n_pad],
                                group.layer_name, entry.label,
                                "padded_tokens", rank, n_pad - n_real)


# ---------------------------------------------------------------------------
# Final-output check (called directly, not via registry)
# ---------------------------------------------------------------------------

def check_final_output(hidden_states: torch.Tensor,
                       num_tokens: int,
                       num_tokens_padded: int) -> None:
    """Check the final model output tensor for NaN/Inf.

    Works even under CUDA graph replay because the output tensor
    is the graph's output buffer and always holds current values.
    """
    if not nan_check_enabled():
        return
    rank = (torch.distributed.get_rank()
            if torch.distributed.is_initialized() else 0)
    if isinstance(hidden_states, torch.Tensor):
        _report(hidden_states[:num_tokens], "model_output",
                "final_hidden_states", "real_tokens", rank, num_tokens)
        if num_tokens_padded > num_tokens:
            _report(hidden_states[num_tokens:num_tokens_padded],
                    "model_output", "final_hidden_states",
                    "padded_tokens", rank, num_tokens_padded - num_tokens)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

_padded_hit_counts: dict[str, int] = {}
_PADDED_LOG_INTERVAL = 500


def _report_flags(flags: torch.Tensor, layer_name: str, label: str,
                  region: str, rank: int, count: int,
                  metadata: dict[str, torch.Tensor] | None = None
                  ) -> None:
    """Report NaN/Inf from a per-token flag buffer (int8, 1 = bad)."""
    bad_indices = flags.nonzero(as_tuple=False).flatten()
    if bad_indices.numel() == 0:
        return
    msg = (
        "[NaN Check] NaN/Inf flagged in %s | layer=%s region=%s "
        "rank=%d n_tokens_in_region=%d bad_token_count=%d "
        "bad_token_indices=%s"
    )
    idx_list = bad_indices[:20].tolist()
    args = (label, layer_name, region, rank, count,
            int(bad_indices.numel()), idx_list)

    extra = ""
    if metadata:
        if "slot_mapping" in metadata:
            slots = metadata["slot_mapping"]
            slot_vals = [int(slots[i].item()) for i in idx_list
                         if i < slots.shape[0]]
            extra += f" slot_mapping_for_bad={slot_vals}"
        if "seq_lens" in metadata:
            seq_lens = metadata["seq_lens"]
            sl_vals = [int(seq_lens[i].item()) for i in idx_list
                       if i < seq_lens.shape[0]]
            extra += f" seq_lens_for_bad={sl_vals}"

    if region == "real_tokens":
        logger.error(msg + extra, *args)
    else:
        key = f"{layer_name}|{label}|{rank}"
        _padded_hit_counts[key] = _padded_hit_counts.get(key, 0) + 1
        if _padded_hit_counts[key] % _PADDED_LOG_INTERVAL == 1:
            logger.warning(msg + extra + " (hit #%d, logging every %d)",
                           *args, _padded_hit_counts[key],
                           _PADDED_LOG_INTERVAL)


def _report(tensor: torch.Tensor, layer_name: str, label: str,
            region: str, rank: int, count: int) -> None:
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if not (has_nan or has_inf):
        return
    problems = []
    if has_nan:
        nan_count = int(torch.isnan(tensor).sum().item())
        problems.append(f"NaN(count={nan_count})")
    if has_inf:
        inf_count = int(torch.isinf(tensor).sum().item())
        problems.append(f"Inf(count={inf_count})")

    msg = (
        "[NaN Check] %s detected in %s | layer=%s region=%s "
        "shape=%s dtype=%s rank=%d n_tokens_in_region=%d"
    )
    args = ("/".join(problems), label, layer_name, region,
            list(tensor.shape), tensor.dtype, rank, count)

    if region == "real_tokens":
        logger.error(msg, *args)
    else:
        key = f"{layer_name}|{label}|{rank}"
        _padded_hit_counts[key] = _padded_hit_counts.get(key, 0) + 1
        if _padded_hit_counts[key] % _PADDED_LOG_INTERVAL == 1:
            logger.warning(msg + " (hit #%d, logging every %d)",
                           *args, _padded_hit_counts[key],
                           _PADDED_LOG_INTERVAL)


# ---------------------------------------------------------------------------
# Pre-forward batch sanity check
# ---------------------------------------------------------------------------

def check_batch_sanity(
    seq_lens: np.ndarray,
    num_computed_tokens: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    num_reqs: int,
    num_reqs_padded: int,
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    rank: int,
    req_ids: list[str] | None = None,
) -> None:
    """Validate batch composition before forward pass.

    Checks for degenerate conditions like seq_lens=0 for real requests,
    and real tokens falling in the padded-request region.
    """
    if not nan_check_enabled():
        return
    _ensure_file_logger()

    # Check 1: any real request (index < num_reqs) with seq_lens == 0
    real_seq_lens = seq_lens[:num_reqs]
    zero_mask = real_seq_lens == 0
    if zero_mask.any():
        zero_indices = np.where(zero_mask)[0]
        for idx in zero_indices[:20]:
            req_id = req_ids[idx] if req_ids else f"idx={idx}"
            logger.error(
                "[Batch Sanity] seq_lens=0 for REAL request | rank=%d "
                "req=%s req_idx=%d num_computed=%d num_scheduled=%d | "
                "num_reqs=%d num_reqs_padded=%d "
                "num_tokens_unpadded=%d num_tokens_padded=%d",
                rank, req_id, int(idx),
                int(num_computed_tokens[idx]),
                int(num_scheduled_tokens[idx]),
                num_reqs, num_reqs_padded,
                num_tokens_unpadded, num_tokens_padded,
            )

    # Check 2: padded-region requests (index >= num_reqs) that have
    # tokens counted inside num_tokens_unpadded.  In uniform decode
    # (1 token per req), token index == request index, so padded
    # requests beyond num_reqs should not contribute real tokens.
    if num_reqs_padded > num_reqs:
        padded_seq_lens = seq_lens[num_reqs:num_reqs_padded]
        nonzero_padded = padded_seq_lens != 0
        if nonzero_padded.any():
            bad = np.where(nonzero_padded)[0]
            for idx in bad[:10]:
                abs_idx = num_reqs + idx
                logger.error(
                    "[Batch Sanity] PADDED request has seq_lens!=0 | "
                    "rank=%d req_idx=%d seq_len=%d | "
                    "num_reqs=%d num_reqs_padded=%d",
                    rank, abs_idx, int(padded_seq_lens[idx]),
                    num_reqs, num_reqs_padded,
                )

    # Check 3: log the full batch shape for diagnosis on every invocation
    # where there are zero-seq-len real requests
    if zero_mask.any():
        logger.error(
            "[Batch Sanity] Full batch dump | rank=%d "
            "num_reqs=%d num_reqs_padded=%d "
            "num_tokens_unpadded=%d num_tokens_padded=%d | "
            "seq_lens[:num_reqs_padded]=%s | "
            "num_computed[:num_reqs]=%s | "
            "num_scheduled[:num_reqs]=%s",
            rank, num_reqs, num_reqs_padded,
            num_tokens_unpadded, num_tokens_padded,
            seq_lens[:num_reqs_padded].tolist(),
            num_computed_tokens[:num_reqs].tolist(),
            num_scheduled_tokens[:num_reqs].tolist(),
        )


# ---------------------------------------------------------------------------
# Block table stability tracking
# ---------------------------------------------------------------------------

_prev_block_tables: dict[str, np.ndarray] = {}
_prev_num_tokens: dict[str, int] = {}


def _check_block_table_stability(
    bt_cpu: torch.Tensor,
    num_tokens_per_seq: np.ndarray,
    num_reqs: int,
    block_size: int,
    rank: int,
    req_ids: list[str] | None,
) -> None:
    """Detect if any existing blocks were reassigned between steps.

    Only checks blocks that covered tokens in the PREVIOUS step — new blocks
    appended for newly generated tokens are expected and not flagged.
    """
    global _prev_block_tables, _prev_num_tokens

    bt_np = bt_cpu.numpy()

    for seq_idx in range(num_reqs):
        req_id = req_ids[seq_idx] if req_ids else f"idx={seq_idx}"

        n_valid_now = int(num_tokens_per_seq[seq_idx])
        prev_n_valid = _prev_num_tokens.get(req_id, 0)

        if req_id in _prev_block_tables and prev_n_valid > 0:
            prev_bt = _prev_block_tables[req_id]
            n_prev_blocks = (prev_n_valid + block_size - 1) // block_size
            n_prev_blocks = min(n_prev_blocks, prev_bt.shape[0],
                                bt_np.shape[1])
            cur_row = bt_np[seq_idx, :n_prev_blocks]
            old_row = prev_bt[:n_prev_blocks]

            if not np.array_equal(cur_row, old_row):
                changed = np.where(cur_row != old_row)[0]
                for ci in changed[:20]:
                    logger.error(
                        "[Block Table CHANGED] rank=%d req=%s | "
                        "block_pos=%d old_block=%d new_block=%d | "
                        "prev_seq_len=%d cur_seq_len=%d",
                        rank, req_id,
                        int(ci), int(old_row[ci]), int(cur_row[ci]),
                        prev_n_valid, n_valid_now,
                    )

        n_blocks_now = (n_valid_now + block_size - 1) // block_size
        n_blocks_now = min(n_blocks_now, bt_np.shape[1])
        _prev_block_tables[req_id] = bt_np[seq_idx, :n_blocks_now].copy()
        _prev_num_tokens[req_id] = n_valid_now

    stale = set(_prev_block_tables.keys())
    active = set(req_ids[:num_reqs]) if req_ids else set()
    for rid in stale - active:
        del _prev_block_tables[rid]
        del _prev_num_tokens[rid]


# ---------------------------------------------------------------------------
# KV cache scanning — check all KV entries for sequences in the batch
# ---------------------------------------------------------------------------

def check_kv_caches_for_sequences(
    kv_caches: list[torch.Tensor],
    block_table_tensor: torch.Tensor,
    num_tokens_per_seq: np.ndarray,
    num_prompt_tokens_per_seq: np.ndarray,
    num_reqs: int,
    block_size: int,
    slot_mapping: torch.Tensor | None,
    rank: int,
    phase: str,
    req_ids: list[str] | None = None,
) -> None:
    """Scan KV cache blocks for every sequence in the batch.

    Args:
        kv_caches: list of per-layer KV cache tensors, each
            shaped (num_blocks, block_size, head_dim) for MLA.
        block_table_tensor: (num_reqs, max_blocks_per_req) int32, on GPU.
        num_tokens_per_seq: numpy array (num_reqs,) of valid token counts
            per sequence (num_computed for pre-forward, seq_lens for post).
        num_prompt_tokens_per_seq: numpy array (num_reqs,) of prompt token
            counts per sequence (used to compute completed decode steps).
        num_reqs: number of active requests.
        block_size: tokens per block.
        slot_mapping: (num_tokens_scheduled,) int64 on GPU — newly written
            slots in this forward step.  None if unavailable.
        rank: distributed rank.
        phase: "PRE_FORWARD" or "POST_FORWARD".
        req_ids: optional list of request ID strings for reporting.
    """
    if not nan_check_enabled():
        return
    _ensure_file_logger()

    new_slots: set[int] = set()
    if slot_mapping is not None:
        sm_cpu = slot_mapping.cpu().tolist()
        new_slots = {s for s in sm_cpu if s >= 0}

    bt_cpu = block_table_tensor[:num_reqs].cpu()

    if phase == "PRE_FORWARD":
        _check_block_table_stability(
            bt_cpu, num_tokens_per_seq, num_reqs, block_size, rank, req_ids)

    for layer_idx, kv_cache in enumerate(kv_caches):
        if kv_cache is None:
            continue
        if kv_cache.numel() == 0:
            continue

        _check_layer_kv(
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            bt_cpu=bt_cpu,
            num_tokens_per_seq=num_tokens_per_seq,
            num_prompt_tokens_per_seq=num_prompt_tokens_per_seq,
            num_reqs=num_reqs,
            block_size=block_size,
            new_slots=new_slots,
            rank=rank,
            phase=phase,
            req_ids=req_ids,
        )


def _check_layer_kv(
    kv_cache: torch.Tensor,
    layer_idx: int,
    bt_cpu: torch.Tensor,
    num_tokens_per_seq: np.ndarray,
    num_prompt_tokens_per_seq: np.ndarray,
    num_reqs: int,
    block_size: int,
    new_slots: set[int],
    rank: int,
    phase: str,
    req_ids: list[str] | None,
) -> None:
    """Check one layer's KV cache for all sequences."""

    # MLA: (num_blocks, block_size, head_dim)
    # Standard MHA: (2, num_blocks, block_size, head_dim) — handle both
    is_mla = kv_cache.dim() == 3
    if not is_mla:
        return

    num_kv_blocks = kv_cache.shape[0]

    all_block_ids: list[int] = []
    block_to_seqs: dict[int, list[tuple[int, int]]] = {}

    for seq_idx in range(num_reqs):
        n_valid = int(num_tokens_per_seq[seq_idx])
        if n_valid <= 0:
            continue
        n_blocks = (n_valid + block_size - 1) // block_size
        n_blocks = min(n_blocks, bt_cpu.shape[1])
        blocks = bt_cpu[seq_idx, :n_blocks].tolist()
        for blk_pos, blk_id in enumerate(blocks):
            if blk_id < 0 or blk_id >= num_kv_blocks:
                continue
            all_block_ids.append(blk_id)
            block_to_seqs.setdefault(blk_id, []).append((seq_idx, blk_pos))

    if not all_block_ids:
        return

    unique_ids = list(dict.fromkeys(all_block_ids))
    unique_tensor = torch.tensor(
        unique_ids, dtype=torch.long, device=kv_cache.device)

    gathered = kv_cache[unique_tensor]  # (n_unique, block_size, head_dim)
    if gathered.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        gathered = gathered.float()

    # per-slot NaN/Inf check: (n_unique, block_size)
    has_bad = (
        torch.isnan(gathered).any(dim=-1)
        | torch.isinf(gathered).any(dim=-1)
    )
    bad_block_mask = has_bad.any(dim=1)  # (n_unique,)

    if not bad_block_mask.any().item():
        return

    bad_block_indices = bad_block_mask.nonzero(as_tuple=False).flatten()
    bad_block_indices_cpu = bad_block_indices.cpu().tolist()
    has_bad_cpu = has_bad.cpu()

    for bad_idx in bad_block_indices_cpu:
        blk_id = unique_ids[bad_idx]
        bad_offsets = has_bad_cpu[bad_idx].nonzero(as_tuple=False).flatten().tolist()

        for seq_idx, blk_pos in block_to_seqs.get(blk_id, []):
            n_valid = int(num_tokens_per_seq[seq_idx])
            n_prompt = int(num_prompt_tokens_per_seq[seq_idx])
            decode_steps = max(0, n_valid - n_prompt)
            req_id = req_ids[seq_idx] if req_ids else f"idx={seq_idx}"

            for offset in bad_offsets:
                token_pos = blk_pos * block_size + offset
                if token_pos >= n_valid:
                    continue
                global_slot = blk_id * block_size + offset
                is_new = global_slot in new_slots

                logger.error(
                    "[KV NaN %s] layer=%d rank=%d | "
                    "req=%s seq_len=%d prompt_len=%d "
                    "decode_steps_done=%d token_pos=%d | "
                    "block=%d slot_in_block=%d global_slot=%d | "
                    "newly_written=%s",
                    phase, layer_idx, rank,
                    req_id, n_valid, n_prompt,
                    decode_steps, token_pos,
                    blk_id, offset, global_slot,
                    is_new,
                )