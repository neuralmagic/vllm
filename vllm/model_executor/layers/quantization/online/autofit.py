# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Calibration-free per-layer weight-bitwidth autofit.

For each Linear, sweep a small ladder of integer-group-quant schemes and
pick the smallest bit width whose weight-reconstruction relative L2 error
stays under a threshold. The chosen scheme is applied as
``dequant(quant(W))`` so the forward path remains a standard bf16 GEMM. This
is a calibration-free, online procedure that only inspects the weight tensor
itself; nothing about activations or dataset samples is used.

Tunables (env vars):
  VLLM_AUTOFIT_THRESHOLD   default 0.10
  VLLM_AUTOFIT_GROUP_SIZE  default 128
  VLLM_AUTOFIT_BITS        comma-separated, default "4,5,6,8"
"""

from __future__ import annotations

import os

import torch
from torch.nn import Module

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter

logger = init_logger(__name__)


def _parse_bits(s: str) -> tuple[int, ...]:
    return tuple(sorted({int(x) for x in s.split(",") if x.strip()}))


def _autofit_config() -> tuple[float, int, tuple[int, ...]]:
    threshold = float(os.environ.get("VLLM_AUTOFIT_THRESHOLD", "0.10"))
    group_size = int(os.environ.get("VLLM_AUTOFIT_GROUP_SIZE", "128"))
    bits = _parse_bits(os.environ.get("VLLM_AUTOFIT_BITS", "4,5,6,8"))
    return threshold, group_size, bits


def _group_quant_dequant_uint(
    W: torch.Tensor, num_bits: int, group_size: int
) -> torch.Tensor:
    """Asymmetric uint group quant + dequant. Returns same dtype as W.

    Pure-PyTorch implementation: per-group min/max, integer round, dequantize.
    Equivalent (up to fp precision) to ``humming.utils.weight.quantize_weight``
    with ``has_zero_point=True, scale_dtype=bf16``.
    """
    assert W.ndim == 2, f"expected 2D weight, got {W.shape}"
    N, K = W.shape
    if K % group_size != 0:
        # Caller is responsible for skipping; surface clearly if reached.
        raise ValueError(
            f"K={K} not divisible by group_size={group_size} for autofit"
        )
    G = K // group_size
    qmax = float((1 << num_bits) - 1)

    Wg = W.float().view(N, G, group_size)
    wmin = Wg.amin(dim=-1, keepdim=True)
    wmax = Wg.amax(dim=-1, keepdim=True)
    scale = (wmax - wmin).clamp_min(1e-8) / qmax
    # bf16 is the storage dtype the runtime kernel would use.
    scale_q = scale.to(torch.bfloat16).float()
    zp = (-wmin / scale_q).round().clamp(0, qmax)
    q = ((Wg / scale_q) + zp).round().clamp(0, qmax)
    deq = (q - zp) * scale_q
    return deq.view(N, K).to(W.dtype)


def _rel_l2(W_ref: torch.Tensor, W_q: torch.Tensor) -> float:
    return (
        (W_q - W_ref).norm() / W_ref.norm().clamp_min(1e-12)
    ).item()


def _pick_bits(
    W: torch.Tensor, threshold: float, group_size: int, bits_ladder: tuple[int, ...]
) -> tuple[int, float, torch.Tensor]:
    """Walk bits low->high; return (chosen_bits, err, dequant_weight).

    Falls back to the highest-bits candidate if none meets the threshold.
    """
    best_high = None  # type: tuple[int, float, torch.Tensor] | None
    for b in sorted(bits_ladder):
        Wq = _group_quant_dequant_uint(W, b, group_size)
        err = _rel_l2(W, Wq)
        if err <= threshold:
            return b, err, Wq
        best_high = (b, err, Wq)
    assert best_high is not None
    return best_high


class AutofitOnlineLinearMethod(LinearMethodBase):
    """Online autofit: per-layer pick smallest INT-group-quant scheme s.t.
    weight reconstruction rel-L2 error stays under threshold.

    The dequantized weight is stored back as bf16/fp16 so the forward path is
    a plain ``torch.nn.functional.linear``. No GEMM kernel changes; the value
    here is to expose the *bit allocation* to downstream eval and to lay the
    groundwork for swapping in a low-bit GEMM (e.g. Humming) per layer based
    on the chosen scheme.
    """

    def __init__(self) -> None:
        threshold, group_size, bits = _autofit_config()
        self.threshold = threshold
        self.group_size = group_size
        self.bits = bits
        # populated by process_weights_after_loading per layer (for logging)
        self._chosen: dict[str, tuple[int, float]] = {}

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        # `set_weight_attrs`-style: stash any other attrs on the param so
        # the rest of the loader/sharding machinery still finds them.
        for k, v in extra_weight_attrs.items():
            setattr(weight, k, v)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        W = layer.weight.data
        if W.ndim != 2 or W.shape[-1] % self.group_size != 0:
            # Skip oddly shaped layers (e.g. K not divisible by group_size).
            logger.warning(
                "[autofit] skipping layer with weight shape %s "
                "(K not divisible by group_size=%d)",
                tuple(W.shape),
                self.group_size,
            )
            layer._already_called_process_weights_after_loading = True
            return

        chosen_bits, err, Wq = _pick_bits(
            W, self.threshold, self.group_size, self.bits
        )
        layer.weight.data.copy_(Wq)
        # Record the choice on the layer for inspection/logging.
        layer._autofit_bits = chosen_bits
        layer._autofit_err = err
        layer._autofit_group_size = self.group_size
        logger.info(
            "[autofit] %s: bits=u%d group=%d err=%.4f shape=%s",
            getattr(layer, "_autofit_name", "<linear>"),
            chosen_bits,
            self.group_size,
            err,
            tuple(W.shape),
        )

        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, layer.weight, bias)
