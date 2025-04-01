import torch
from typing import List, Optional, Tuple

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.utils import cdiv

def _fp8_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    block_shape: Optional[List[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform fp8 quantization on the inputs.  If a block_shape
    is provided, the output will be blocked.
    """
    if block_shape is None:
        A, A_scale = ops.scaled_fp8_quant(A, A_scale)
    else:
        assert len(block_shape) == 2
        _, block_k = block_shape[0], block_shape[1]
        A, A_scale = per_token_group_quant_fp8(A, block_k)
        assert cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
    return A, A_scale


def _fp8_perm(m: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    A permutation routine that works on fp8 types.
    """
    if torch.is_floating_point(m) and torch.finfo(m.dtype).bits == 8:
        return m.view(dtype=torch.uint8)[idx, ...].view(dtype=m.dtype)
    else:
        return m[idx, ...]
