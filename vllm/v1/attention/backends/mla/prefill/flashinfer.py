# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer backend for MLA prefill."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.mla.prefill.base import (
    MLAPrefillBackend,
    MLAPrefillImpl,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability

try:
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
except ImportError:
    BatchPrefillWithRaggedKVCacheWrapper = object  # type: ignore[misc,assignment]


# Import base class for metadata - runtime import to avoid circular dependency
def _get_base_metadata_cls():
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )

    return MLACommonPrefillMetadata


@dataclass
class FlashInferPrefillMetadata(_get_base_metadata_cls()):  # type: ignore[misc]
    """FlashInfer-specific prefill metadata."""

    prefill_main: BatchPrefillWithRaggedKVCacheWrapper | None = None
    prefill_chunks: list[BatchPrefillWithRaggedKVCacheWrapper] = field(
        default_factory=list
    )


class FlashInferPrefillBackend(MLAPrefillBackend):
    """FlashInfer backend for MLA prefill.

    This backend is optimized for Blackwell (SM100) architecture.
    """

    requires_r1_mla_dimensions = True

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_PREFILL"

    @staticmethod
    def get_prefill_impl_cls() -> type["FlashInferPrefillImpl"]:
        return FlashInferPrefillImpl

    @staticmethod
    def get_prefill_metadata_cls() -> type["FlashInferPrefillMetadata"]:
        return FlashInferPrefillMetadata

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        return device_capability.major == 10

    @classmethod
    def is_available(cls) -> bool:
        try:
            from flashinfer import (
                BatchPrefillWithRaggedKVCacheWrapper,  # noqa: F401
            )

            return True
        except ImportError:
            return False


class FlashInferPrefillImpl(MLAPrefillImpl):
    """FlashInfer implementation for MLA prefill."""

    requires_v_padding: bool = False

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
            device=device,
        )

    def run_prefill_new_tokens(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(prefill_metadata, FlashInferPrefillMetadata)
        assert prefill_metadata.prefill_main is not None

        ret = prefill_metadata.prefill_main.run(
            q=q,
            k=k,
            v=v,
            return_lse=return_softmax_lse,
        )

        if isinstance(ret, tuple):
            # Convert from (q_len, num_heads) to (num_heads, q_len)
            return ret[0], ret[1].transpose(0, 1).contiguous()
        return ret

    def run_prefill_context_chunk(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(prefill_metadata, FlashInferPrefillMetadata)

        attn_out, lse = prefill_metadata.prefill_chunks[chunk_idx].run(
            q=q,
            k=k,
            v=v,
            return_lse=True,
        )

        # Convert from (q_len, num_heads) to (num_heads, q_len)
        return attn_out, lse.transpose(0, 1).contiguous()
