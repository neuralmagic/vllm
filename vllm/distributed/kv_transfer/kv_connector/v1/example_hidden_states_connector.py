# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import safetensors
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    """Lightweight per-request metadata for hidden-state accumulation."""

    req_id: str
    filename: str
    token_ids: list[int]
    new_req: bool


@dataclass
class ExampleHiddenStatesConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        req_id: str,
        filename: str,
        token_ids: list[int],
        new_req: bool = True,
    ) -> None:
        self.requests.append(
            ReqMeta(
                req_id=req_id,
                filename=filename,
                token_ids=token_ids,
                new_req=new_req,
            )
        )


class ExampleHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):
    """Hidden-states connector that bypasses the KV cache.

    The proposer stacks target-model hidden states and calls
    ``save_hidden_states()`` directly.  This connector accumulates the
    per-request tensors on CPU and writes them to disk when the request
    finishes.  No CacheOnly attention layers or KV cache groups are
    required, so hybrid models work without any core KV-cache changes.
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return False

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
        )
        logger.info(self._kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)

        assert self._vllm_config.speculative_config is not None, (
            "ExampleHiddenStatesConnector only works when using "
            "'extract_hidden_states' speculative method"
        )

        # Per-request accumulation buffers (CPU tensors).
        self._accumulated_hs: dict[str, list[torch.Tensor]] = {}
        self._accumulated_tokens: dict[str, list[int]] = {}
        self._request_filenames: dict[str, str] = {}
        self._active_requests: dict[str, NewRequestData] = {}

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, *args, **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def wait_for_save(self):
        pass

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        pass  # No CacheOnly layers in bypass mode.

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        pass  # Not used — hidden states arrive via save_hidden_states().

    def save_hidden_states(
        self,
        hidden_states: torch.Tensor,
        tokens_per_req: list[int],
    ) -> None:
        """Receive stacked hidden states from the proposer, accumulate, and
        persist to disk.

        Because ``request_finished()`` runs on the *scheduler* process (which
        has a separate connector instance), we must write the file here on the
        worker side.  Each call overwrites the previous file for a given
        request so that the latest accumulated state is always on disk.

        Args:
            hidden_states: [total_tokens, num_hidden_states, hidden_size]
                on the GPU.
            tokens_per_req: Number of tokens for each request in the batch
                (same order as connector metadata requests).
        """
        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, ExampleHiddenStatesConnectorMetadata)

        os.makedirs(self._storage_path, exist_ok=True)

        offset = 0
        for i, request in enumerate(connector_metadata.requests):
            if i >= len(tokens_per_req):
                break
            num_tokens = tokens_per_req[i]
            hs = hidden_states[offset : offset + num_tokens].detach().cpu()
            offset += num_tokens

            req_id = request.req_id
            if req_id not in self._accumulated_hs:
                self._accumulated_hs[req_id] = []
            self._accumulated_hs[req_id].append(hs)
            self._accumulated_tokens[req_id] = request.token_ids

            # Write the full accumulated hidden states to disk so the file
            # is ready when the scheduler calls request_finished().
            full_hs = torch.cat(self._accumulated_hs[req_id], dim=0)
            tensors = {
                "hidden_states": full_hs,
                "token_ids": torch.tensor(request.token_ids),
            }
            safetensors.torch.save_file(tensors, request.filename)

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        assert num_external_tokens == 0, "This connector is store-only"

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = ExampleHiddenStatesConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            filename = os.path.join(self._storage_path, f"{new_req.req_id}.safetensors")
            meta.add_request(
                new_req.req_id,
                filename=filename,
                token_ids=token_ids,
            )
            self._request_filenames[new_req.req_id] = filename
            self._active_requests[new_req.req_id] = new_req

        # Include cached (continuing) requests so the worker-side
        # save_hidden_states can match metadata entries to batch order.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            if req_id not in self._active_requests:
                continue
            cached_req = self._active_requests[req_id]
            filename = os.path.join(self._storage_path, f"{req_id}.safetensors")
            meta.add_request(
                req_id=req_id,
                filename=filename,
                token_ids=cached_req.prompt_token_ids or [],
                new_req=False,
            )

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        filename = self._request_filenames.pop(req_id, None)
        _ = self._active_requests.pop(req_id, None)
        # File was already written by save_hidden_states() on the worker.
        return False, {"hidden_states_path": filename}

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        flat = block_ids[0] if block_ids else []
        return self.request_finished(request, flat)

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        if cls is KVConnectorBase_V1:
            raise TypeError(
                "get_required_kvcache_layout should not be called "
                "on the abstract base class"
            )
        return "NHD"
