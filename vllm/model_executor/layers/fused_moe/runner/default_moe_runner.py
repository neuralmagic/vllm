# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from contextlib import nullcontext

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.distributed import (
    get_ep_group,
    get_pcp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import (
    aux_stream,
    current_stream,
    direct_register_custom_op,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)


def unpack_pair(
    y: torch.Tensor | None,
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(x, tuple):
        if y is not None:
            assert x[0] is None
            return y, x[1]
        else:
            return x
    else:
        return y, x


class DefaultMoERunner(MoERunner):
    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        moe_quant_config: FusedMoEQuantConfig | None,
        router: FusedMoERouter,
        gate: torch.nn.Module | None,
        shared_experts: torch.nn.Module | None,
        quant_method: FusedMoEMethodBase,
        reduce_results: bool,
        enable_dbo: bool,
        capture: Callable[[torch.Tensor], None] | None = None,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.moe_quant_config = moe_quant_config
        self.router = router
        self.gate = gate
        self.shared_experts = shared_experts
        self.quant_method = quant_method
        self.reduce_results = reduce_results
        self.enable_dbo = enable_dbo
        self.capture = capture

        # Chunked all2all staging tensor
        self.batched_hidden_states: torch.Tensor | None = None
        self.batched_router_logits: torch.Tensor | None = None

        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        self.use_shared_experts_stream = False
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.debug_once("Disabling MoE shared_experts cuda stream", scope="local")
            self.shared_experts_stream = None
        else:
            # TODO(rob): enable shared expert overlap with non-cuda-alike.
            # aux_stream() returns None on non-cuda-alike platforms.
            self.shared_experts_stream = aux_stream()
            if self.shared_experts_stream is not None:
                logger.debug_once(
                    "Enabled separate cuda stream for MoE shared_experts", scope="local"
                )

        self._use_flashinfer_cutlass_kernels = (
            self.moe_config.use_flashinfer_cutlass_kernels
        )

        self.moe_forward = self._select_forward(layer)

    def _select_forward(self, layer: torch.nn.Module) -> Callable:
        # Note: these are local functions so they capture the layer.
        def _moe_forward(
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
        ) -> torch.Tensor:
            router_logits = self._maybe_gate(hidden_states, router_logits)
            with self._sequence_parallel_context():
                if self.use_dp_chunking:
                    return self.forward_impl_chunked(
                        layer, hidden_states, router_logits
                    )
                else:
                    return self.forward_impl(layer, hidden_states, router_logits)

        def _moe_forward_shared(
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            router_logits = self._maybe_gate(hidden_states, router_logits)
            with self._sequence_parallel_context():
                if self.use_dp_chunking:
                    return self.forward_impl_chunked(
                        layer, hidden_states, router_logits
                    )
                else:
                    return self.forward_impl(layer, hidden_states, router_logits)

        # TODO: Once the OOM issue for the TPU backend is resolved, we will
        # switch to using the moe_forward custom op.
        # Note: CPU doesn't require wrapped forward_impl.
        if current_platform.is_tpu() or current_platform.is_cpu():
            return _moe_forward if self.shared_experts is None else _moe_forward_shared

        op_name = f"moe_forward{layer.layer_name.replace('.', '_')}"

        if not hasattr(torch.ops.vllm, op_name):
            if self.shared_experts is None:
                fn = _moe_forward
                fake_fn = DefaultMoERunner._moe_forward_fake
            else:
                fn = _moe_forward_shared
                fake_fn = DefaultMoERunner._moe_forward_shared_fake

            direct_register_custom_op(
                op_name=op_name,
                op_func=fn,
                mutates_args=["hidden_states"],
                fake_impl=fake_fn,
                tags=(torch.Tag.needs_fixed_stride_order,),
            )

        return getattr(torch.ops.vllm, op_name)

    @staticmethod
    def _moe_forward_fake(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states)

    @staticmethod
    def _moe_forward_shared_fake(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared_out = torch.empty_like(hidden_states)
        fused_out = torch.empty_like(hidden_states)
        return shared_out, fused_out

    @property
    def use_flashinfer_cutlass_kernels(self):
        return (
            self.moe_quant_config is not None
            and self.moe_quant_config.quant_dtype == "nvfp4"
            and self._use_flashinfer_cutlass_kernels
        )

    # Note: this needs to be a runtime method because it requires
    # a moe_quant_config which may not be set up til later.
    @property
    def use_dp_chunking(self) -> bool:
        return (
            self.moe_config.use_pplx_kernels
            or self.moe_config.use_deepep_ll_kernels
            or self.moe_config.use_mori_kernels
            or (self.moe_config.dp_size > 1 and self.use_flashinfer_cutlass_kernels)
        ) and envs.VLLM_ENABLE_MOE_DP_CHUNK

    def _maybe_setup_shared_experts_stream(
        self,
        hidden_states: torch.Tensor,
        use_chunked_impl: bool,
    ) -> torch.Tensor | None:
        self.use_shared_experts_stream = (
            current_platform.is_cuda()
            and self.has_separate_shared_experts
            and not use_chunked_impl
            and self.shared_experts_stream is not None
            and (
                hidden_states.shape[0]
                <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            )
        )

        hidden_states_clone: torch.Tensor | None = None
        if self.use_shared_experts_stream:
            assert self.shared_experts_stream is not None

            # Clone BEFORE switching streams to avoid race condition
            # where routed_expert kernel may mutate hidden_states.
            hidden_states_clone = hidden_states.clone()

            # Record that the clone will be used by shared_experts_stream
            # to avoid gc issue from deallocation of hidden_states_clone
            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
            # NOTE: We don't need shared_output.record_stream(current_stream())
            # because we synch the streams before using shared_output.
            hidden_states_clone.record_stream(self.shared_experts_stream)

            # Mark sync start point for the separate shared experts
            # stream here since we want to run in parallel with the
            # router/gate (next op below)
            assert self.shared_experts_stream is not None
            self.shared_experts_stream.wait_stream(current_stream())

        return hidden_states_clone

    @property
    def has_separate_shared_experts(self) -> bool:
        return (
            not isinstance(self.quant_method, FusedMoEModularMethod)
            and self.shared_experts is not None
        )

    def _apply_shared_experts(
        self,
        shared_output: torch.Tensor | None,
        hidden_states: torch.Tensor,
        allow_streaming: bool = False,
    ) -> torch.Tensor | None:
        if self.has_separate_shared_experts and shared_output is None:
            assert self.shared_experts is not None

            if self.use_shared_experts_stream and allow_streaming:
                # Run shared experts in parallel on a separate stream
                # NOTE: We start the separate stream here and mark the
                # sync end point immediately after it is done. This is
                # important to avoid excessive stream allocations by the cuda
                # graph replay later.
                with torch.cuda.stream(self.shared_experts_stream):
                    # Note that hidden_states clone() is necessary here to avoid
                    # conflict with the main stream
                    shared_output = self.shared_experts(hidden_states)
                current_stream().wait_stream(self.shared_experts_stream)
            else:
                shared_output = self.shared_experts(hidden_states)

        return shared_output

    def _ensure_dp_chunking_init(self):
        if not self.use_dp_chunking or self.batched_hidden_states is not None:
            return

        states_shape: tuple[int, ...]
        logits_shape: tuple[int, ...]

        moe = self.moe_config

        if self.enable_dbo:
            states_shape = (2, moe.max_num_tokens, self.moe_config.hidden_dim)
            logits_shape = (2, moe.max_num_tokens, self.moe_config.num_logical_experts)
        else:
            states_shape = (moe.max_num_tokens, self.moe_config.hidden_dim)
            logits_shape = (moe.max_num_tokens, self.moe_config.num_logical_experts)

        self.batched_hidden_states = torch.zeros(
            states_shape, dtype=moe.in_dtype, device=torch.cuda.current_device()
        )

        self.batched_router_logits = torch.zeros(
            logits_shape,
            dtype=moe.router_logits_dtype,
            device=torch.cuda.current_device(),
        )

    def must_reduce_shared_expert_outputs(self) -> bool:
        """
        The shared_experts are typically computed using the RowParallelLinear
        layer. The result of this function is typically used as
        the reduce_results argument to the module.
        When just tensor-parallel is used, it is not required to reduce
        the shared_experts results immediately. Instead we reduce at the
        once at the end of the MoE op. (Refer to DeepSeekV2MoE module)
        With EP and all2all kernels - this is no longer viable as all
        GPU ranks in DP, produce the complete set of hidden_states.
        Therefore it is required that we reduce the shared_experts output
        early.
        """
        return (
            isinstance(self.quant_method, FusedMoEModularMethod)
            and self.quant_method.fused_experts.output_is_reduced()
        )

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        """
        Some combine kernels reduce across GPU ranks by default.
        """
        if self.must_reduce_shared_expert_outputs():
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def _maybe_reduce_output(
        self,
        states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        trunc_dim: int,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        def trunc(x: torch.Tensor) -> torch.Tensor:
            return x[..., :trunc_dim]

        def reduce_and_trunc(x: torch.Tensor) -> torch.Tensor:
            return trunc(self.maybe_all_reduce_tensor_model_parallel(x))

        if (
            not self.moe_config.is_sequence_parallel
            and not self.use_dp_chunking
            and self.reduce_results
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
        ):
            func = reduce_and_trunc
        else:
            func = trunc

        if isinstance(states, tuple):
            return tuple([func(s) for s in states])
        else:
            return func(states)

    def _maybe_pad_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        og_hidden_states = hidden_states.shape[-1]
        if self.moe_config.hidden_dim != og_hidden_states:
            hidden_states = F.pad(
                hidden_states,
                (0, self.moe_config.hidden_dim - og_hidden_states),
                mode="constant",
                value=0.0,
            )
        return hidden_states, og_hidden_states

    def quant_method_apply(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        extra_tensor: torch.Tensor | None,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # TODO(bnell): deal with fp4 flashinfer tuple hidden states hack (#30014).
        # Figure out nicer way to do this.
        x_arg = hidden_states if extra_tensor is None else (hidden_states, extra_tensor)

        if self.quant_method.is_monolithic:
            return self.quant_method.apply_monolithic(
                layer=layer,
                x=x_arg,
                router_logits=router_logits,
            )
        else:
            topk_weights, topk_ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

            if self.capture is not None:
                self.capture(topk_ids)

            return self.quant_method.apply(
                layer=layer,
                x=x_arg,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )

    def _sequence_parallel_context(self):
        ctx = get_forward_context()
        return (
            ctx.dp_metadata.sp_local_sizes(self.moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

    def _maybe_gate(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        # If router/gate provided, then apply it here.
        # (Note: This code runs only when "overlapped mode" is on to allow
        #        parallel execution of shared experts with the FusedMoE via
        #        separate cuda stream)
        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)
        return router_logits

    @property
    def do_naive_dispatch_combine(self) -> bool:
        return self.moe_config.dp_size > 1 and not isinstance(
            self.quant_method, FusedMoEModularMethod
        )

    def _maybe_dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        extra_tensor: torch.Tensor | None = None

        if self.do_naive_dispatch_combine:
            post_quant_allgather = (
                self.moe_config.dp_size > 1
                and self.moe_config.use_ep
                and getattr(self.quant_method, "do_post_quant_allgather", False)
            )
            if post_quant_allgather:
                hidden_states_to_dispatch, extra_tensors = (
                    self.quant_method.prepare_dp_allgather_tensor(
                        self, hidden_states, router_logits
                    )
                )
            else:
                hidden_states_to_dispatch = hidden_states

            hidden_states, router_logits, extra_tensors_dispatched = (
                get_ep_group().dispatch(
                    hidden_states_to_dispatch,
                    router_logits,
                    self.moe_config.is_sequence_parallel,
                    extra_tensors=extra_tensors,
                )
            )

            if extra_tensors_dispatched is not None:
                assert len(extra_tensors_dispatched) == 1
                extra_tensor = extra_tensors_dispatched[0]

        # NOTE: Similar with DP, PCP also needs dispatch and combine. For
        # simplicity, AgRsAll2All was added separately for PCP here. Maybe
        # we should modify All2AllManager abstract to better support PCP.
        if self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().all_gather(
                hidden_states,
                dim=0,
            )
            router_logits = get_pcp_group().all_gather(
                router_logits,
                dim=0,
            )

        return hidden_states, router_logits, extra_tensor

    def _maybe_combine(
        self,
        shared_output: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        if self.do_naive_dispatch_combine:
            hidden_states = get_ep_group().combine(
                hidden_states, self.moe_config.is_sequence_parallel
            )

        if self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().reduce_scatter(
                hidden_states,
                dim=0,
            )

        if self.shared_experts is not None:
            assert shared_output is not None
            return shared_output, hidden_states
        else:
            return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden_states, og_hidden_states = self._maybe_pad_hidden_states(hidden_states)
        fused_output = self.moe_forward(hidden_states, router_logits)
        return self._maybe_reduce_output(fused_output, og_hidden_states)

    def forward_impl_chunked(
        self,
        layer: torch.nn.Module,
        full_hidden_states: torch.Tensor,
        full_router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._ensure_dp_chunking_init()

        assert self.batched_hidden_states is not None
        assert self.batched_router_logits is not None
        assert self.batched_hidden_states.dtype == full_hidden_states.dtype, (
            f"{self.batched_hidden_states.dtype} == {full_hidden_states.dtype}"
        )
        assert self.batched_router_logits.dtype == full_router_logits.dtype, (
            f"{self.batched_router_logits.dtype} == {full_router_logits.dtype}"
        )
        # Check size compatibility.
        assert self.batched_hidden_states.size(-1) == full_hidden_states.size(-1)
        assert self.batched_router_logits.size(-1) == full_router_logits.size(-1)

        full_fused_final_hidden_states = torch.empty_like(full_hidden_states)
        if self.shared_experts is not None:
            full_shared_final_hidden_states = torch.empty_like(full_hidden_states)

        def process_chunk(chunk_start, chunk_end, skip_result_store=False):
            chunk_size = chunk_end - chunk_start
            hidden_states = full_hidden_states[chunk_start:chunk_end, :]
            router_logits = full_router_logits[chunk_start:chunk_end, :]

            assert self.batched_hidden_states is not None
            assert self.batched_router_logits is not None
            # This is only true when DBO has been enabled in the config.
            # Both tensors will have an outer dimension for the ubatch id
            if self.batched_hidden_states.dim() == 3:
                assert self.batched_router_logits.dim() == 3
                batch_buffer_idx = dbo_current_ubatch_id()
                batched_hidden_states = self.batched_hidden_states[batch_buffer_idx, :]
                batched_router_logits = self.batched_router_logits[batch_buffer_idx, :]
            else:
                batched_hidden_states = self.batched_hidden_states
                batched_router_logits = self.batched_router_logits

            assert (
                batched_hidden_states.size(0)  # type: ignore
                >= chunk_size
            )
            assert (
                batched_router_logits.size(0)  # type: ignore
                >= chunk_size
            )
            staged_hidden_states = batched_hidden_states[:chunk_size, :]  # type: ignore
            staged_router_logits = batched_router_logits[:chunk_size, :]  # type: ignore
            staged_hidden_states.copy_(hidden_states, non_blocking=True)
            staged_router_logits.copy_(router_logits, non_blocking=True)

            shared_output, hidden_states = unpack_pair(
                None,
                self.quant_method_apply(
                    layer=layer,
                    hidden_states=staged_hidden_states,
                    extra_tensor=None,
                    router_logits=staged_router_logits,
                ),
            )

            shared_output = self._apply_shared_experts(
                shared_output, staged_hidden_states
            )

            if not skip_result_store:
                if self.shared_experts is None:
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        hidden_states, non_blocking=True
                    )
                else:
                    assert shared_output is not None
                    full_shared_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        shared_output, non_blocking=True
                    )
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        hidden_states, non_blocking=True
                    )

        ctx = get_forward_context()
        # flashinfer_cutlass_kernels can handle: optional DP + TP/EP
        max_tokens_across_dispatchers = ctx.dp_metadata.max_tokens_across_dp_cpu
        moe_dp_chunk_size_per_rank = self.moe_config.max_num_tokens

        # If the input to the MoE is sequence parallel then divide by sp_size
        # to find the maximum number of tokens for any individual dispatcher.
        if self.moe_config.is_sequence_parallel:
            max_tokens_across_dispatchers = cdiv(
                max_tokens_across_dispatchers, self.moe_config.sp_size
            )

        num_tokens = full_hidden_states.size(0)
        for chunk_idx, chunk_start_ in enumerate(
            range(0, max_tokens_across_dispatchers, moe_dp_chunk_size_per_rank)
        ):
            chunk_start = chunk_start_
            chunk_end = min(
                chunk_start + moe_dp_chunk_size_per_rank, max_tokens_across_dispatchers
            )
            # clamp start and end
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)
            with ctx.dp_metadata.chunked_sizes(
                self.moe_config.sp_size, moe_dp_chunk_size_per_rank, chunk_idx
            ):
                process_chunk(
                    chunk_start, chunk_end, skip_result_store=chunk_start_ >= num_tokens
                )

        if self.shared_experts is None:
            return full_fused_final_hidden_states
        else:
            return (full_shared_final_hidden_states, full_fused_final_hidden_states)

    def forward_impl(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        shared_output: torch.Tensor | None = None

        # TODO: set this up better so that we don't need to clone, i.e.
        # disable inplace?
        hidden_states_clone = self._maybe_setup_shared_experts_stream(
            hidden_states, self.use_dp_chunking
        )

        # Check if we need to run shared experts before matrix multiply because
        # matrix multiply may modify the hidden_states.
        run_shared_experts_before = (
            self.has_separate_shared_experts and not self.use_shared_experts_stream
        )

        # TODO(bnell): the dispatch/combine steps should go away once
        # #32567 lands and the remaining kernels are made MKs.
        hidden_states, router_logits, extra_tensor = self._maybe_dispatch(
            hidden_states,
            router_logits,
        )

        if run_shared_experts_before:
            shared_output = self._apply_shared_experts(
                shared_output,
                hidden_states,
                False,  # TODO: why don't we use streaming here?
            )

        shared_output, hidden_states = unpack_pair(
            shared_output,
            self.quant_method_apply(
                layer=layer,
                hidden_states=hidden_states,
                extra_tensor=extra_tensor,
                router_logits=router_logits,
            ),
        )

        if not run_shared_experts_before:
            shared_output = self._apply_shared_experts(
                shared_output,
                hidden_states_clone,
                True,
            )

        # TODO(bnell): will go away
        return self._maybe_combine(
            shared_output,
            hidden_states,
        )
