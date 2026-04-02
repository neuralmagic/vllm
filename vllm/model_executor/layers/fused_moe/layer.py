# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.distributed import (
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
)
from vllm.model_executor.layers.fused_moe.eplb_manager import EplbManager
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    ExpertMapManager,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner_factory import (
    create_moe_runner,
)
from vllm.model_executor.layers.fused_moe.utils import (
    disable_inplace,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

logger = init_logger(__name__)


# TODO: merge with class method
def determine_expert_placement_strategy(
    expert_placement_strategy: ExpertPlacementStrategy,
    moe_parallel_config: FusedMoEParallelConfig,
    num_expert_group: int | None,
    num_redundant_experts: int,
    enable_eplb: bool,
) -> ExpertPlacementStrategy:
    if expert_placement_strategy == "round_robin":
        round_robin_supported = (
            (num_expert_group is not None and num_expert_group > 1)
            and num_redundant_experts == 0
            and not enable_eplb
        )

        if not round_robin_supported:
            logger.warning(
                "Round-robin expert placement is only supported for "
                "models with multiple expert groups and no redundant "
                "experts. Falling back to linear expert placement."
            )
            return "linear"
        if (
            moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.use_deepep_ll_kernels
            and not moe_parallel_config.use_nixl_ep_kernels
        ):
            logger.warning(
                "Round-robin expert placement currently only supports "
                "the DeepEP low-latency or NIXL EP backend, but '%s' was configured. "
                "Falling back to linear expert placement.",
                moe_parallel_config.all2all_backend,
            )
            return "linear"

    return expert_placement_strategy


# TODO(rob): move this down to the kernel.
def maybe_roundup_hidden_size(
    hidden_size: int,
    act_dtype: torch.dtype,
    moe_parallel_config: FusedMoEParallelConfig,
    is_lora_enabled: bool,
    model_type: str | None,
) -> int:
    """
    Given layer hidden size and MoE configurations, round up hidden_size
    if necessary.

    Args:
        hidden_size: Layer hidden-size
        act_dtype: Data type of the layer activations.
        moe_parallel_config: Fused MoE parallelization strategy configuration.
        is_lora_enabled: True if the engine is enabled with LoRA. This
            is used in the case of mxfp4 quantization in selecting the
            MxFP4Backend.
        model_type: for checking if gpt-oss

    Return:
        Rounded up hidden_size if rounding up is required based on the configs.
        Original hidden size otherwise.
    """
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_roundup_layer_hidden_size,
    )

    hidden_size = maybe_roundup_layer_hidden_size(
        hidden_size, act_dtype, moe_parallel_config
    )

    return hidden_size


def register_layer_for_moe_forward_op(
    vllm_config: VllmConfig,
    layer: torch.nn.Module,  # FusedMoE for now
):
    # For smuggling this layer into the fused moe custom op
    prefix = layer.layer_name
    compilation_config = vllm_config.compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError("Duplicate layer name: {}".format(prefix))
    compilation_config.static_forward_context[prefix] = layer
    compilation_config.static_all_moe_layers.append(prefix)


# --8<-- [start:fused_moe]
@CustomOp.register("fused_moe")
class FusedMoE(CustomOp):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        renormalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
        enable_eplb: Whether to enable expert parallelism load balancer.
        router_logits_dtype: Data type for router logits buffers.
    """

    # --8<-- [end:fused_moe]

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        quant_config: QuantizationConfig | None = None,
        tp_size: int | None = None,
        ep_size: int | None = None,
        dp_size: int | None = None,
        pcp_size: int | None = None,
        prefix: str = "",
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        is_act_and_mul: bool = True,
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
        has_bias: bool = False,
        is_sequence_parallel=False,
        expert_mapping: list[tuple[str, str, int, str]] | None = None,
        n_shared_experts: int | None = None,
        router_logits_dtype: torch.dtype | None = None,
        gate: torch.nn.Module | None = None,
        shared_experts: torch.nn.Module | None = None,
        routed_input_transform: torch.nn.Module | None = None,
        routed_output_transform: torch.nn.Module | None = None,
        apply_scale_to_output: bool = False,
        zero_expert_type: str | None = None,
    ):
        super().__init__()

        # IMPORTANT: RoutedExperts must have same layer_name/prefix as FusedMoE for now
        # This is still needed
        self.layer_name = prefix

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        vllm_config = get_current_vllm_config()

        # FIXME (varun): We should have a better way of inferring the activation
        # datatype. This works for now as the tensor datatype entering the MoE
        # operation is typically unquantized (i.e. float16/bfloat16).
        if vllm_config.model_config is not None:
            moe_in_dtype = vllm_config.model_config.dtype
        else:
            # TODO (bnell): This is a hack to get test_mixtral_moe to work
            # since model_config is not set in the pytest test.
            moe_in_dtype = params_dtype

        tp_size_ = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        dp_size_ = dp_size if dp_size is not None else get_dp_group().world_size
        pcp_size_ = pcp_size if pcp_size is not None else get_pcp_group().world_size

        is_sequence_parallel = is_sequence_parallel
        sp_size = tp_size_ if is_sequence_parallel else 1

        moe_parallel_config = FusedMoEParallelConfig.make(
            tp_size_=tp_size_,
            pcp_size_=pcp_size_,
            dp_size_=dp_size_,
            sp_size_=sp_size,
            vllm_parallel_config=vllm_config.parallel_config,
        )

        assert moe_parallel_config.is_sequence_parallel == is_sequence_parallel

        logger.debug("FusedMoEParallelConfig = %s", str(moe_parallel_config))

        global_num_experts = num_experts + num_redundant_experts
        logical_num_experts = num_experts

        # Initialize EPLB manager (or None?)
        eplb_manager: EplbManager | None = None
        if enable_eplb:
            eplb_manager = EplbManager(num_redundant_experts=num_redundant_experts)

        expert_placement_strategy: ExpertPlacementStrategy = (
            vllm_config.parallel_config.expert_placement_strategy
        )

        # ROCm aiter shared experts fusion
        # AITER only supports gated activations (silu/gelu), so disable it
        # for non-gated MoE (is_act_and_mul=False)
        rocm_aiter_fmoe_enabled = (
            rocm_aiter_ops.is_fused_moe_enabled() and is_act_and_mul
        )
        aiter_fmoe_shared_expert_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled() and is_act_and_mul
        )

        num_fused_shared_experts = (
            n_shared_experts
            if n_shared_experts is not None and aiter_fmoe_shared_expert_enabled
            else 0
        )
        if not aiter_fmoe_shared_expert_enabled and num_fused_shared_experts != 0:
            raise ValueError(
                "n_shared_experts is only supported on ROCm aiter when "
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled"
            )

        # Determine expert maps
        if moe_parallel_config.use_ep:
            if eplb_manager is not None:
                # Validate EPLB configuration
                eplb_manager.validate_configuration(global_num_experts, ep_size)
            else:
                assert num_redundant_experts == 0, (
                    "Redundant experts are only supported with EPLB."
                )

            # Determine expert placement strategy before creating manager
            expert_placement_strategy_effective = determine_expert_placement_strategy(
                expert_placement_strategy=expert_placement_strategy,
                moe_parallel_config=moe_parallel_config,
                num_expert_group=num_expert_group,
                num_redundant_experts=num_redundant_experts,
                enable_eplb=eplb_manager is not None,
            )
        else:
            expert_placement_strategy_effective = expert_placement_strategy

        # Create expert map manager
        self.expert_map_manager = ExpertMapManager(
            max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
            top_k=top_k,
            global_num_experts=global_num_experts,
            logical_num_experts=logical_num_experts,
            moe_parallel_config=moe_parallel_config,
            placement_strategy=expert_placement_strategy_effective,
            num_fused_shared_experts=num_fused_shared_experts,
            rocm_aiter_enabled=rocm_aiter_fmoe_enabled,
            device=vllm_config.device_config.device,
        )

        tp_size = moe_parallel_config.tp_size
        assert intermediate_size % tp_size == 0
        intermediate_size_per_partition = intermediate_size // tp_size

        self._runner: MoERunner

        # TODO(bnell): we should not have to create a router if the kernel is
        # monolithic.
        router = create_fused_moe_router(
            top_k=top_k,
            global_num_experts=global_num_experts,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor
            if not apply_scale_to_output
            else 1.0,
            e_score_correction_bias=e_score_correction_bias,
            num_fused_shared_experts=num_fused_shared_experts,
            eplb_manager=eplb_manager,
            # TODO(bnell): once we can construct the MK at init time, we
            # can make this a value.
            indices_type_getter=lambda: self._runner.routed_experts.quant_method.topk_indices_dtype,  # noqa: E501
            zero_expert_type=zero_expert_type,
            num_logical_experts=logical_num_experts,
        )

        # When using zero experts, slice e_score_correction_bias to cover
        # only real experts, for compatibility with monolithic kernels that
        # read it directly.
        if zero_expert_type is not None and e_score_correction_bias is not None:
            self.e_score_correction_bias = e_score_correction_bias[logical_num_experts]

        # Round up hidden size before creating moe_config.
        # This way moe_config is created with the correct hidden_size from the start.
        unpadded_hidden_size = hidden_size
        model_type = (
            vllm_config.model_config.hf_config.model_type
            if vllm_config.model_config is not None
            else None
        )
        hidden_size = maybe_roundup_hidden_size(
            hidden_size=hidden_size,
            act_dtype=moe_in_dtype,
            moe_parallel_config=moe_parallel_config,
            is_lora_enabled=vllm_config.lora_config is not None,
            model_type=model_type,
        )

        moe_activation = MoEActivation.from_str(activation)

        moe_config = FusedMoEConfig(
            num_experts=global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            num_local_experts=self.expert_map_manager.local_num_experts,
            num_logical_experts=logical_num_experts,
            moe_parallel_config=moe_parallel_config,
            in_dtype=moe_in_dtype,
            moe_backend=vllm_config.kernel_config.moe_backend,
            router_logits_dtype=router_logits_dtype,
            max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
            has_bias=has_bias,
            is_act_and_mul=is_act_and_mul,
            is_lora_enabled=vllm_config.lora_config is not None,
            activation=moe_activation,
            device=vllm_config.device_config.device,
            routing_method=router.routing_method_type,
            # TODO: in_dtype == out_dtype?
            disable_inplace=disable_inplace() or shared_experts is not None,
        )

        quant_config = quant_config

        logger.debug("FusedMoEConfig = %s", moe_config)

        # Create RoutedExperts instance BEFORE create_weights()
        # This will hold all expert weight parameters
        routed_experts = RoutedExperts(
            self.layer_name,
            params_dtype,
            unpadded_hidden_size,
            intermediate_size,
            moe_config,
            quant_config,
            expert_map_manager=self.expert_map_manager,
            # Extra params that are needed by quant_methods, pass along for now
            rocm_aiter_fmoe_enabled=rocm_aiter_fmoe_enabled,  # get from moe config
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,  # get from router?
            apply_router_weight_on_input=apply_router_weight_on_input,
            activation=moe_activation,
        )

        # TODO(bnell): this needs to be stored as a parameter for weight loading.
        # ditch this eventually.
        self.routed_experts = routed_experts

        # Where to move this?
        quant_method = routed_experts.quant_method
        if enable_eplb and not quant_method.supports_eplb:
            # TODO: Add support for additional quantization methods.
            # The implementation for other quantization methods does not
            # contain essential differences, but the current quant API
            # design causes duplicated work when extending to new
            # quantization methods, so I'm leaving it for now.
            # If you plan to add support for more quantization methods,
            # please refer to the implementation in `Fp8MoEMethod`.
            raise NotImplementedError(
                f"EPLB is not supported {quant_method.__class__.__name__}."
            )

        # Storing the runner in the FusedMoE is an intermediate state, eventually
        # the runner will own the FusedMoE layer and provide the execution interface
        # for MoE ops.
        self._runner = create_moe_runner(
            layer_name=self.layer_name,
            moe_config=moe_config,
            router=router,
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
            gate=gate,
            shared_experts=shared_experts,
            routed_experts=routed_experts,
            enable_dbo=vllm_config.parallel_config.enable_dbo,
            apply_scale_to_output=apply_scale_to_output,
            routed_scaling_factor=routed_scaling_factor,
        )

        # HACK XXXXXXXXXXXXXXXXXXXXXXXX
        routed_experts.shared_experts = self._runner.shared_experts

        # For smuggling this layer into the fused moe custom op
        register_layer_for_moe_forward_op(vllm_config, self)

    # TODO(bnell): This method is provided as a hook so vllm/lora/layers/fused_moe.py
    # and vllm/distributed/elastic_ep/elastic_execute.py
    # can safely swap out the quant_method. We should figure out a less
    # intrusive way to do this.
    def _replace_quant_method(self, mk: FusedMoEMethodBase):
        self._runner._replace_quant_method(mk)

    # Note: maybe_init_modular_kernel should only be called by
    # prepare_communication_buffer_for_model.
    # This is called after all weight loading and post-processing, so it
    # should be safe to swap out the quant_method.
    def maybe_init_modular_kernel(self) -> None:
        # NOTE(rob): WIP refactor. For quant methods that own the MK
        # we create the MK during process_weights_after_loading.
        if (
            self._runner.routed_experts.quant_method.supports_internal_mk
            or self._runner.routed_experts.quant_method.is_monolithic
        ):
            return None

        self._runner.routed_experts._ensure_moe_quant_config_init()
        # routing_tables only needed for round-robin expert placement with
        # DeepEP all2all backend.
        routing_tables = self._maybe_init_expert_routing_tables()

        if isinstance(self._runner.routed_experts.quant_method, FusedMoEModularMethod):
            base_quant_method = (
                self._runner.routed_experts.quant_method.old_quant_method
            )
        else:
            base_quant_method = self._runner.routed_experts.quant_method

        prepare_finalize = base_quant_method.maybe_make_prepare_finalize(
            routing_tables=routing_tables
        )
        if prepare_finalize is not None:
            logger.debug(
                "%s for %s(%s)", prepare_finalize.__class__.__name__, self, id(self)
            )
            self._replace_quant_method(
                FusedMoEModularMethod.make(
                    self,
                    base_quant_method,
                    prepare_finalize,
                    self._runner.shared_experts,
                    inplace=not base_quant_method.moe.disable_inplace,
                )
            )

    #
    # Properties
    #

    @property
    def layer_id(self):
        # Delayed import to avoid circular dependency
        from vllm.model_executor.models.utils import extract_layer_index

        return extract_layer_index(self.layer_name)

    #
    # Attributes still needed by models
    #

    @property
    def is_monolithic(self) -> bool:
        return self._runner.routed_experts.quant_method.is_monolithic

    @property
    def activation(self) -> MoEActivation:
        return self._runner.routed_experts.activation

    @property
    def is_internal_router(self) -> bool:
        # By default, router/gate is called before FusedMoE forward pass
        return self._runner.is_internal_router

    #
    # Expert maps
    #

    @property
    def expert_placement_strategy(self) -> ExpertPlacementStrategy:
        return self.expert_map_manager.placement_strategy

    @property
    def expert_global_to_physical(self) -> torch.Tensor | None:
        tables = self.expert_map_manager.routing_tables
        return tables[0] if tables else None

    @property
    def expert_physical_to_global(self) -> torch.Tensor | None:
        """Routing table: physical expert ID to global expert ID."""
        tables = self.expert_map_manager.routing_tables
        return tables[1] if tables else None

    @property
    def expert_local_to_global(self) -> torch.Tensor | None:
        """Routing table: local expert ID to global expert ID."""
        tables = self.expert_map_manager.routing_tables
        return tables[2] if tables else None

    @property
    def expert_map(self) -> torch.Tensor | None:
        return self._runner.routed_experts.expert_map

    def _maybe_init_expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        return self._runner.routed_experts._maybe_init_expert_routing_tables()

    def update_expert_map(self):
        self._runner.routed_experts.update_expert_map()

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        """Map global expert ID to local expert ID."""
        return self._runner.routed_experts._map_global_expert_id_to_local_expert_id(
            expert_id
        )

    #
    # EPLB
    #

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        """Delegate to EPLB manager."""
        if self._runner.router.eplb_manager is not None:
            return self._runner.router.eplb_manager.get_expert_weights(self)
        else:
            return []

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        """
        Register the EPLB state in this layer.

        This is used later in forward pass, where we get the expert mapping
        and record the load metrics in `expert_load_view`.
        """
        if self._runner.router.eplb_manager is not None:
            self._runner.router.eplb_manager.set_state(
                moe_layer_idx,
                expert_load_view,
                logical_to_physical_map,
                logical_replica_count,
            )

    @classmethod
    def make_expert_params_mapping(
        cls,
        model: torch.nn.Module,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        num_redundant_experts: int = 0,
    ) -> list[tuple[str, str, int, str]]:
        """Delegate to EPLB manager."""
        return RoutedExperts.make_expert_params_mapping(
            model,
            ckpt_gate_proj_name,
            ckpt_down_proj_name,
            ckpt_up_proj_name,
            num_experts,
            num_redundant_experts,
        )

    #
    # Execution
    #

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        return self._runner.forward(
            hidden_states,
            router_logits,
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(hidden_states, router_logits)
