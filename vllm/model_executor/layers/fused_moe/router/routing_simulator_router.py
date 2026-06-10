# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import Any

import torch

import vllm.envs as envs
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

logger = init_logger(__name__)


class RoutingStrategy(ABC):
    """Base class for token-to-expert routing strategies."""

    @abstractmethod
    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            hidden_states: Input hidden states [num_tokens, hidden_size]
            router_logits: Router logits [num_tokens, num_experts]
            top_k: Number of experts to select per token
            indices_type: Data type for expert indices

        Returns:
            tuple of (topk_weights, topk_ids)
        """
        pass


class DistributionBasedRouting(RoutingStrategy):
    """
    Distribution-based random routing strategy with configurable distributions.

    This routing strategy randomly selects experts for each token based on
    different probability distributions. Currently supports uniform and normal
    distributions for testing different routing patterns.
    """

    def __init__(self, distribution: str = "uniform", **distribution_params: Any):
        """
        Initialize distribution-based routing.

        Args:
            distribution: Type of distribution to use for sampling
                - "uniform": Uniform distribution (default)
                - "normal": Normal/Gaussian distribution
            **distribution_params: Parameters specific to the
                chosen distribution
                For "uniform": No additional parameters needed
                For "normal": mean (default: 0.0), std (default: 1.0)
        """
        self.distribution = distribution.lower()
        self.distribution_params = distribution_params

        # Validate distribution and parameters
        self._validate_distribution_params()

    def _validate_distribution_params(self):
        """Validate distribution type and parameters."""
        valid_distributions = ["uniform", "normal"]

        if self.distribution not in valid_distributions:
            raise ValueError(
                f"Unsupported distribution: {self.distribution}. "
                f"Supported distributions: {valid_distributions}"
            )

        # Set default parameters if not provided
        if self.distribution == "normal":
            self.distribution_params.setdefault("mean", 0.0)
            self.distribution_params.setdefault("std", 1.0)

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly select experts for each token using the specified distribution.

        Args:
            hidden_states: Input hidden states [num_tokens, hidden_size]
            router_logits: Router logits [num_tokens, num_experts]
            top_k: Number of experts to select per token
            indices_type: Data type for expert indices

        Returns:
            tuple of (topk_weights, topk_ids) where:
            - topk_weights: Weights based on distribution sampling
            - topk_ids: Expert indices sampled from the distribution
        """
        num_tokens = hidden_states.shape[0]
        num_experts = router_logits.shape[-1]

        if indices_type is None:
            indices_type = torch.long

        # Generate expert IDs based on the specified distribution
        topk_ids = self._sample_expert_ids(
            num_tokens, num_experts, top_k, hidden_states.device, indices_type
        )

        # Generate weights based on the distribution
        topk_weights = self._generate_weights(num_tokens, top_k, hidden_states.device)

        return topk_weights, topk_ids

    def _sample_expert_ids(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        device: torch.device,
        indices_type: torch.dtype,
    ) -> torch.Tensor:
        """Sample expert IDs based on the specified distribution.

        All distributions sample without replacement so that each token
        is routed to top_k *distinct* experts, matching real MoE semantics.
        """

        if self.distribution == "uniform":
            probs = torch.ones(
                num_tokens, num_experts, device=device, dtype=torch.float32
            )
            return torch.multinomial(probs, num_samples=top_k, replacement=False).to(
                dtype=indices_type
            )

        elif self.distribution == "normal":
            mean = self.distribution_params["mean"]
            std = self.distribution_params["std"]
            # Per-expert probabilities derived from a normal distribution
            # centered on the expert index space.
            expert_indices = torch.arange(
                num_experts, device=device, dtype=torch.float32
            )
            logits = -((expert_indices - mean) ** 2 / (2 * std**2))
            # Expand to [num_tokens, num_experts] and convert to
            # probabilities via softmax.
            probs = torch.softmax(logits, dim=-1).unsqueeze(0).expand(num_tokens, -1)
            return torch.multinomial(probs, num_samples=top_k, replacement=False).to(
                dtype=indices_type
            )

        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

    def _generate_weights(
        self, num_tokens: int, top_k: int, device: torch.device
    ) -> torch.Tensor:
        """Generate weights based on the distribution."""
        if self.distribution == "uniform":
            # All-ones weights for uniform distribution
            return torch.ones(
                (num_tokens, top_k),
                dtype=torch.float32,
                device=device,
            )

        elif self.distribution == "normal":
            mean = self.distribution_params["mean"]
            std = self.distribution_params["std"]
            raw = torch.normal(mean, std, size=(num_tokens, top_k), device=device)
            weights = torch.abs(raw)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            return weights

        else:
            raise ValueError(
                f"Unsupported distribution for weight generation: {self.distribution}"
            )

    def get_distribution_info(self) -> dict:
        """Get information about the current distribution configuration."""
        return {
            "distribution": self.distribution,
            "parameters": self.distribution_params.copy(),
        }


class RoutingSimulator:
    """
    Token-to-Expert Routing Simulator.

    This class provides a framework for testing and comparing different
    routing strategies for MoE models. It can simulate routing behavior
    and collect statistics for analysis.
    """

    # Class-level registry of routing strategies
    _routing_strategies: dict[str, RoutingStrategy] = {
        # Basic routing strategies
        "uniform_random": DistributionBasedRouting(
            distribution="uniform", mean=0.0, std=1.0
        ),
        "normal_routing": DistributionBasedRouting(
            distribution="normal", mean=0.0, std=1.0
        ),
    }

    @classmethod
    def register_strategy(cls, name: str, strategy: RoutingStrategy):
        """
        Register a custom routing strategy.

        Args:
            name: Name of the strategy
            strategy: RoutingStrategy instance
        """
        cls._routing_strategies[name] = strategy

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """
        Get list of available routing strategy names.

        Returns:
            List of available strategy names
        """
        return list(cls._routing_strategies.keys())

    @staticmethod
    def simulate_routing(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        strategy_name: str,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate token-to-expert routing using the specified strategy.

        Args:
            hidden_states: Input hidden states [num_tokens, hidden_size]
            router_logits: Router logits [num_tokens, num_experts]
            strategy_name: Name of the routing strategy to use
            top_k: Number of experts to select per token
            indices_type: Data type for expert indices

        Returns:
            tuple of (topk_weights, topk_ids)
        """
        if strategy_name not in RoutingSimulator._routing_strategies:
            raise ValueError(
                f"Unknown routing strategy: {strategy_name}. "
                f"Available strategies: "
                f"{list(RoutingSimulator._routing_strategies.keys())}"
            )
        logger.warning_once(
            "Simulating MoE routing using a %s strategy. "
            "This should only be used for performance testing. "
            "Model outputs will not be valid.",
            strategy_name,
        )

        strategy = RoutingSimulator._routing_strategies[strategy_name]
        return strategy.route_tokens(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            indices_type=indices_type,
        )


class RoutingSimulatorRouter(BaseRouter):
    """Router that uses routing simulation strategies for testing/debugging."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.Simulated

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use routing simulator to compute routing."""
        routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        topk_weights, topk_ids = RoutingSimulator.simulate_routing(
            hidden_states=hidden_states,
            router_logits=router_logits,
            strategy_name=routing_strategy,
            top_k=self.top_k,
            indices_type=indices_type,
        )
        return topk_weights, topk_ids
