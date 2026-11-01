# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.utils.torch_utils import PIN_MEMORY


@dataclass(frozen=True)
class ElasticWorkspacePlan:
    """Resolved capacity for a GPU-resident MoE expert workspace.

    ``expert_capacity`` is an assignment limit per local expert and source
    dispatcher.  The resident workspace therefore has ``num_dispatchers *
    expert_capacity`` rows per local expert.
    """

    expert_capacity: int
    max_num_tokens_per_rank: int
    num_dispatchers: int
    num_local_experts: int
    num_global_experts: int | None = None
    top_k: int = 1
    spill_capacity_rows: int = 8192
    spill_staging_rows: int = 256

    def __post_init__(self) -> None:
        if self.expert_capacity <= 0:
            raise ValueError("expert_capacity must be positive")
        if self.max_num_tokens_per_rank <= 0:
            raise ValueError("max_num_tokens_per_rank must be positive")
        if self.num_dispatchers <= 0:
            raise ValueError("num_dispatchers must be positive")
        if self.num_local_experts <= 0:
            raise ValueError("num_local_experts must be positive")
        if self.num_global_experts is not None and self.num_global_experts <= 0:
            raise ValueError("num_global_experts must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.spill_capacity_rows < 0:
            raise ValueError("spill_capacity_rows must be nonnegative")
        if self.spill_staging_rows < 0:
            raise ValueError("spill_staging_rows must be nonnegative")

    @property
    def resident_num_tokens_per_rank(self) -> int:
        return min(self.expert_capacity, self.max_num_tokens_per_rank)

    @property
    def resident_rows_per_expert(self) -> int:
        return self.num_dispatchers * self.resident_num_tokens_per_rank

    @property
    def full_rows_per_expert(self) -> int:
        return self.num_dispatchers * self.max_num_tokens_per_rank

    @property
    def resident_standard_rows(self) -> int:
        """Resident assignment rows for a compact Standard layout.

        DeepEP v2's compact Standard output has one row per routed assignment,
        not one padded row per local expert.  A source token therefore
        contributes ``top_k`` assignment rows, and the resident row budget is
        the source-token capacity multiplied by the number of dispatchers and
        top-k.  The per-expert capacity also bounds the total to
        ``num_local_experts * num_dispatchers * capacity``.
        """
        return self._standard_rows(self.resident_num_tokens_per_rank)

    @property
    def full_standard_rows(self) -> int:
        """Full-size assignment rows for a compact Standard layout."""
        return self._standard_rows(self.max_num_tokens_per_rank)

    def _standard_rows(self, num_tokens_per_rank: int) -> int:
        local_assignment_budget = (
            self.num_dispatchers * num_tokens_per_rank * self.top_k
        )
        num_global_experts = self.num_global_experts or self.num_local_experts
        local_fraction_budget = (
            local_assignment_budget * self.num_local_experts + num_global_experts - 1
        ) // num_global_experts
        per_expert_budget = (
            self.num_local_experts * self.num_dispatchers * num_tokens_per_rank
        )
        return min(local_assignment_budget, local_fraction_budget, per_expert_budget)

    @property
    def workspace_reduction_fraction(self) -> float:
        return 1.0 - self.resident_rows_per_expert / self.full_rows_per_expert

    def resident_shape(self, width: int) -> tuple[int, int, int]:
        return (self.num_local_experts, self.resident_rows_per_expert, width)

    def full_shape(self, resident_shape: tuple[int, ...]) -> tuple[int, ...] | None:
        """Expand the dispatched row extent to its full-size equivalent."""
        if self.resident_rows_per_expert == self.full_rows_per_expert:
            return resident_shape

        row_axis: int | None = None
        if (
            len(resident_shape) >= 2
            and resident_shape[0] == self.num_local_experts
            and resident_shape[1] == self.resident_rows_per_expert
        ):
            row_axis = 1
        if row_axis is None and resident_shape:
            if resident_shape[0] == self.resident_rows_per_expert:
                row_axis = 0
            else:
                row_axis = next(
                    (
                        axis
                        for axis, size in enumerate(resident_shape)
                        if size == self.resident_rows_per_expert
                    ),
                    None,
                )
        if row_axis is None:
            return None

        full_shape = list(resident_shape)
        full_shape[row_axis] = self.full_rows_per_expert
        return tuple(full_shape)

    def standard_shape_fits(self, resident_shape: tuple[int, ...]) -> bool:
        """Whether compact Standard rows fit the GPU-resident capacity."""
        return bool(resident_shape) and resident_shape[0] <= self.resident_standard_rows

    def full_standard_shape(
        self, resident_shape: tuple[int, ...]
    ) -> tuple[int, ...] | None:
        """Expand a compact Standard row extent to its full-size equivalent."""
        if not resident_shape:
            return None
        full_shape = list(resident_shape)
        full_shape[0] = self.full_standard_rows
        return tuple(full_shape)


@dataclass
class ElasticWorkspace:
    """Fixed-capacity expert workspace with a cold host-backed spill tail.

    The GPU-resident prefix is ordinary expert-major storage and is allocated by
    the regular workspace manager.  This object owns the dispatcher-independent
    sidecars used by a future overflow path.  No dispatcher currently uses the
    sidecars; they are allocated up front so capture and locking see a fixed
    footprint.
    """

    plan: ElasticWorkspacePlan
    spill_host: torch.Tensor | None = None
    spill_staging: torch.Tensor | None = None
    routing: torch.Tensor | None = None

    @classmethod
    def allocate_sidecars(
        cls,
        plan: ElasticWorkspacePlan,
        *,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "ElasticWorkspace":
        if plan.spill_capacity_rows == 0 or plan.spill_staging_rows == 0:
            return cls(plan)

        from vllm.v1.worker.workspace import current_workspace_manager

        staging_rows = min(plan.spill_staging_rows, plan.spill_capacity_rows)
        key = (
            "elastic_moe_spill",
            hidden_size,
            dtype,
            device,
            plan.spill_capacity_rows,
            staging_rows,
        )

        def allocate() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            spill_host = torch.empty(
                (plan.spill_capacity_rows, hidden_size),
                dtype=dtype,
                pin_memory=PIN_MEMORY,
            )
            spill_staging = torch.empty(
                (staging_rows, hidden_size), dtype=dtype, device=device
            )
            routing = torch.empty(
                (plan.spill_capacity_rows, 2), dtype=torch.int32, device=device
            )
            return spill_host, spill_staging, routing

        spill_host, spill_staging, routing = (
            current_workspace_manager().get_persistent_resource(key, allocate)
        )
        return cls(
            plan,
            spill_host=spill_host,
            spill_staging=spill_staging,
            routing=routing,
        )
