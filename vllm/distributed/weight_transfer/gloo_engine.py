# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLOO-based weight transfer engine for CPU."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)


@dataclass
class GLOOWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for GLOO weight transfer backend."""

    master_address: str
    master_port: int
    rank_offset: int
    world_size: int


@dataclass
class GLOOTrainerSendWeightsArgs:
    """Arguments for GLOO trainer_send_weights method."""

    group: dist.ProcessGroup
    """Process group for GLOO communication."""
    src: int = 0
    """Source rank (default 0, trainer is typically rank 0)."""
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor] | None = None
    """Optional function to apply to each (name, tensor) pair before broadcasting.
    If None, extracts just the tensor."""


@dataclass
class GLOOWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for GLOO weight transfer backend."""

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]

    def __post_init__(self):
        """Validate that all lists have the same length and warn about unsupported options."""
        import warnings

        num_params = len(self.names)
        if len(self.dtype_names) != num_params:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`: "
                f"got {len(self.dtype_names)} and {len(self.names)}"
            )
        if len(self.shapes) != num_params:
            raise ValueError(
                f"`shapes` should be of the same size as `names`: "
                f"got {len(self.shapes)} and {len(self.names)}"
            )

        # Warn about unsupported options
        if not self.layerwise_cpu_buffer:
            warnings.warn(
                "GLOO weight transfer always uses CPU buffers. "
                "Setting layerwise_cpu_buffer=False has no effect and will be ignored.",
                UserWarning,
                stacklevel=2,
            )


class GLOOWeightTransferEngine(
    WeightTransferEngine[GLOOWeightTransferInitInfo, GLOOWeightTransferUpdateInfo]
):
    """
    Weight transfer engine using GLOO for CPU-based communication between trainer and
    workers.

    This implementation uses GLOO broadcast operations to transfer weights from
    the trainer (rank 0) to all inference workers in a process group over CPU.
    """

    # Define backend-specific dataclass types
    init_info_cls = GLOOWeightTransferInitInfo
    update_info_cls = GLOOWeightTransferUpdateInfo

    def __init__(
        self, config: WeightTransferConfig, parallel_config: ParallelConfig
    ) -> None:
        """
        Initialize the GLOO weight transfer engine.

        Args:
            config: The configuration for the weight transfer engine
            parallel_config: The configuration for the parallel setup
        """
        super().__init__(config, parallel_config)
        self.model_update_group: dist.ProcessGroup | None = None

    def parse_update_info(
        self, update_dict: dict[str, Any]
    ) -> GLOOWeightTransferUpdateInfo:
        """
        Construct typed update info from dict with validation.

        Warns if packed tensor options are specified (not supported by GLOO).

        Args:
            update_dict: Dictionary containing backend-specific update parameters

        Returns:
            Typed backend-specific update info dataclass

        Raises:
            ValueError: If update_dict is invalid for this backend
        """
        import warnings

        # Warn if packed tensor options are specified
        if update_dict.get("packed", False):
            warnings.warn(
                "GLOO weight transfer does not support packed tensor broadcasting. "
                "The 'packed' option will be ignored. Weights will be transferred "
                "one-by-one. For packed transfer, use the NCCL backend instead.",
                UserWarning,
                stacklevel=2,
            )
            # Remove from dict to avoid TypeError
            update_dict = {k: v for k, v in update_dict.items() if k != "packed"}

        if "packed_buffer_size_bytes" in update_dict:
            warnings.warn(
                "GLOO weight transfer does not support packed tensor broadcasting. "
                "The 'packed_buffer_size_bytes' option will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            update_dict = {
                k: v for k, v in update_dict.items() if k != "packed_buffer_size_bytes"
            }

        if "packed_num_buffers" in update_dict:
            warnings.warn(
                "GLOO weight transfer does not support packed tensor broadcasting. "
                "The 'packed_num_buffers' option will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            update_dict = {
                k: v for k, v in update_dict.items() if k != "packed_num_buffers"
            }

        # Call parent implementation
        return super().parse_update_info(update_dict)

    def init_transfer_engine(self, init_info: GLOOWeightTransferInitInfo) -> None:
        """
        Initialize GLOO process group with the trainer.

        Args:
            init_info: GLOO initialization info containing master address, port,
                      rank offset, and world size
        """

        # Calculate the global rank in the trainer-worker process group
        # Must account for data parallel to get unique ranks across all workers
        dp_rank = self.parallel_config.data_parallel_index
        world_size_per_dp = self.parallel_config.world_size  # TP * PP
        rank_within_dp = self.parallel_config.rank

        # Unique rank across all DP groups
        worker_rank = dp_rank * world_size_per_dp + rank_within_dp
        rank = worker_rank + init_info.rank_offset

        # Create process group using torch.distributed with gloo backend
        self.model_update_group = GLOOWeightTransferEngine._init_process_group(
            init_info.master_address,
            init_info.master_port,
            rank,
            init_info.world_size,
        )

    def receive_weights(
        self,
        update_info: GLOOWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """
        Receive weights from trainer via GLOO broadcast and load them incrementally.

        Args:
            update_info: GLOO update info containing parameter names, dtypes, shapes
            load_weights: Callable that loads weights into the model. Called
                         incrementally for each weight to avoid OOM.
        """
        if self.model_update_group is None:
            raise RuntimeError(
                "GLOO weight transfer not initialized. "
                "Call init_transfer_engine() first."
            )

        # Use simple one-by-one broadcasting on CPU
        for name, dtype_name, shape in zip(
            update_info.names, update_info.dtype_names, update_info.shapes
        ):
            dtype = getattr(torch, dtype_name)
            weight = torch.empty(shape, dtype=dtype, device="cpu")
            dist.broadcast(weight, src=0, group=self.model_update_group)
            load_weights([(name, weight)])
            del weight

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            # Destroy the process group
            dist.destroy_process_group(self.model_update_group)
            self.model_update_group = None

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | GLOOTrainerSendWeightsArgs,
    ) -> None:
        """Broadcast weights from trainer to vLLM workers using GLOO.

        Args:
            iterator: Iterator of model parameters. Returns (name, tensor) tuples.
                     Tensors should be on CPU.
            trainer_args: Dictionary or GLOOTrainerSendWeightsArgs instance containing
                         GLOO-specific arguments. If a dict, should contain keys from
                         GLOOTrainerSendWeightsArgs.

        Example:
            >>> from vllm.distributed.weight_transfer.gloo_engine import (
            ...     GLOOWeightTransferEngine,
            ...     GLOOTrainerSendWeightsArgs,
            ... )
            >>> param_iter = ((n, p.cpu()) for n, p in model.named_parameters())
            >>> args = GLOOTrainerSendWeightsArgs(group=group)
            >>> GLOOWeightTransferEngine.trainer_send_weights(param_iter, args)
        """
        # Parse trainer args - accept either dict or dataclass instance
        if isinstance(trainer_args, dict):
            args = GLOOTrainerSendWeightsArgs(**trainer_args)
        else:
            args = trainer_args

        if args.post_iter_func is None:
            # Default: extract just the tensor from (name, tensor) tuple
            post_iter_func = lambda x: x[1]
        else:
            post_iter_func = args.post_iter_func

        # Use simple one-by-one broadcasting
        for item in iterator:
            tensor = post_iter_func(item)
            # Ensure tensor is on CPU
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            dist.broadcast(tensor, src=args.src, group=args.group)

    @staticmethod
    def trainer_init(
        init_info: GLOOWeightTransferInitInfo | dict,
    ) -> dist.ProcessGroup:
        """
        Initialize GLOO process group for trainer-side weight transfer.

        The trainer is always rank 0 in the process group.

        Args:
            init_info: Either a GLOOWeightTransferInitInfo object or a dict with keys:
                - master_address: str
                - master_port: int
                - world_size: int

        Returns:
            ProcessGroup for weight transfer.

        Example:
            >>> from vllm.distributed.weight_transfer.gloo_engine import (
            ...     GLOOWeightTransferEngine,
            ... )
            >>> group = GLOOWeightTransferEngine.trainer_init(
            ...     dict(
            ...         master_address=master_address,
            ...         master_port=master_port,
            ...         world_size=world_size,
            ...     ),
            ... )
        """
        if isinstance(init_info, dict):
            master_address = init_info["master_address"]
            master_port = init_info["master_port"]
            world_size = init_info["world_size"]
        else:
            # GLOOWeightTransferInitInfo object
            master_address = init_info.master_address
            master_port = init_info.master_port
            world_size = init_info.world_size

        # Trainer is always rank 0
        return GLOOWeightTransferEngine._init_process_group(
            master_address,
            master_port,
            0,
            world_size,
        )

    @staticmethod
    def _init_process_group(
        master_address: str, master_port: int, rank: int, world_size: int
    ) -> dist.ProcessGroup:
        """
        Initialize a GLOO process group for CPU-based communication.

        This creates an independent process group using a dedicated TCPStore,
        similar to how NCCL uses StatelessProcessGroup. The group does not
        interfere with any existing torch.distributed groups.

        Args:
            master_address: Master node address
            master_port: Master node port
            rank: Rank of this process in the weight transfer group
            world_size: Total number of processes in the weight transfer group

        Returns:
            ProcessGroup for GLOO communication
        """
        from datetime import timedelta

        from torch.distributed import PrefixStore, ProcessGroupGloo, TCPStore

        # Create a TCPStore for this specific weight transfer group
        # This is independent of any existing torch.distributed state
        is_master = rank == 0
        store = TCPStore(
            host_name=master_address,
            port=master_port,
            world_size=world_size,
            is_master=is_master,
            timeout=timedelta(seconds=300),
        )

        # Use a prefix to isolate this group's data in the store
        prefix = f"gloo_weight_transfer_{master_port}"
        prefix_store = PrefixStore(prefix, store)

        # Create ProcessGroupGloo directly with the store
        # This bypasses the global torch.distributed state
        pg = ProcessGroupGloo(
            prefix_store,
            rank,
            world_size,
            timeout=timedelta(seconds=300),
        )

        return pg
