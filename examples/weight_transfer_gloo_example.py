#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example: GLOO-based weight transfer for CPU training.

This example demonstrates how to use the GLOOWeightTransferEngine to transfer
model weights from a trainer process to vLLM inference workers over CPU using
the GLOO backend.

GLOO is ideal for:
- CPU-only environments where NCCL is not available
- Development and testing scenarios
- Multi-process training on systems without GPU support

For production GPU scenarios, use the NCCL backend instead.
"""

import argparse
import multiprocessing as mp

import torch
import torch.distributed as dist

from vllm.distributed.weight_transfer.gloo_engine import (
    GLOOTrainerSendWeightsArgs,
    GLOOWeightTransferEngine,
    GLOOWeightTransferInitInfo,
)


def trainer_process(master_address: str, master_port: int, world_size: int):
    """Trainer process that broadcasts weights using GLOO."""
    print("[Trainer] Starting trainer process...")

    # Initialize GLOO process group for trainer (rank 0)
    init_info = GLOOWeightTransferInitInfo(
        master_address=master_address,
        master_port=master_port,
        rank_offset=0,
        world_size=world_size,
    )
    group = GLOOWeightTransferEngine.trainer_init(init_info)
    print("[Trainer] Initialized GLOO process group")

    # Create dummy model weights on CPU
    weights = {
        "layer1.weight": torch.randn(100, 100, dtype=torch.float32),
        "layer1.bias": torch.randn(100, dtype=torch.float32),
        "layer2.weight": torch.randn(50, 100, dtype=torch.float32),
        "layer2.bias": torch.randn(50, dtype=torch.float32),
    }

    print(f"[Trainer] Created {len(weights)} weight tensors")
    print(f"[Trainer] Total params: {sum(w.numel() for w in weights.values()):,}")

    # Create iterator over weights
    def param_iterator():
        for name, tensor in weights.items():
            print(f"[Trainer] Sending {name}: {list(tensor.shape)}")
            yield (name, tensor.cpu())  # Ensure on CPU

    # Send weights to all workers using GLOO
    trainer_args = GLOOTrainerSendWeightsArgs(
        group=group,
        src=0,  # Trainer is rank 0
    )

    print("[Trainer] Broadcasting weights to workers...")
    GLOOWeightTransferEngine.trainer_send_weights(
        param_iterator(),
        trainer_args,
    )

    print("[Trainer] ✓ All weights broadcast successfully")

    # Cleanup
    dist.destroy_process_group(group)


def worker_process(rank: int, master_address: str, master_port: int, world_size: int):
    """Worker process that receives weights using GLOOWeightTransferEngine."""
    print(f"[Worker {rank}] Starting worker process...")

    # Mock parallel config for worker
    from unittest.mock import MagicMock

    from vllm.config.parallel import ParallelConfig
    from vllm.config.weight_transfer import WeightTransferConfig

    config = WeightTransferConfig(backend="gloo")
    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = 0
    parallel_config.world_size = 1
    parallel_config.data_parallel_index = 0

    # Create GLOO engine
    engine = GLOOWeightTransferEngine(config, parallel_config)

    # Initialize with trainer
    init_info = GLOOWeightTransferInitInfo(
        master_address=master_address,
        master_port=master_port,
        rank_offset=rank,  # Worker ranks start after trainer (rank 0)
        world_size=world_size,
    )
    engine.init_transfer_engine(init_info)
    print(f"[Worker {rank}] Initialized GLOO engine")

    # Prepare to receive weights
    received_weights = {}

    def load_weights(weights_batch):
        """Callback to load received weights."""
        for name, tensor in weights_batch:
            received_weights[name] = tensor.clone()
            print(f"[Worker {rank}] Received {name}: {list(tensor.shape)}")

    # Receive weights from trainer
    from vllm.distributed.weight_transfer.gloo_engine import (
        GLOOWeightTransferUpdateInfo,
    )

    update_info = GLOOWeightTransferUpdateInfo(
        names=["layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias"],
        dtype_names=["float32", "float32", "float32", "float32"],
        shapes=[[100, 100], [100], [50, 100], [50]],
    )

    print(f"[Worker {rank}] Waiting for weights...")
    engine.receive_weights(update_info, load_weights)

    # Verify
    print(f"[Worker {rank}] ✓ Received {len(received_weights)} weight tensors")
    total_params = sum(w.numel() for w in received_weights.values())
    print(f"[Worker {rank}] Total params received: {total_params:,}")

    # Cleanup
    engine.shutdown()


def main():
    """Run GLOO weight transfer example."""
    parser = argparse.ArgumentParser(
        description="GLOO weight transfer example (CPU-based)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes (default: 2)",
    )
    parser.add_argument(
        "--master-address",
        type=str,
        default="127.0.0.1",
        help="Master address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Master port (default: 29500)",
    )
    args = parser.parse_args()

    world_size = 1 + args.num_workers  # trainer + workers

    print("=" * 70)
    print("GLOO Weight Transfer Example (CPU)")
    print("=" * 70)
    print(f"Master: {args.master_address}:{args.master_port}")
    print(f"Workers: {args.num_workers}")
    print(f"World size: {world_size}")
    print("=" * 70)

    # Start trainer and workers
    processes = []

    # Start trainer (rank 0)
    trainer = mp.Process(
        target=trainer_process,
        args=(args.master_address, args.master_port, world_size),
    )
    trainer.start()
    processes.append(trainer)

    # Start workers (ranks 1, 2, ...)
    for worker_idx in range(args.num_workers):
        worker = mp.Process(
            target=worker_process,
            args=(
                1 + worker_idx,  # Worker ranks start at 1
                args.master_address,
                args.master_port,
                world_size,
            ),
        )
        worker.start()
        processes.append(worker)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("=" * 70)
    print("✅ Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
