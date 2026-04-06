# GLOO Engine

The GLOO weight transfer engine uses [torch.distributed with GLOO backend](https://pytorch.org/docs/stable/distributed.html#backends) broadcast operations to transfer weights from the trainer to inference workers over CPU. It is designed for **CPU-only** environments where NCCL is not available.

## When to Use GLOO

- Training and inference on **CPU-only** systems without GPU support
- **Development and testing** scenarios where GPU hardware is not required
- **Multi-process** CPU-based training with distributed inference workers
- Environments where NCCL is not available or supported

!!! warning
    For production GPU-based workloads, use the **NCCL** backend instead. GLOO is significantly slower than NCCL for GPU-to-GPU transfers and is primarily intended for CPU-only scenarios.

## How It Works

1. The trainer and all inference workers join a shared GLOO process group using `torch.distributed.init_process_group` with the `gloo` backend.
2. The trainer broadcasts weights to all workers simultaneously over CPU. Each worker receives and loads weights incrementally.
3. Unlike NCCL, GLOO does not support packed tensor broadcasting, so weights are transferred one-by-one.

## Initialization

GLOO requires explicit process group setup. The trainer and inference workers must agree on a master address, port, and world size.

### Inference Side

```python
from vllm.distributed.weight_transfer.base import WeightTransferInitRequest

# rank_offset accounts for the trainer occupying rank 0
llm.init_weight_transfer_engine(
    WeightTransferInitRequest(
        init_info=dict(
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,
            world_size=world_size,  # trainer + all inference workers
        )
    )
)
```

### Trainer Side

```python
from vllm.distributed.weight_transfer.gloo_engine import (
    GLOOWeightTransferEngine,
)

group = GLOOWeightTransferEngine.trainer_init(
    dict(
        master_address=master_address,
        master_port=master_port,
        world_size=world_size,
    )
)
```

!!! note
    `trainer_init` always assigns the trainer to rank 0. Inference workers start at `rank_offset` (typically 1).

!!! important
    GLOO weight transfer requires `torch.distributed` to be uninitialized before calling `trainer_init` or `init_transfer_engine`. If your application uses `torch.distributed` for other purposes, consider using the **IPC** backend instead.

## Sending Weights

```python
from vllm.distributed.weight_transfer.gloo_engine import (
    GLOOTrainerSendWeightsArgs,
    GLOOWeightTransferEngine,
)

# Ensure weights are on CPU
def param_iterator():
    for name, param in model.named_parameters():
        yield (name, param.cpu())

trainer_args = GLOOTrainerSendWeightsArgs(
    group=group,
    src=0,  # Trainer is rank 0
)

GLOOWeightTransferEngine.trainer_send_weights(
    iterator=param_iterator(),
    trainer_args=trainer_args,
)
```

!!! important
    All tensors must be on **CPU** before broadcasting with GLOO. If your model is on GPU, make sure to call `.cpu()` on each tensor before yielding it to the iterator.

See [`GLOOTrainerSendWeightsArgs`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/weight_transfer/gloo_engine.py) for the full list of configurable fields.

## Receiving Weights (Inference Side)

The inference side triggers weight reception by calling `update_weights`:

```python
from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest

llm.update_weights(
    WeightTransferUpdateRequest(
        update_info=dict(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
        )
    )
)
```

The `names`, `dtype_names`, and `shapes` lists describe each parameter. These must match the order in which the trainer iterates over its parameters.

## Examples

See [`examples/weight_transfer_gloo_example.py`](https://github.com/vllm-project/vllm/blob/main/examples/weight_transfer_gloo_example.py) for a complete working example demonstrating GLOO-based weight transfer between a trainer and multiple worker processes.

## Limitations

- **No packed tensor broadcasting**: Weights are transferred one-by-one, which can be slower for models with many small tensors.
- **CPU-bound**: GLOO is optimized for CPU communication and will be significantly slower than NCCL for GPU workloads.
- **Requires uninitialized torch.distributed**: Cannot be used if `torch.distributed.init_process_group()` has already been called.
- **No CUDA stream overlap**: Unlike NCCL, GLOO does not support overlapping communication with computation on GPU streams.

For production GPU-based workloads, prefer the **NCCL** backend for optimal performance.
