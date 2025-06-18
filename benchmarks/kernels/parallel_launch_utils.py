# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import dataclasses
import os
import traceback
from typing import Callable

import torch
from torch.multiprocessing import spawn  # pyright: ignore[reportPrivateImportUsage]
from typing_extensions import Concatenate, ParamSpec

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel

## Parallel Launch Utils ####

P = ParamSpec("P")


@dataclasses.dataclass
class ProcessGroupInfo:
    world_size: int
    world_local_size: int
    rank: int
    node_rank: int
    local_rank: int
    device: torch.device


def _worker_parallel_launch(
    local_rank: int,
    world_size: int,
    world_local_size: int,
    node_rank: int,
    init_method: str,
    worker: Callable[Concatenate[ProcessGroupInfo, P], None],
    vllm_config: VllmConfig,
    env_dict: dict,
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    rank = node_rank * world_local_size + local_rank

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    barrier = torch.tensor([rank], device=device)
    torch.distributed.all_reduce(barrier)

    set_current_vllm_config(vllm_config)

    # setup environment
    os.environ.update(env_dict)

    import tempfile

    temp_file = tempfile.mkstemp()[1]
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"file://{temp_file}",
            local_rank=local_rank,
            backend="nccl",
        )

        initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

        cpu_group = torch.distributed.new_group(list(range(world_size)), backend="gloo")

    try:
        worker(
            ProcessGroupInfo(
                world_size=world_size,
                world_local_size=world_local_size,
                rank=rank,
                node_rank=node_rank,
                local_rank=local_rank,
                device=device,
            ),
            vllm_config,
            cpu_group,
            *args,
            **kwargs,
        )
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        raise
    finally:
        torch.distributed.destroy_process_group()


def parallel_launch(
    world_size: int,
    worker: Callable[Concatenate[ProcessGroupInfo, P], None],
    vllm_config: VllmConfig,
    env_dict: dict[any, any],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    assert not kwargs
    spawn(
        _worker_parallel_launch,
        args=(
            world_size,
            world_size,
            0,
            "tcp://localhost:29500",
            worker,
            vllm_config,
            env_dict,
        )
        + args,
        nprocs=world_size,
        join=True,
    )
