# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import os

import numpy as np
import pytest
import torch
import torch.distributed

from vllm.distributed.communication_op import (  # noqa
    tensor_model_parallel_all_reduce)
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             get_world_group, graph_capture,
                                             init_distributed_environment)
from vllm.utils import update_environment_variables


def distributed_run(fn, world_size):
    number_of_processes = world_size
    processes: list[multiprocessing.Process] = []
    for i in range(number_of_processes):
        env: dict[str, str] = {}
        env['RANK'] = str(i)
        env['LOCAL_RANK'] = str(i)
        env['WORLD_SIZE'] = str(number_of_processes)
        env['LOCAL_WORLD_SIZE'] = str(number_of_processes)
        env['MASTER_ADDR'] = 'localhost'
        env['MASTER_PORT'] = '12345'
        p = multiprocessing.Process(target=fn, args=(env, ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def worker_fn_wrapper(fn):
    # `multiprocessing.Process` cannot accept environment variables directly
    # so we need to pass the environment variables as arguments
    # and update the environment variables in the function
    def wrapped_fn(env):
        update_environment_variables(env)
        local_rank = os.environ['LOCAL_RANK']
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        init_distributed_environment()
        fn()

    return wrapped_fn


@worker_fn_wrapper
def worker_fn():
    pynccl_comm = PyNcclCommunicator(get_world_group().cpu_group,
                                     device=get_world_group().device)
    tensor = torch.ones(16, 1024, 1024,
                        dtype=torch.float32).cuda(pynccl_comm.rank)
    tensor = pynccl_comm.all_reduce(tensor)
    torch.cuda.synchronize()
    assert torch.all(tensor == pynccl_comm.world_size).cpu().item()


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl():
    distributed_run(worker_fn, 2)


@worker_fn_wrapper
def multiple_allreduce_worker_fn():
    device = torch.device(f"cuda:{torch.distributed.get_rank()}")
    groups = [
        torch.distributed.new_group(ranks=[0, 1], backend="gloo"),
        torch.distributed.new_group(ranks=[2, 3], backend="gloo")
    ]
    group = groups[0] if torch.distributed.get_rank() in [0, 1] else groups[1]
    pynccl_comm = PyNcclCommunicator(group=group, device=device)
    tensor = torch.ones(16, 1024, 1024, dtype=torch.float32, device=device)
    # two groups can communicate independently
    if torch.distributed.get_rank() in [0, 1]:
        tensor = pynccl_comm.all_reduce(tensor)
        tensor = pynccl_comm.all_reduce(tensor)
        torch.cuda.synchronize()
        assert torch.all(tensor == 4).cpu().item()
    else:
        tensor = pynccl_comm.all_reduce(tensor)
        torch.cuda.synchronize()
        assert torch.all(tensor == 2).cpu().item()


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs to run the test.")
def test_pynccl_multiple_allreduce():
    # this tests pynccl for multiple tp groups, in a standalone way
    # i.e. call `pynccl_comm.all_reduce` directly
    distributed_run(multiple_allreduce_worker_fn, 4)


@worker_fn_wrapper
def multiple_allreduce_with_vllm_worker_fn():
    device = torch.device(f"cuda:{torch.distributed.get_rank()}")
    ensure_model_parallel_initialized(2, 2)
    tensor = torch.ones(16, 1024, 1024, dtype=torch.float32, device=device)
    with graph_capture(device=device):
        # two tp groups can communicate independently
        if torch.distributed.get_rank() in [0, 1]:
            tensor = tensor_model_parallel_all_reduce(tensor)
            tensor = tensor_model_parallel_all_reduce(tensor)
            torch.cuda.synchronize()
            assert torch.all(tensor == 4).cpu().item()
        else:
            tensor = tensor_model_parallel_all_reduce(tensor)
            torch.cuda.synchronize()
            assert torch.all(tensor == 2).cpu().item()


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs to run the test.")
def test_pynccl_multiple_allreduce_with_vllm():
    # this tests pynccl for multiple tp groups, together with vllm
    # i.e. call `tensor_model_parallel_all_reduce`
    distributed_run(multiple_allreduce_with_vllm_worker_fn, 4)


@worker_fn_wrapper
def worker_fn_with_cudagraph():
    with torch.no_grad():
        graph = torch.cuda.CUDAGraph()
        pynccl_comm = PyNcclCommunicator(get_world_group().cpu_group,
                                         device=get_world_group().device)
        # run something in the default stream to initialize torch engine
        a = torch.ones((4, 4), device=f'cuda:{pynccl_comm.rank}')
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            a_out = pynccl_comm.all_reduce(a)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.all(a_out == pynccl_comm.world_size).cpu().item()


@worker_fn_wrapper
def all_gather_worker_fn():
    pynccl_comm = PyNcclCommunicator(get_world_group().cpu_group,
                                     device=get_world_group().device)

    rank = pynccl_comm.rank
    world_size = pynccl_comm.world_size
    device = f'cuda:{pynccl_comm.rank}'

    num_elems = 1000
    tensor = torch.arange(num_elems, dtype=torch.float32,
                          device=device) + rank * num_elems
    result = torch.zeros(num_elems * world_size,
                         dtype=torch.float32,
                         device=device)

    expected = torch.cat([
        torch.arange(num_elems, dtype=torch.float32) + r * num_elems
        for r in range(world_size)
    ]).to(device)

    pynccl_comm.all_gather(result, tensor)
    torch.cuda.synchronize()
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl_all_gather():
    distributed_run(all_gather_worker_fn, 2)


@worker_fn_wrapper
def all_gatherv_worker_fn():
    pynccl_comm = PyNcclCommunicator(get_world_group().cpu_group,
                                     device=get_world_group().device)

    rank = pynccl_comm.rank
    world_size = pynccl_comm.world_size
    device = f'cuda:{pynccl_comm.rank}'

    assert world_size <= 8
    sizes = [81, 20, 57, 52, 81, 5, 49, 49][:world_size]
    num_elems = sizes[rank]
    tensor = torch.arange(num_elems, dtype=torch.float32,
                          device=device) + rank * 100
    result = torch.zeros(sum(sizes), dtype=torch.float32, device=device)

    expected = torch.cat([
        torch.arange(sizes[r], dtype=torch.float32) + r * 100
        for r in range(world_size)
    ]).to(device)

    pynccl_comm.all_gatherv(result, tensor, sizes=sizes)
    torch.cuda.synchronize()
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl_all_gatherv():
    distributed_run(all_gatherv_worker_fn, 2)


@worker_fn_wrapper
def reduce_scatter_worker_fn():
    pynccl_comm = PyNcclCommunicator(get_world_group().cpu_group,
                                     device=get_world_group().device)

    rank = pynccl_comm.rank
    world_size = pynccl_comm.world_size
    device = f'cuda:{pynccl_comm.rank}'

    num_elems = 1000
    tensor = torch.arange(num_elems, dtype=torch.float32,
                          device=device) + rank * num_elems
    assert (num_elems % world_size == 0)
    result = torch.zeros(num_elems // world_size,
                         dtype=torch.float32,
                         device=device)

    # Calculate expected result for this rank's chunk
    scattered_size = num_elems // world_size
    all_tensors = [
        torch.arange(num_elems, dtype=torch.float32) + r * num_elems
        for r in range(world_size)
    ]
    expected = sum(tensor[rank * scattered_size:(rank + 1) * scattered_size]
                   for tensor in all_tensors).to(device)

    pynccl_comm.reduce_scatter(result, tensor)
    torch.cuda.synchronize()
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl_reduce_scatter():
    distributed_run(reduce_scatter_worker_fn, 2)


@worker_fn_wrapper
def reduce_scatterv_worker_fn():
    pynccl_comm = PyNcclCommunicator(get_world_group().cpu_group,
                                     device=get_world_group().device)

    rank = pynccl_comm.rank
    world_size = pynccl_comm.world_size
    device = f'cuda:{pynccl_comm.rank}'

    assert world_size <= 8
    sizes = [81, 20, 57, 52, 81, 5, 49, 49][:world_size]
    num_elems = sum(sizes)
    tensor = torch.arange(num_elems, dtype=torch.float32,
                          device=device) + rank * 100
    result = torch.zeros(sizes[rank], dtype=torch.float32, device=device)

    # Calculate expected result for this rank's chunk
    all_tensors = [
        torch.arange(num_elems, dtype=torch.float32) + r * 100
        for r in range(world_size)
    ]
    sizes_cumsum = np.cumsum(sizes)
    start = 0 if rank == 0 else sizes_cumsum[rank - 1]
    end = sizes_cumsum[rank]
    expected = sum(tensor[start:end] for tensor in all_tensors).to(device)

    pynccl_comm.reduce_scatterv(result, tensor, sizes=sizes)
    torch.cuda.synchronize()
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl_reduce_scatterv():
    distributed_run(reduce_scatterv_worker_fn, 2)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl_with_cudagraph():
    distributed_run(worker_fn_with_cudagraph, 2)


@worker_fn_wrapper
def send_recv_worker_fn():
    pynccl_comm = PyNcclCommunicator(get_world_group().cpu_group,
                                     device=get_world_group().device)
    if pynccl_comm.rank == 0:
        tensor = torch.ones(16, 1024, 1024,
                            dtype=torch.float32).cuda(pynccl_comm.rank)
    else:
        tensor = torch.empty(16, 1024, 1024,
                             dtype=torch.float32).cuda(pynccl_comm.rank)

    if pynccl_comm.rank == 0:
        pynccl_comm.send(tensor,
                         dst=(pynccl_comm.rank + 1) % pynccl_comm.world_size)
    else:
        pynccl_comm.recv(tensor,
                         src=(pynccl_comm.rank - 1) % pynccl_comm.world_size)
    torch.cuda.synchronize()
    assert torch.all(tensor == 1).cpu().item()


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl_send_recv():
    distributed_run(send_recv_worker_fn, 2)


@worker_fn_wrapper
def multiple_send_recv_worker_fn():
    device = torch.device(f"cuda:{torch.distributed.get_rank()}")
    groups = [
        torch.distributed.new_group(ranks=[0, 2], backend="gloo"),
        torch.distributed.new_group(ranks=[1, 3], backend="gloo")
    ]
    group = groups[0] if torch.distributed.get_rank() in [0, 2] else groups[1]
    pynccl_comm = PyNcclCommunicator(group=group, device=device)
    if torch.distributed.get_rank() == 0:
        tensor = torch.ones(16, 1024, 1024, dtype=torch.float32, device=device)
    elif torch.distributed.get_rank() == 1:
        tensor = 2 * torch.ones(
            16, 1024, 1024, dtype=torch.float32, device=device)
    else:
        tensor = torch.empty(16,
                             1024,
                             1024,
                             dtype=torch.float32,
                             device=device)
    if torch.distributed.get_rank() in [0, 1]:
        pynccl_comm.send(tensor,
                         dst=(pynccl_comm.rank + 1) % pynccl_comm.world_size)
    else:
        pynccl_comm.recv(tensor,
                         src=(pynccl_comm.rank - 1) % pynccl_comm.world_size)
    torch.cuda.synchronize()
    if torch.distributed.get_rank() in [0, 2]:
        assert torch.all(tensor == 1).cpu().item()
    else:
        assert torch.all(tensor == 2).cpu().item()


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs to run the test.")
def test_pynccl_multiple_send_recv():
    distributed_run(multiple_send_recv_worker_fn, 4)


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs to run the test.")
def test_pynccl_broadcast():
    distributed_run(broadcast_worker_fn, 4)


@worker_fn_wrapper
def broadcast_worker_fn():
    # Test broadcast for every root rank.
    # Essentially this is an all-gather operation.
    pynccl_comm = PyNcclCommunicator(get_world_group().cpu_group,
                                     device=get_world_group().device)
    recv_tensors = [
        torch.empty(16,
                    1024,
                    1024,
                    dtype=torch.float32,
                    device=pynccl_comm.device)
        for i in range(pynccl_comm.world_size)
    ]
    recv_tensors[pynccl_comm.rank] = torch.ones(
        16, 1024, 1024, dtype=torch.float32,
        device=pynccl_comm.device) * pynccl_comm.rank

    for i in range(pynccl_comm.world_size):
        pynccl_comm.broadcast(recv_tensors[i], src=i)
        # the broadcast op might be launched in a different stream
        # need to synchronize to make sure the tensor is ready
        torch.cuda.synchronize()
        assert torch.all(recv_tensors[i] == i).cpu().item()


def test_ncclGetUniqueId():
    lib = NCCLLibrary()
    unique_id = lib.ncclGetUniqueId()
    # `list(unique_id.internal)` is something like this:
    # [34, -16, 23, 83, 109, -19, 59, 95, 2, 0, -86, 55, 10, -128, 0, 29, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # as long as the function doesn't raise an exception, we're good
    assert unique_id is not None
