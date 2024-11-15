import os
from functools import partial
from typing import Any, List, Optional, Tuple, Iterator
from contextlib import contextmanager
import queue
import threading

import torch
import msgspec
import zmq 

from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger
from vllm.triton_utils import maybe_set_triton_cache_manager
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        get_vllm_instance_id, get_open_zmq_ipc_path)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_worker import WorkerProc

logger = init_logger(__name__)


class MultiprocessingGPUExecutor:

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        assert world_size == tensor_parallel_size, (
            f"world_size ({world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}) -- pipeline "
            f"parallelism is not yet implemented in v1")

        # Ensure that VLLM_INSTANCE_ID is set, to be inherited by workers
        os.environ["VLLM_INSTANCE_ID"] = get_vllm_instance_id()

        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        # Configure thread parallelism if OMP_NUM_THREADS isn't set
        #
        # Helps to avoid CPU contention. The default of spawning a thread per
        # core combined with multiprocessing for each GPU can have a negative
        # impact on performance. The contention is amplified when running in a
        # container where CPU limits can cause throttling.
        default_omp_num_threads = 1
        if "OMP_NUM_THREADS" not in os.environ and (
                current_parallelism :=
                torch.get_num_threads()) > default_omp_num_threads:
            logger.warning(
                "Reducing Torch parallelism from %d threads to %d to avoid "
                "unnecessary CPU contention. Set OMP_NUM_THREADS in the "
                "external environment to tune this value as needed.",
                current_parallelism, default_omp_num_threads)
            os.environ["OMP_NUM_THREADS"] = str(default_omp_num_threads)
            torch.set_num_threads(default_omp_num_threads)

        # workaround for https://github.com/vllm-project/vllm/issues/6103
        if world_size > 1:
            maybe_set_triton_cache_manager()

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # 127.0.0.1 for communication.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port())

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        self.scheduler_output_sender = MessageQueue(world_size, world_size)
        scheduler_output_sender_handle = self.scheduler_output_sender.export_handle()

        # Background Threads and Queues for IO from Worker 0. 
        # These enable us to overlap ZMQ socket IO, and to overlap some 
        # serialization/deserialization with the model forward pass.
        # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
        self.output_queue = queue.Queue()
        input_path = get_open_zmq_ipc_path()
        ready_path = get_open_zmq_ipc_path()
        threading.Thread(target=self.process_input_from_worker,
                         args=(input_path, ),
                         daemon=True).start()

        # Create workers
        self.workers: List[WorkerProc] = []
        for rank in range(world_size):
            worker = WorkerProc(vllm_config, rank, rank,
                        distributed_init_method, scheduler_output_sender_handle,
                        input_path, ready_path)
            self.workers.append(worker)

        # Message queues are not valid until all readers and writers call
        # wait_until_ready()
        self.model_output_receiver.wait_until_ready()

        # Read the number of KV cache blocks from Worker 0 
        self.num_available_blocks = _read_num_blocks()

    @contextmanager
    def make_socket(self, path: str, type: Any) -> Iterator[zmq.Socket]:
        """Context manager for use """

        ctx = zmq.Context()
        try:
            socket = ctx.socket(type)

            if type == zmq.constants.PULL:
                socket.connect(path)
            elif type == zmq.constants.PUSH:
                socket.bind(path)
            else:
                raise ValueError(f"Unknown Socket Type: {type}")

            yield socket

        except KeyboardInterrupt:
            logger.debug("Worker had Keyboard Interrupt.")

        finally:
            ctx.destroy(linger=0)

    def _read_num_blocks(self):
        """Busy loop to read the number of KV cache blocks."""
        request = self.input_queue.get()
        if isinstance(request, Tuple[int, int]):
            return request
        else:
            raise ValueError(f"Unknown RequestType: {request}")


    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.num_available_blocks

    def initialize_cache(self, num_gpu_blocks: int) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # We've already done this.
        ...

    def _read_num_blocks(self):
        """Busy loop to read the number of KV cache blocks."""

        # 1) Poll the input queue until there is work to do.
        req = self.input_queue.get(timeout=POLLING_TIMEOUT_S)
        self._handle_client_request(req)

        if isinstance(request, Tuple[int, int]):
            return request

        else:
            raise ValueError(f"Unknown RequestType: {request}")

    def process_input_from_worker(self, input_path: str):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        decoder_num_blocks = msgspec.msgpack.Decoder(Tuple[int, int])
        decoder_model_runner_output = msgspec.msgpack.Decoder(ModelRunnerOutput)

        with self.make_socket(input_path, zmq.constants.PULL) as socket:
            while True:
                # (RequestType, RequestData)
                type_frame, data_frame = socket.recv_multipart(copy=False)
                request_type = type_frame.buffer
                request_data = data_frame.buffer

                # Deserialize the request data.
                # TODO: refactor so that num blocks is only read once outside of this loop
                if request_type == WorkerOutputType.NUM_BLOCKS.value:
                    request = decoder_num_blocks.decode(request_data)
                elif request_type == WorkerOutputType.MODEL_RUNNER_OUTPUT.value:
                    request = decoder_model_runner_output.decode(request_data)
                else:
                    raise ValueError(f"Unknown RequestType: {request_type}")

                # Push to input queue for core busy loop.
                self.input_queue.put_nowait(request)

    def execute_model(
        self,
        scheduler_output,
    ) -> ModelRunnerOutput:
        self.scheduler_output_sender.enqueue(scheduler_output)
        model_output = self.input_queue.get()
        return model_output

    def check_health(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return
