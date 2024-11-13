import os
from functools import partial
from typing import Any, List, Optional, Tuple

import torch

from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.executor.multiproc_worker_utils import (ProcessWorkerWrapper,
                                                  ResultHandler, WorkerMonitor)
from vllm.logger import init_logger
from vllm.triton_utils import maybe_set_triton_cache_manager
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        get_vllm_instance_id)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_worker import MultiprocessingWorker

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

        # Create workers
        self.workers: List[ProcessWorkerWrapper] = []
        result_handler = ResultHandler()
        for rank in range(world_size):
            worker = ProcessWorkerWrapper(
                result_handler,
                partial(
                    self._create_worker,
                    **dict(
                        rank=rank,
                        local_rank=rank,
                        distributed_init_method=distributed_init_method,
                    )))
            self.workers.append(worker)

        self.worker_monitor = WorkerMonitor(self.workers, result_handler)
        result_handler.start()
        self.worker_monitor.start()

        self._run_workers("initialize")
        self._run_workers("load_model")

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        self.scheduler_output_sender = MessageQueue(world_size, world_size)
        model_output_receiver_handle = self._run_workers(
            "initialize_message_queues",
            self.scheduler_output_sender.export_handle())[0]
        self.model_output_receiver = MessageQueue.create_from_handle(
            model_output_receiver_handle, 0)

        # Message queues are not valid until all readers and writers call
        # wait_until_ready()
        wait_futures = self._run_workers("finish_message_queue_initialization",
                                         run_async=True)
        self.scheduler_output_sender.wait_until_ready()
        self.model_output_receiver.wait_until_ready()
        for output in wait_futures:
            output.get()

        # Flag that's set if workers are waiting in the main execution loop
        self.workers_in_busy_loop = False

    def _create_worker(
        self,
        local_rank: int = 0,
        rank: int = 0,
        distributed_init_method: Optional[str] = None
    ) -> MultiprocessingWorker:
        """Return worker init args for a given rank."""
        # see https://github.com/NVIDIA/nccl/issues/1234
        os.environ['NCCL_CUMEM_ENABLE'] = '0'

        if distributed_init_method is None:
            distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port())

        return MultiprocessingWorker(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
        )

    def _run_workers(
        self,
        method: str,
        *args,
        run_async: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers.

        Args:
            run_async: If True the method will be run asynchronously and return
            a list of futures rather than blocking on the results.
        """

        worker_outputs = [
            worker.execute_method(method, *args, **kwargs)
            for worker in self.workers
        ]

        if run_async:
            return worker_outputs
        else:
            return [output.get() for output in worker_outputs]

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers("determine_num_available_blocks")

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)

        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-GPU configurations.
        logger.info("# GPU blocks: %d", num_gpu_blocks)
        self._run_workers("initialize_cache", num_gpu_blocks)
        self._run_workers("compile_or_warm_up_model")

    def execute_model(
        self,
        scheduler_output,
    ) -> ModelRunnerOutput:
        # TODO: Find a better way to start this loop
        if not self.workers_in_busy_loop:
            self._run_workers("execute_model_busy_loop", run_async=True)
            self.workers_in_busy_loop = True

        self.scheduler_output_sender.enqueue(scheduler_output)
        model_output = self.model_output_receiver.dequeue(ModelRunnerOutput)
        return model_output

    def check_health(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return
