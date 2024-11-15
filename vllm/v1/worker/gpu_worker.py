"""A GPU worker class."""
import gc
import os
from typing import TYPE_CHECKING, Optional, Tuple, Any, Iterator
from contextlib import contextmanager
from multiprocessing.process import BaseProcess
import queue
import threading

import torch
import torch.distributed
import zmq
import msgspec

from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size
from vllm.v1.core.scheduler_output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput


class Worker:

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
    ):

        # TODO: use WorkerBase.__init__(self, vllm_config=vllm_config)
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

        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

    def initialize(self):
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        self.model_runner = GPUModelRunner(self.vllm_config, self.device)

    def load_model(self) -> None:
        self.model_runner.load_model()

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        peak_memory = self.init_gpu_memory - free_gpu_memory
        assert peak_memory > 0, (
            "Error in memory profiling. "
            f"Initial free memory {self.init_gpu_memory}, current free memory"
            f" {free_gpu_memory}. This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

        cache_block_size = _get_cache_block_size(self.cache_config,
                                                 self.model_config,
                                                 self.parallel_config)
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization -
             peak_memory) // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        # if self.model_runner.lora_manager:
        #     self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, 0

    def initialize_cache(self, num_gpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks."""
        if num_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `gpu_memory_utilization` when "
                             "initializing the engine.")

        max_seq_len = self.cache_config.block_size * num_gpu_blocks
        max_model_len = self.model_config.max_model_len
        if max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`gpu_memory_utilization` or decreasing `max_model_len` when "
                "initializing the engine.")

        self.model_runner.initialize_kv_cache(num_gpu_blocks)

    def compile_or_warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        output = self.model_runner.execute_model(scheduler_output)
        return output


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle, # TODO: typing
        output_path: str, # W0 sends ModelRunnerOutput
        ready_path: str,
    ):
        self.rank = rank
        self.worker = Worker(vllm_config, local_rank, rank,
                             distributed_init_method)

        # Initialize MessageQueue for receiving SchedulerOutput
        self.scheduler_output_receiver = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank)

        # Worker 0 initializes a message queue for sending the model output
        if self.rank == 0:
            self.output_queue = queue.Queue()
            threading.Thread(target=self.process_output_socket,
                args=(output_path, ),
                daemon=True).start()

        # The order of the next two steps (1. sending the ready signal and 
        # 2. waiting on the shm message queue) must be done in this order
        # to avoid deadlocks, as the EngineCoreProcess must first initialize
        # all workers and then wait on the message queue.

        # Send Readiness signal to EngineClient.
        with self.make_socket(ready_path, zmq.constants.PUSH) as ready_socket:
            ready_socket.send_string(WorkerProc.READY_STR)

        # All processors wait on the shm message queue
        self.scheduler_output_receiver.wait_until_ready()

        self.worker.initialize()
        self.worker.load_model()

        my_num_blocks = torch.tensor(self.worker.determine_num_available_blocks())
        min_num_blocks = torch.distributed.all_reduce(my_num_blocks, op=dist.ReduceOp.MIN)
        num_gpu_blocks = min_num_blocks[0]

        # Send the number of GPU blocks back to the model executor.
        if self.rank == 0:
            self.output_queue.push_back(min_num_blocks)

        if vllm_config.cache_config.num_gpu_blocks_override is not None:
            num_gpu_blocks_override = cache_config.num_gpu_blocks_override
            logger.info(
                "Overriding num_gpu_blocks=%d with "
                "num_gpu_blocks_override=%d", num_gpu_blocks,
                num_gpu_blocks_override)
            num_gpu_blocks = num_gpu_blocks_override

        self.worker.initialize_cache(num_gpu_blocks)

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
                


    @staticmethod
    def make_worker_core_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle, # Receive SchedulerOutput
        output_path: str, # W0 sends ModelRunnerOutput
        ready_path: str,
    ) -> BaseProcess:
        context = multiprocessing.get_context("spawn")

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "output_path": output_path,
            "ready_path": ready_path,
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(target=Worker.run_engine_core,
                               kwargs=process_kwargs)
        proc.start()

        # Wait for startup
        Worker.wait_for_startup(proc, ready_path)
        return proc

    @staticmethod
    def run_worker(*args, **kwargs):
        """Launch Worker busy loop in background process."""

        try:
            worker = WorkerProc(*args, **kwargs)
            worker.execute_model_busy_loop()

        except KeyboardInterrupt:
            logger.debug("Worker interrupted.")

        except BaseException as e:
            logger.exception(e)
            raise e

    @staticmethod
    def wait_for_startup(
        proc: BaseProcess,
        ready_path: str,
    ) -> None:
        """Wait until the Worker is ready."""

        try:
            sync_ctx = zmq.Context()  # type: ignore[attr-defined]
            socket = sync_ctx.socket(zmq.constants.PULL)
            socket.connect(ready_path)

            # Wait for Worker to send Worker.READY_STR.
            while socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                logger.debug("Waiting for Worker to startup.")

                if not proc.is_alive():
                    raise RuntimeError("WorkerProc failed to start.")

            message = socket.recv_string()
            assert message == WorkerProc.READY_STR

        except BaseException as e:
            logger.exception(e)
            raise e

        finally:
            sync_ctx.destroy(linger=0)

    # Message queues are not valid until all readers and writers call
    # wait_until_ready()
    def finish_message_queue_initialization(self):
        if self.rank == 0:
            self.model_output_sender.wait_until_ready()

    # Main busy loop for Multiprocessing Workers
    def execute_model_busy_loop(self):
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1000,  # Wait 1000 steps so we profile middle iters
                    warmup=10,  # Warm up the scheduler
                    active=3,  # Run a small number of steps so it's legible
                    repeat=1,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "./traces/",
                    worker_name=f"worker_{self.worker.rank}",
                ),
                with_stack=True,
        ) as p:

            while True:
                scheduler_output = self.scheduler_output_receiver.dequeue(
                    SchedulerOutput)
                output = self.worker.execute_model(scheduler_output)
                if self.worker.rank == 0:
                    self.model_output_sender.put_nowait(output)

                p.step()

    def process_output_socket(self, output_path: str):
        """Output socket IO thread to send data to coordinator."""
        
        # Msgpack serialization encoding
        encoder = msgspec.msgpack.Encoder()
        # Reuse send buffer.
        buffer = bytearray()
        
        with self.make_socket(output_path, zmq.constants.PUSH) as socket:
            while True:
                # Get next item to send from output queue
                output = self.output_queue.get()
                
                # Determine the type of the output
                if isinstance(output, tuple) and len(output) == 2 and all(isinstance(x, int) for x in output):
                    type_frame = WorkerOutputType.NUM_BLOCKS.value
                elif isinstance(output, ModelRunnerOutput):
                    type_frame = WorkerOutputType.MODEL_RUNNER_OUTPUT.value
                else:
                    raise ValueError(f"Unknown output type: {type(output)}")

                # Serialize and send to engine core process
                data_frame = encoder.encode_into(output, buffer)
                socket.send_multipart([type_frame, buffer], copy=False)

    # Wrapper methods defined here
    def initialize(self):
        self.worker.initialize()

    def load_model(self):
        self.worker.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        return self.worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int) -> None:
        self.worker.initialize_cache(num_gpu_blocks)

    def compile_or_warm_up_model(self) -> None:
        self.worker.compile_or_warm_up_model()


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:  # noqa: SIM102
        if not current_platform.has_device_capability(80):
            capability = current_platform.get_device_capability()
            gpu_name = current_platform.get_device_name()

            if capability is None:
                compute_str = "does not have a compute capability"
            else:
                version_str = capability.as_version_str()
                compute_str = f"has compute capability {version_str}"

            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU {compute_str}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")


def _get_cache_block_size(
    cache_config: CacheConfig,
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> int:
    head_size = model_config.get_head_size()
    num_heads = model_config.get_num_kv_heads(parallel_config)
    num_attention_layers = model_config.get_num_attention_layers(
        parallel_config)

    key_cache_block = cache_config.block_size * num_heads * head_size
    value_cache_block = key_cache_block
    total = num_attention_layers * (key_cache_block + value_cache_block)
    if cache_config.cache_dtype == "auto":
        dtype = model_config.dtype
    else:
        dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
    dtype_size = get_dtype_size(dtype)
    return dtype_size * total
