# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import socket
import time
from collections import defaultdict

import msgspec
import msgspec.msgpack
import pytest
import zmq
from tqdm import tqdm

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVEventsConfig, KVTransferConfig
from vllm.distributed.kv_events import BlockStored, KVEventBatch
from vllm.platforms import current_platform

_ATTN_BACKENDS: list[str] = []
if current_platform.is_cuda():
    _ATTN_BACKENDS = ["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"]
elif current_platform.is_rocm():
    _ATTN_BACKENDS = ["TRITON_ATTN"]

# (model, attn_backend | None, block_size | None, uses_hma)
#
# - Llama: tested with each attention backend and a custom block_size.
# - Gemma-3: HMA (sliding window + full attention), default backend.
# - Mamba-130m: HMA (attention-free, acts like sliding_window=1),
#   default backend.  Prefix caching must be force-enabled.
# - Falcon-H1-0.5B-Instruct: HMA (parallel SSM/attention in every layer).
#   After page-size unification the mamba and attention groups have
#   different block sizes.
MODEL_PARAMS: list[tuple[str, str | None, int | None, bool]] = [
    ("meta-llama/Llama-3.2-1B-Instruct", backend, 48, False)
    for backend in _ATTN_BACKENDS
]
# HMA / Mamba models are only tested on CUDA (not ROCm).
if current_platform.is_cuda():
    MODEL_PARAMS += [
        ("google/gemma-3-1b-it", None, 48, True),
        # Mamba not working in main
        # ("state-spaces/mamba-130m-hf", None, 48, True),
        # Falcon-H1: parallel hybrid (every layer has both attention and SSM).
        # The mamba and attention groups end up with different GPU block sizes
        # after page-size unification, so we leave cpu_block_size=None
        # (block_size_factor stays 1).
        ("tiiuae/Falcon-H1-0.5B-Instruct", None, None, True),
    ]

# Maximum time (seconds) to wait for the async CPU offload transfer
# to complete before giving up.
_RESET_CACHE_TIMEOUT = 30 if current_platform.is_rocm() else 10

# ZMQ poll timeout (ms) for the first event.
_FIRST_EVENT_POLL_MS = 10_000 if current_platform.is_rocm() else 1000

# Hard ceiling (seconds) on how long get_new_stored_events may loop,
# to prevent hangs if non-CPU events keep arriving indefinitely.
_EVENT_DRAIN_TIMEOUT = 60


class MockSubscriber:
    """Helper class to receive and verify published events"""

    def __init__(
        self,
        endpoint: str,
        topic: str,
    ):
        self.ctx = zmq.Context.instance()
        self.topic_bytes = topic.encode("utf-8")

        # Set up subscriber socket
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, self.topic_bytes)
        self.sub.connect(endpoint)

        self.store_events: dict[str, BlockStored] = defaultdict(list)
        self.decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

    def get_new_stored_events(self, medium: str) -> list[BlockStored]:
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)

        poll_ms = _FIRST_EVENT_POLL_MS
        deadline = time.monotonic() + _EVENT_DRAIN_TIMEOUT
        while time.monotonic() < deadline:
            events = dict(poller.poll(poll_ms))

            if events.get(self.sub) != zmq.POLLIN:
                return self.store_events.pop(medium, [])

            topic_bytes, _, payload = self.sub.recv_multipart()

            assert topic_bytes == self.topic_bytes

            event_batch = self.decoder.decode(payload)
            assert isinstance(event_batch, KVEventBatch)
            for event in event_batch.events:
                if isinstance(event, BlockStored):
                    self.store_events[event.medium].append(event)
                    poll_ms = 100

        return self.store_events.pop(medium, [])

    def close(self):
        """Clean up resources"""
        self.sub.close()


def _dummy_engine_step(llm: LLM):
    llm.generate(
        [TokensPrompt(prompt_token_ids=[0])],
        SamplingParams(max_tokens=1),
        use_tqdm=False,
    )


def _wait_for_prefix_cache_reset(llm: LLM) -> None:
    """Wait for async offload transfers to finish so prefix cache can reset.

    The GPU-to-CPU offload runs on a CUDA stream asynchronously. While blocks
    are still held by the offload worker, ``reset_prefix_cache`` returns
    ``False``. Between retries we send a dummy single-token prefill to force
    the engine to step, which polls the worker for completed transfers and
    frees GPU blocks.
    """
    _dummy_params = SamplingParams(max_tokens=1)
    deadline = time.monotonic() + _RESET_CACHE_TIMEOUT
    while not llm.reset_prefix_cache():
        if time.monotonic() > deadline:
            raise TimeoutError(
                "reset_prefix_cache did not succeed within "
                f"{_RESET_CACHE_TIMEOUT}s - async offload may be stuck"
            )
        # Force an engine step so the scheduler polls get_finished()
        # and releases GPU blocks held by in-flight async stores.
        _dummy_engine_step(llm)


def _prepare_load_from_secondary_tier(llm: LLM):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
    from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager

    def get_tiering_manager():
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        connector = scheduler.connector
        assert isinstance(connector, OffloadingConnector)
        manager = connector.connector_scheduler.manager
        assert isinstance(manager, TieringOffloadingManager)
        return manager

    def _evict_all_cpu_blocks(manager: TieringOffloadingManager) -> None:
        """Remove all evictable blocks from CPU primary tier only."""
        manager._maybe_process_finished_jobs()  # finish in-flight tier jobs first
        primary = manager.primary_tier
        bt = primary.block_tracker

        deadline = time.monotonic() + _RESET_CACHE_TIMEOUT
        while bt.get_num_free_blocks() != bt.num_blocks:
            # Force and engine step to trigger the offloading mechanics
            _dummy_engine_step(llm)

            if time.monotonic() > deadline:
                raise TimeoutError(
                    "cpu eviction did not succeed within "
                    f"{_RESET_CACHE_TIMEOUT}s - async offload may be stuck. "
                    f"{bt.num_blocks=} != {bt.get_num_free_blocks()=}."
                )

            assert isinstance(primary, CPUOffloadingManager)
            for key, block in list(primary._policy.blocks.items()):
                if block.ref_cnt == 0:
                    primary._policy.remove(key)
                    primary.block_tracker.free_block(block)
            list(manager.take_events())

    # Clear all CPU cache
    _evict_all_cpu_blocks(get_tiering_manager())

    # Remove GPU cache
    _wait_for_prefix_cache_reset(llm)


def _verify_kv_stores(subscriber: MockSubscriber | None, medium: str):
    if subscriber is not None:
        assert subscriber.get_new_stored_events(medium), (
            f"No {medium} stored events received."
            "async offload may not have completed in time"
        )


def _latency_test(
    llm: LLM,
    subscriber: MockSubscriber | None,
    is_tiered_storage_offloader: bool = False,
):
    sampling_params = SamplingParams(max_tokens=1)

    def _generate_time(prompts) -> float:
        start_time = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        return time.time() - start_time

    num_times_cpu_better_than_cold = 0
    num_times_secondary_tier_better_than_cold = 0
    num_tests = 10
    total_cold_time = 0.0
    total_gpu_hit_time = 0.0
    total_cpu_hit_time = 0.0
    total_secondary_tier_hit_time = 0.0
    max_model_len = llm.llm_engine.vllm_config.model_config.max_model_len
    # Use a long prompt that fits within the model's context window.
    prompt_len = min(10001, max_model_len - 1)
    prompt_token_ids = [0] * prompt_len
    for i in tqdm(range(num_tests), desc="Running tests"):
        prompt_token_ids[0] = i
        prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)]

        # run generation - this should trigger saving KV cache
        cold_time = _generate_time(prompts)
        total_cold_time += cold_time

        # run generation again - should hit the GPU prefix cache
        total_gpu_hit_time += _generate_time(prompts)

        # Wait for the async CPU offload to finish, then reset prefix cache
        # so the next generate() must reload from CPU rather than GPU.
        _wait_for_prefix_cache_reset(llm)

        # Verify CPU stored events arrived (offload is done before we
        # attempt to load from CPU).
        _verify_kv_stores(subscriber, "CPU")

        # run generation again - this should trigger loading from CPU
        cpu_hit_time = _generate_time(prompts)
        total_cpu_hit_time += cpu_hit_time

        if is_tiered_storage_offloader:
            _verify_kv_stores(subscriber, "STORAGE")
            _prepare_load_from_secondary_tier(llm)
            secondary_tier_hit_time = _generate_time(prompts)
            total_secondary_tier_hit_time += secondary_tier_hit_time
            if secondary_tier_hit_time < cold_time:
                num_times_secondary_tier_better_than_cold += 1

        if cpu_hit_time < cold_time:
            num_times_cpu_better_than_cold += 1

    print("Average times:")
    print(f"    Cold: {total_cold_time * 1000 / num_tests:.2f}ms")
    print(f"    GPU hit: {total_gpu_hit_time * 1000 / num_tests:.2f}ms")
    print(f"    CPU hit: {total_cpu_hit_time * 1000 / num_tests:.2f}ms")
    if is_tiered_storage_offloader:
        avg_hit_time = total_secondary_tier_hit_time * 1000 / num_tests
        print(f"    Secondary Tier hit: {avg_hit_time:.2f}ms")

    assert num_times_cpu_better_than_cold >= 0.8 * num_tests
    # Loading from storage tier is typically very slow.
    # if is_tiered_storage_offloader:
    #    assert num_times_secondary_tier_better_than_cold >= 0.8 * num_tests


def _accuracy_test(
    llm: LLM,
    subscriber: MockSubscriber | None,
    is_tiered_storage_offloader: bool = False,
):
    sampling_params = SamplingParams(max_tokens=1)
    extra_config = (
        llm.llm_engine.vllm_config.kv_transfer_config.kv_connector_extra_config
    )
    cpu_block_size = extra_config.get("block_size")
    if cpu_block_size is None:
        # No custom offloaded block_size: offloaded blocks match GPU blocks.
        # Use the hash block_size (cache_config.block_size) for alignment.
        cpu_block_size = llm.llm_engine.vllm_config.cache_config.block_size

    # Clear out any pending CPU store events.
    if subscriber is not None:
        subscriber.get_new_stored_events("CPU")

    # Pad prompt so its token count is a multiple of cpu_block_size.
    # Use the tokenizer directly to avoid expensive llm.generate() calls.
    tokenizer = llm.get_tokenizer()
    prompt = "Let's count to 10. One, two, three, four,"
    # +1 for sliding window groups
    while len(tokenizer.encode(prompt)) % (cpu_block_size + 1) != 0:
        prompt = ". " + prompt

    # Seed the CPU cache with the prompt.
    llm.generate(prompt, sampling_params, use_tqdm=False)

    def _test():
        test_count = 20
        results = llm.generate([prompt] * test_count, sampling_params, use_tqdm=False)
        success_count = sum(1 for r in results if r.outputs[0].text == " five")
        assert success_count >= 0.5 * test_count

    # Dummy engine step for force get_finished polling
    _dummy_engine_step(llm)

    # Wait for the async CPU offload to finish, then reset prefix cache
    # so the next generate() must reload from CPU rather than GPU.
    _wait_for_prefix_cache_reset(llm)
    _verify_kv_stores(subscriber, "CPU")
    _test()

    # Dummy engine step for force get_finished polling
    _dummy_engine_step(llm)

    if is_tiered_storage_offloader:
        _verify_kv_stores(subscriber, "STORAGE")
        _prepare_load_from_secondary_tier(llm)
        _test()


@pytest.mark.parametrize("model, attn_backend, cpu_block_size, uses_hma", MODEL_PARAMS)
def test_cpu_offloading(
    model: str,
    attn_backend: str | None,
    cpu_block_size: int | None,
    uses_hma: bool,
) -> None:
    """
    Tests OffloadingConnector with CPUOffloadingSpec.
    """
    # configure OffloadingConnector (spec_name=CPUOffloadingSpec by default)
    extra_config: dict = {"cpu_bytes_to_use": 500 << 20}
    if cpu_block_size is not None:
        extra_config["block_size"] = cpu_block_size
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )

    # KV events are incompatible with HMA (setting kv_events_config
    # would force HMA off), so only enable them for non-HMA models.
    subscriber: MockSubscriber | None = None
    kv_events_config: KVEventsConfig | None = None
    if not uses_hma:
        port: int
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", 0))
            port = s.getsockname()[1]

        events_endpoint = f"tcp://*:{port}"
        kv_events_config = KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint=events_endpoint,
            topic="test",
        )

    # Attention-free / hybrid models disable prefix caching by default
    # (ModelConfig.is_prefix_caching_supported returns False).  Without it,
    # mamba_block_size falls back to max_model_len, making GPU blocks too
    # large for any reasonable offloaded block_size.  Force-enable it.
    force_prefix_caching = uses_hma

    llm = LLM(
        model=model,
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        kv_events_config=kv_events_config,
        kv_transfer_config=kv_transfer_config,
        **({"attention_config": {"backend": attn_backend}} if attn_backend else {}),
        # HMA models need explicit opt-in when kv_transfer_config is set
        **({"disable_hybrid_kv_cache_manager": False} if uses_hma else {}),
        **({"enable_prefix_caching": True} if force_prefix_caching else {}),
        # ROCm: batch size 1 to reduce variability
        **({"max_num_seqs": 1} if current_platform.is_rocm() else {}),
    )

    if kv_events_config is not None:
        events_endpoint = events_endpoint.replace("*", "127.0.0.1")
        subscriber = MockSubscriber(events_endpoint, topic=kv_events_config.topic)

    try:
        _latency_test(llm, subscriber)
        _accuracy_test(llm, subscriber)
    finally:
        if subscriber is not None:
            subscriber.close()
        del llm


_MODEL_PARAMS = [
    (model, attn, block_size, uses_hma, tp)
    for tp in [1, 2]
    for model, attn, block_size, uses_hma in MODEL_PARAMS
]


@pytest.mark.parametrize(
    "model, attn_backend, cpu_block_size, uses_hma, tp_size", _MODEL_PARAMS
)
def test_tiered_storage_offloading(
    model: str,
    attn_backend: str | None,
    cpu_block_size: int | None,
    uses_hma: bool,
    tp_size: int,
    tmp_path,  # tmp_path generated by pytest
    monkeypatch,
) -> None:
    # To be able to reach into the engine_core to get the tiering manager.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # KV events are incompatible with HMA (setting kv_events_config
    # would force HMA off), so only enable them for non-HMA models.
    subscriber: MockSubscriber | None = None
    kv_events_config: KVEventsConfig | None = None
    if not uses_hma:
        port: int
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", 0))
            port = s.getsockname()[1]

        events_endpoint = f"tcp://*:{port}"
        kv_events_config = KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint=events_endpoint,
            topic="test",
        )

    # Attention-free / hybrid models disable prefix caching by default
    # (ModelConfig.is_prefix_caching_supported returns False).  Without it,
    # mamba_block_size falls back to max_model_len, making GPU blocks too
    # large for any reasonable offloaded block_size.  Force-enable it.
    force_prefix_caching = uses_hma

    extra_config: dict = {
        "spec_name": "TieringOffloadingSpec",
        "cpu_bytes_to_use": 500 << 20,  # ~500MB
        "secondary_tiers": [
            {
                "type": "STORAGE",
                "storage_root_path": str(tmp_path),
                "max_storage_size_gb": 1,
            }
        ],
    }
    if cpu_block_size is not None:
        extra_config["block_size"] = cpu_block_size

    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )

    llm = LLM(
        model=model,
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=tp_size,
        kv_events_config=kv_events_config,
        kv_transfer_config=kv_transfer_config,
        **({"attention_config": {"backend": attn_backend}} if attn_backend else {}),
        # HMA models need explicit opt-in when kv_transfer_config is set
        **({"disable_hybrid_kv_cache_manager": False} if uses_hma else {}),
        **({"enable_prefix_caching": True} if force_prefix_caching else {}),
        # ROCm: batch size 1 to reduce variability
        **({"max_num_seqs": 1} if current_platform.is_rocm() else {}),
    )

    if kv_events_config is not None:
        events_endpoint = events_endpoint.replace("*", "127.0.0.1")
        subscriber = MockSubscriber(events_endpoint, topic=kv_events_config.topic)

    try:
        # _latency_test(llm, subscriber, is_tiered_storage_offloader=True)
        _accuracy_test(llm, subscriber, is_tiered_storage_offloader=True)
    finally:
        if subscriber is not None:
            subscriber.close()
        llm.llm_engine.engine_core.shutdown()
        del llm


def test_tiering_offloading() -> None:
    """Tests OffloadingConnector with TieringOffloadingSpec."""
    extra_config: dict = {
        "cpu_bytes_to_use": 500 << 20,
        "block_size": 48,
        "spec_name": "TieringOffloadingSpec",
        "secondary_tiers": [{"type": "example"}],
    }
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )

    port: int
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]
    events_endpoint = f"tcp://*:{port}"
    kv_events_config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=events_endpoint,
        topic="test",
    )

    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        kv_events_config=kv_events_config,
        kv_transfer_config=kv_transfer_config,
    )
    subscriber = MockSubscriber(
        events_endpoint.replace("*", "127.0.0.1"),
        topic=kv_events_config.topic,
    )
    try:
        _latency_test(llm, subscriber)
        _accuracy_test(llm, subscriber)
    finally:
        subscriber.close()
        del llm
