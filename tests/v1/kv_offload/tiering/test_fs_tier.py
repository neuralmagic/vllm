# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for FileSystemTierManager.

These tests use real disk I/O to verify the filesystem tier implementation.
The tier manager writes KV cache blocks to disk and reads them back, verifying
data integrity throughout the process.
"""

import gc
import mmap
import os
import random
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingKVEventsConfig,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.fs import thread_pool as thread_pool_mod
from vllm.v1.kv_offload.tiering.fs.manager import (
    FileSystemTierManager,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

# The work-stealing pool moved its queues and per-block I/O into the
# fs_io_C extension; there is no pure-Python fallback, so every test in this
# module requires it to be built.
pytestmark = pytest.mark.skipif(
    not thread_pool_mod._HAS_FSIO_POOL_C, reason="fs_io_C extension not built"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUM_BLOCKS = 8
_BLOCK_ELEMENTS = 128 * mmap.PAGESIZE  # 2MB per block for pagesize 4096.
_STRESS_BLOCK_ELEMENTS = mmap.PAGESIZE // 4  # 1 page (float32 * 4 bytes).
_DTYPE: torch.dtype = torch.float32
_CTX = ReqContext(req_id="test")


def _make_offloading_spec(
    enable_kv_cache_events: bool = False,
    *,
    tp_size: int = 1,
    rank: int = 0,
    world_size: int | None = None,
    replicated_layout: bool = False,
    is_parallelism_agnostic: bool = False,
) -> MagicMock:
    """Mock spec with an explicit global KV events flag."""
    if world_size is None:
        world_size = tp_size
    spec = MagicMock()
    spec.config = OffloadingConfig(
        groups=(),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=enable_kv_cache_events,
        extra_config={},
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float32"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=world_size,
            tp_size=tp_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=is_parallelism_agnostic,
        ),
        replicated_layout=replicated_layout,
    )
    spec.blocks_per_chunk = 1
    spec.kv_events_config = OffloadingKVEventsConfig(
        enable_kv_cache_events=enable_kv_cache_events,
        self_describing_kv_events=False,
    )
    return spec


_MOCK_OFFLOADING_SPEC = _make_offloading_spec(enable_kv_cache_events=False)


def key(n: int) -> OffloadKey:
    return make_offload_key(n.to_bytes(8, "big"), 0)


def make_job(
    job_id: int,
    keys: list[OffloadKey],
    block_ids: list[int] | None = None,
    is_promotion: bool = False,
) -> TransferJob:
    if block_ids is None:
        block_ids = list(range(len(keys)))
    return TransferJob(
        job_id=job_id,
        keys=keys,
        block_ids=np.array(block_ids, dtype=np.int64),
        is_promotion=is_promotion,
        req_context=_CTX,
    )


def drain(tier: FileSystemTierManager) -> list:
    """Block until all in-flight jobs finish, then collect results."""
    tier.drain_jobs()
    return list(tier.get_finished_jobs())


def lookup_and_wait(
    tier: FileSystemTierManager,
    keys: list[OffloadKey],
    ctx: ReqContext = _CTX,
    timeout: float = 1.0,
) -> list[LookupResult]:
    """Perform a full async lookup cycle and return resolved results."""
    for k in keys:
        tier.lookup(k, ctx)
    tier.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not tier._lookup_manager._pending_results.empty():
            break
        time.sleep(0.01)
    return [tier.lookup(k, ctx) for k in keys]


def _page_aligned_zero_tensor(
    num_blocks: int, block_elements: int, dtype: torch.dtype = _DTYPE
) -> torch.Tensor:
    page_size = mmap.PAGESIZE
    dtype_num_bytes = torch.tensor([], dtype=dtype).element_size()

    num_bytes = num_blocks * block_elements * dtype_num_bytes
    num_bytes_aligned = num_bytes + page_size
    t = torch.zeros(num_bytes_aligned, dtype=torch.uint8)

    ptr = t.data_ptr()
    alignment_offset = ptr % page_size
    # Move tensor to next page regardless.
    shift = page_size - alignment_offset
    t = t[shift : shift + num_bytes]
    return t.view(dtype).view(num_blocks, block_elements)


def _page_aligned_rand_tensor(
    num_blocks: int, block_elements: int, dtype: torch.dtype = _DTYPE
) -> torch.Tensor:
    rand_tensor = _page_aligned_zero_tensor(num_blocks, block_elements)
    rand_tensor[:] = torch.rand(num_blocks, block_elements, dtype=dtype)
    return rand_tensor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fs_tier(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=mock_view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
    )
    yield tier, tensor
    tier.shutdown()


@pytest.fixture
def fs_tier_with_events(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=mock_view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
        enable_kv_events=True,
        locality="LOCAL",
    )
    yield tier
    tier.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lookup_empty_tier(fs_tier):
    tier, _ = fs_tier
    results = lookup_and_wait(tier, [key(1), key(2)])
    assert results == [LookupResult.MISS, LookupResult.MISS]


def test_store_creates_file_and_lookup_succeeds(fs_tier):
    tier, _ = fs_tier
    job = make_job(1, [key(1)], [0])
    tier.submit_store(job)
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert lookup_and_wait(tier, [key(1)]) == [LookupResult.HIT]
    dest = tier.file_mapper.get_file_name(key(1))
    assert os.path.exists(dest), f"Expected file at {dest}"


def test_store_then_load_roundtrip(fs_tier):
    tier, _ = fs_tier
    job_s = make_job(1, [key(1), key(2)], [0, 1])
    tier.submit_store(job_s)
    store_results = drain(tier)
    assert all(r.success for r in store_results)

    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]

    job_l = make_job(2, [key(1), key(2)], [2, 3], is_promotion=True)
    tier.submit_load(job_l)
    load_results = drain(tier)
    assert all(r.success for r in load_results)
    # A successful load must NOT touch the file: the delete path fires only on
    # a provable short read, so a good block stays on disk (guards against an
    # over-eager delete regressing to upstream's delete-on-any-error).
    for k in (key(1), key(2)):
        assert os.path.exists(tier.file_mapper.get_file_name(k))
    # Blocks stay on disk after load
    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]


def test_invalid_path_raises_at_construction():
    """Construction must fail immediately when the config file cannot be written."""
    tensor = _page_aligned_zero_tensor(32, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())

    with pytest.raises(OSError):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="fs",
            root_dir="/dev/null/invalid_path",
        )


@pytest.mark.parametrize("locality", ["local", ""])
def test_invalid_locality_raises_at_construction(tmp_path, locality):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)

    with pytest.raises(ValueError, match="Locality"):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=memoryview(tensor.numpy()),
            tier_type="fs",
            root_dir=str(tmp_path),
            locality=locality,
        )


def test_factory_forwards_locality_to_fs_tier(tmp_path):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            "type": "fs",
            "root_dir": str(tmp_path),
            "n_read_threads": 1,
            "n_write_threads": 1,
            "locality": "LOCAL",
        },
        memoryview(tensor.numpy()),
        _MOCK_OFFLOADING_SPEC,
    )
    try:
        assert isinstance(tier, FileSystemTierManager)
        assert tier.locality is Locality.LOCAL
    finally:
        tier.shutdown()


def test_failed_load_missing_file(fs_tier):
    """Test that loading a block whose file does not exist results in a failed job."""
    tier, _ = fs_tier
    job = make_job(1, [key(99)], [0], is_promotion=True)
    tier.submit_load(job)
    results = drain(tier)
    assert len(results) == 1
    assert not results[0].success


def test_multiple_jobs_tracked_independently(fs_tier):
    tier, _ = fs_tier
    job1 = make_job(1, [key(1)], [0])
    job2 = make_job(2, [key(2)], [1])
    tier.submit_store(job1)
    tier.submit_store(job2)
    results = drain(tier)
    job_ids = {r.job_id for r in results}
    assert job_ids == {1, 2}
    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]


def test_multi_block_job_partial_failure(fs_tier):
    """A load job where one block file is missing yields a single failed JobResult."""
    tier, _ = fs_tier
    # Store two of three keys
    tier.submit_store(make_job(1, [key(10), key(11)], [0, 1]))
    assert all(r.success for r in drain(tier))

    # Load all three — key(99) was never stored
    tier.submit_load(
        make_job(2, [key(10), key(11), key(99)], [0, 1, 2], is_promotion=True)
    )
    results = drain(tier)

    assert len(results) == 1
    assert results[0].job_id == 2
    assert not results[0].success


def test_shutdown_discards_pending_tasks(fs_tier):
    """Shutdown stops all worker threads immediately, without waiting for
    pending tasks to drain, and is safe to call more than once (the fixture's
    teardown calls it again)."""
    tier, _ = fs_tier
    # Submit many tasks to ensure some remain pending
    for i in range(10):
        tier.submit_store(make_job(i, [key(i)], [i % 4]))

    # Shutdown immediately without draining
    tier.shutdown()
    assert all(not t.is_alive() for t in tier._pool._threads)

    # Must not raise: shutdown() clears/destroys pool state on the first
    # call, so a second call must be a safe no-op.
    tier.shutdown()


@pytest.mark.parametrize("batch_size", [0, 1, 2, 5])
@pytest.mark.parametrize("use_c_ext", [True, False])
def test_store_load_data_integrity(fs_tier, monkeypatch, use_c_ext, batch_size):
    """Data written by store must be exactly recovered by load, for batches
    of any size -- including the empty batch."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, tensor = fs_tier
    # Populate tensor with random data
    tensor[:] = _page_aligned_rand_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)

    keys = [key(i) for i in range(batch_size)]
    store_block_ids = list(range(batch_size))
    load_block_ids = list(range(_NUM_BLOCKS - batch_size, _NUM_BLOCKS))
    expected = tensor[:batch_size].clone()

    tier.submit_store(make_job(1, keys, store_block_ids))
    store_results = drain(tier)
    assert len(store_results) == 1
    assert store_results[0].success
    assert all(os.path.exists(tier.file_mapper.get_file_name(k)) for k in keys)

    # reset tensor to prove data is read from disk
    tensor[:] = 0.0

    # Load into a range disjoint by index from the store ids, to also
    # exercise loading a block into a different id than it was stored from.
    tier.submit_load(make_job(2, keys, load_block_ids, is_promotion=True))
    load_results = drain(tier)
    assert len(load_results) == 1
    assert load_results[0].success

    for i, bid in enumerate(load_block_ids):
        assert torch.allclose(tensor[bid], expected[i]), (
            f"Block {bid} data mismatch after store+load"
        )


def test_store_load_roundtrip_without_o_direct(tmp_path, monkeypatch):
    """Buffered fallback must round-trip data when O_DIRECT is unsupported.

    Simulates filesystems (e.g. overlayfs, some NFS) that reject O_DIRECT by
    forcing the capability probe to report it unavailable.
    """
    monkeypatch.setattr(
        "vllm.v1.kv_offload.tiering.fs.manager.probe_o_direct",
        lambda _dir: False,
    )
    tensor = _page_aligned_rand_tensor(4, _BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
    )
    try:
        assert tier._use_o_direct is False

        keys = [key(0), key(1)]
        expected = tensor[:2].clone()
        tier.submit_store(make_job(1, keys, [0, 1]))
        assert all(r.success for r in drain(tier))

        tensor[:2] = 0.0
        tier.submit_load(make_job(2, keys, [2, 3], is_promotion=True))
        assert all(r.success for r in drain(tier))

        for i, bid in enumerate([2, 3]):
            assert torch.allclose(tensor[bid], expected[i])
    finally:
        tier.shutdown()


def test_wait_idle_blocks_until_tasks_complete(tmp_path, monkeypatch):
    """wait_idle must not return while a task is still in flight.

    The actual I/O (a single small block write) completes almost instantly,
    so to get a deterministic window in which the job is provably still
    in-flight, a gate is inserted at the wait_and_run() C-call boundary
    itself (the only per-item synchronization point in the new pool),
    rather than via an arbitrary injected callable -- the new pool only
    understands (path, offset) work items, not arbitrary callables.
    """
    import vllm.v1.kv_offload.tiering.fs.thread_pool as thread_pool_mod

    gate = threading.Event()
    real_wait_and_run = thread_pool_mod.wait_and_run

    def gated_wait_and_run(pool, load_priority):
        # Either the read- or write-priority thread may win the race to
        # steal this store job (both fall back to the other's queue when
        # idle), so the gate applies regardless of which one calls in.
        gate.wait(timeout=5.0)
        return real_wait_and_run(pool, load_priority)

    monkeypatch.setattr(thread_pool_mod, "wait_and_run", gated_wait_and_run)

    block_size = 16
    buf = bytearray(block_size)
    pool = DualQueueThreadPool(
        n_read_threads=1,
        n_write_threads=1,
        n_write_excl_threads=1,
        primary_kv_view=memoryview(buf),
        block_size=block_size,
    )
    pool.enqueue_store(job_id=1, paths=[str(tmp_path / "b0.bin")], offsets=[0])

    waiter = threading.Thread(target=pool.wait_idle)
    waiter.start()
    try:
        waiter.join(timeout=0.2)
        assert waiter.is_alive(), "wait_idle returned before task completed"
        gate.set()
        waiter.join(timeout=5.0)
        assert not waiter.is_alive(), "wait_idle did not unblock"
    finally:
        gate.set()
        pool.shutdown(wait=True)
        waiter.join(timeout=5.0)


def test_batch_lookup_c_extension(tmp_path):
    """Validates batch_lookup_C: empty, single, all-existing, all-missing,
    mixed ordering, and input type validation."""
    try:
        from vllm.fs_io_C import batch_lookup as batch_lookup_C
    except ImportError:
        pytest.skip("fs_io_C extension not built")

    # Setup
    all_exist = [str(tmp_path / f"e{i}.bin") for i in range(3)]
    for p in all_exist:
        open(p, "w").close()
    all_missing = [str(tmp_path / f"m{i}.bin") for i in range(3)]

    # Empty list
    assert batch_lookup_C([]) == []

    # Single existing / missing
    assert batch_lookup_C([all_exist[0]]) == [True]
    assert batch_lookup_C([all_missing[0]]) == [False]

    # All existing / all missing
    assert batch_lookup_C(all_exist) == [True, True, True]
    assert batch_lookup_C(all_missing) == [False, False, False]

    # Mixed — verifies index ordering is preserved
    paths = [val for pair in zip(all_exist, all_missing) for val in pair]
    assert batch_lookup_C(paths) == [True, False, True, False, True, False]

    # Input validation: non-list argument
    with pytest.raises(TypeError):
        batch_lookup_C(("/tmp/foo",))
    with pytest.raises(TypeError):
        batch_lookup_C(None)

    # Input validation: non-str elements in list
    with pytest.raises(TypeError):
        batch_lookup_C([None])
    with pytest.raises(TypeError):
        batch_lookup_C([b"/tmp/foo"])
    with pytest.raises(TypeError):
        batch_lookup_C([42])
    with pytest.raises(TypeError):
        batch_lookup_C([all_exist[0], None])  # valid first, invalid mid-list


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_batch_lookup_dispatch(fs_tier, monkeypatch, use_c_ext):
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    if use_c_ext and not mgr_mod._HAS_BATCH_LOOKUP_C:
        pytest.skip("fs_io_C extension not built")

    monkeypatch.setattr(mgr_mod, "_HAS_BATCH_LOOKUP_C", use_c_ext)

    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))

    results = lookup_and_wait(tier, [key(1), key(2)])
    assert results == [LookupResult.HIT, LookupResult.MISS]


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_out_of_bounds_block_id_smoke(fs_tier, monkeypatch, use_c_ext):
    """Smoke test: a block id beyond the primary tensor's block count must
    fail the job, for both the C extension and the Python fallback."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, tensor = fs_tier
    out_of_bounds_bid = tensor.shape[0]  # one past the last valid block

    tier.submit_store(make_job(1, [key(1)], [out_of_bounds_bid]))
    store_results = drain(tier)
    assert len(store_results) == 1
    assert not store_results[0].success

    tier.submit_load(make_job(2, [key(1)], [out_of_bounds_bid], is_promotion=True))
    load_results = drain(tier)
    assert len(load_results) == 1
    assert not load_results[0].success


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_failed_load_corrects_verdict_and_removes_corrupt_file(
    fs_tier, monkeypatch, use_c_ext
):
    """Failed-load livelock regression, covering the whole contract.

    A successful promotion leaves the cached HIT and the on-disk block intact.
    A promotion that short-reads a truncated (corrupt) block fails, and in
    get_finished_jobs() the tier removes the corrupt file (stores are atomic,
    so a too-short file is genuine corruption) and marks the cached verdict
    False. The SAME request's next lookup is then a MISS served from cache with
    NO re-probe, so the scheduler cannot re-issue the doomed promotion.
    """
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))
    path = tier.file_mapper.get_file_name(key(1))

    ctx = ReqContext(req_id="livelock-req")
    assert lookup_and_wait(tier, [key(1)], ctx=ctx) == [LookupResult.HIT]

    # A successful promotion must NOT touch the verdict or the file.
    tier.submit_load(make_job(2, [key(1)], [0], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1 and results[0].success
    assert tier.lookup(key(1), ctx) == LookupResult.HIT
    assert os.path.exists(path)

    # Truncate below block_size so the next promotion short-reads.
    with open(path, "wb") as f:
        f.write(b"x" * 10)
    tier.submit_load(make_job(3, [key(1)], [0], is_promotion=True))
    results = drain(tier)  # get_finished_jobs() marks the verdict False here
    assert len(results) == 1 and not results[0].success

    # Corrupt file removed; the SAME request now misses from cache, no re-probe.
    assert not os.path.exists(path)
    lm = tier._lookup_manager
    assert tier.lookup(key(1), ctx) == LookupResult.MISS
    assert lm._lookup_batch == []

    # A FRESH request re-probes the tier (no cached verdict) and misses too,
    # since the corrupt file is gone -- the real batch_lookup re-probe path.
    fresh = ReqContext(req_id="fresh-after-short-read")
    assert lookup_and_wait(tier, [key(1)], ctx=fresh) == [LookupResult.MISS]


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_batched_partial_load_failure_keeps_loaded_blocks(
    fs_tier, monkeypatch, use_c_ext
):
    """A batched promotion stops at the first bad block and reports how many
    loaded before it (#50321). Corrupt the LAST block: the earlier blocks load
    fine, so the job reports successful_keys for them and marks only the failed
    tail a miss. The earlier keys stay HIT — including for the same request —
    while the corrupt block stays a MISS (its file was removed)."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, _ = fs_tier
    keys = [key(1), key(2), key(3)]  # last one is the "bad" block
    tier.submit_store(make_job(1, keys, [0, 1, 2]))
    assert all(r.success for r in drain(tier))
    bad_path = tier.file_mapper.get_file_name(key(3))

    ctx = ReqContext(req_id="batch-req")
    assert lookup_and_wait(tier, keys, ctx=ctx) == [LookupResult.HIT] * 3

    # Corrupt only the last block, then load the whole batch as one job.
    with open(bad_path, "wb") as f:
        f.write(b"x" * 10)
    tier.submit_load(make_job(2, keys, [0, 1, 2], is_promotion=True))
    results = drain(tier)
    # (a) the job fails but reports the two blocks that loaded before the bad one.
    assert len(results) == 1 and not results[0].success
    assert tuple(results[0].successful_keys) == (key(1), key(2))

    # (b) Only the failed tail is a miss; the loaded blocks stay HIT on the same
    # request, and nothing was re-probed.
    lm = tier._lookup_manager
    assert [tier.lookup(k, ctx) for k in keys] == [
        LookupResult.HIT,
        LookupResult.HIT,
        LookupResult.MISS,
    ]
    assert lm._lookup_batch == []

    # (c) A fresh request re-probes: the loaded blocks are still on disk (HIT),
    # only the corrupt block was removed (MISS).
    tier.on_request_finished(ctx)
    fresh = ReqContext(req_id="fresh-batch-req")
    assert lookup_and_wait(tier, keys, ctx=fresh) == [
        LookupResult.HIT,
        LookupResult.HIT,
        LookupResult.MISS,
    ]


def test_batched_load_first_block_fails_only_that_block_is_a_miss(fs_tier):
    """Blocks within a batch are loaded independently (work-stealing across
    threads), so a failure in the FIRST block does not poison the rest of
    the batch: the other blocks are attempted and reported as successful
    regardless of position, unlike a strictly sequential "stop at first
    failure" scheme."""
    tier, _ = fs_tier
    keys = [key(1), key(2), key(3)]  # first one is the "bad" block
    tier.submit_store(make_job(1, keys, [0, 1, 2]))
    assert all(r.success for r in drain(tier))

    ctx = ReqContext(req_id="batch-first-fail")
    assert lookup_and_wait(tier, keys, ctx=ctx) == [LookupResult.HIT] * 3

    with open(tier.file_mapper.get_file_name(key(1)), "wb") as f:
        f.write(b"x" * 10)
    tier.submit_load(make_job(2, keys, [0, 1, 2], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1 and not results[0].success
    # key(2) and key(3) loaded fine independently of key(1)'s failure.
    assert set(results[0].successful_keys) == {key(2), key(3)}
    # Only the specifically-failed block is a miss for this request.
    assert [tier.lookup(k, ctx) for k in keys] == [
        LookupResult.MISS,
        LookupResult.HIT,
        LookupResult.HIT,
    ]


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_transient_load_failure_leaves_file(fs_tier, monkeypatch, use_c_ext):
    """A transient host error (here ELOOP on open) is NOT a short read: the job
    fails but the block file must survive untouched, on both the C and Python
    paths. Deleting on a transient error would turn a passing hiccup into
    permanent data loss."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))
    path = tier.file_mapper.get_file_name(key(1))
    with open(path, "rb") as f:
        original = f.read()

    # Make open() fail with ELOOP (fd < 0) without truncating the block. Not
    # chmod 000: CI runs as root, which bypasses permission bits, so open()
    # would succeed and the load would not fail at all.
    saved = path + ".saved"
    loop = path + ".loop"
    os.rename(path, saved)
    os.symlink(loop, path)
    os.symlink(path, loop)

    tier.submit_load(make_job(2, [key(1)], [0], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1 and not results[0].success

    # The path is left alone: a non-short-read error must not unlink.
    assert os.path.lexists(path)

    os.unlink(path)
    os.unlink(loop)
    os.rename(saved, path)
    with open(path, "rb") as f:
        assert f.read() == original


# ---------------------------------------------------------------------------
# KV events
# ---------------------------------------------------------------------------


def test_successful_store_emits_stored_event(fs_tier_with_events):
    """A completed store job emits one stored event with the job's keys."""
    tier = fs_tier_with_events
    keys = [key(1), key(2)]
    tier.submit_store(make_job(1, keys, [0, 1]))
    assert all(r.success for r in drain(tier))

    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].keys == keys
    assert events[0].medium == Medium.STORAGE
    assert events[0].locality is Locality.LOCAL
    assert not events[0].removed
    # take_events drains the buffer.
    assert list(tier.take_events()) == []


@pytest.mark.parametrize(
    ("locality", "expected"),
    [(None, None), ("REMOTE", Locality.REMOTE)],
)
def test_store_event_uses_configured_locality(tmp_path, locality, expected):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    locality_config = {} if locality is None else {"locality": locality}
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
        **locality_config,
    )
    try:
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert all(r.success for r in drain(tier))

        events = list(tier.take_events())
        assert len(events) == 1
        assert events[0].locality is expected
    finally:
        tier.shutdown()


def test_load_job_emits_no_event(fs_tier_with_events):
    tier = fs_tier_with_events
    tier.submit_store(make_job(1, [key(1)], [0]))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    list(tier.take_events())

    tier.submit_load(make_job(2, [key(1)], [1], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert list(tier.take_events()) == []


def test_mixed_job_results_emit_event_only_for_successful_job(fs_tier_with_events):
    """With a failed and a successful store job in flight, exactly one event
    is emitted and its keys belong to the successful job."""
    tier = fs_tier_with_events
    # An out-of-bounds block id fails deterministically (see
    # test_out_of_bounds_block_id_smoke) without depending on any
    # implementation detail of how the store itself fails.
    out_of_bounds_bid = _NUM_BLOCKS

    tier.submit_store(make_job(1, [key(1)], [out_of_bounds_bid]))
    tier.submit_store(make_job(2, [key(2)], [1]))
    results = drain(tier)
    assert len(results) == 2
    by_id = {r.job_id: r for r in results}
    assert not by_id[1].success
    assert by_id[2].success

    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].keys == [key(2)]


def test_partially_failed_store_emits_no_event(fs_tier_with_events):
    """A store job with any failed block emits no event for the whole job."""
    tier = fs_tier_with_events
    out_of_bounds_bid = _NUM_BLOCKS

    tier.submit_store(make_job(1, [key(1), key(2)], [0, out_of_bounds_bid]))
    results = drain(tier)
    assert len(results) == 1
    assert not results[0].success
    assert list(tier.take_events()) == []
    assert tier._store_job_keys == {}


def test_events_disabled_by_default(fs_tier):
    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert tier.events is None
    assert tier._store_job_keys == {}
    assert list(tier.take_events()) == []


def test_events_require_global_kv_events_flag(tmp_path):
    """Tier-level opt-in alone is not enough; the global flag gates events."""
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=False),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
    )
    try:
        assert tier.events is None
        tier.submit_store(make_job(1, [key(1)], [0]))
        results = drain(tier)
        assert len(results) == 1
        assert results[0].success
        assert list(tier.take_events()) == []
        assert tier._store_job_keys == {}
    finally:
        tier.shutdown()


def test_cascade_store_emits_fs_event_through_tiering_manager(tmp_path):
    """A GPU->CPU->fs cascade surfaces the tier-owned FS stored event via the
    TieringOffloadingManager's aggregated take_events()."""
    from vllm.v1.kv_offload.tiering.manager import (
        CPUPrimaryTierOffloadingManager,
        TieringOffloadingManager,
    )

    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    view = memoryview(tensor.numpy())
    mock_region = MagicMock()
    mock_region.create_kv_memoryview.return_value = view
    primary = CPUPrimaryTierOffloadingManager(num_blocks=4, mmap_region=mock_region)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=primary.get_kv_memoryview(),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
    )
    manager = TieringOffloadingManager(primary_tier=primary, secondary_tiers=[tier])
    try:
        keys = [key(1), key(2)]
        manager.on_new_request(_CTX)
        assert manager.prepare_store(keys, _CTX) is not None
        manager.complete_store(keys, _CTX)  # cascades to the fs tier

        events: list[OffloadingEvent] = []
        ctx = ScheduleEndContext(new_req_ids=[], preempted_req_ids=())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not events:
            manager.on_schedule_end(ctx)
            events.extend(manager.take_events())
            time.sleep(0.01)

        fs_events = [e for e in events if e.medium == Medium.STORAGE]
        assert len(fs_events) == 1
        assert set(fs_events[0].keys) == set(keys)
        assert not fs_events[0].removed
    finally:
        tier.shutdown()


def test_fs_tier_cross_tp_round_trip(tmp_path):
    """TP=2 replicated writer and TP=4 reader share namespace and bytes."""
    root = str(tmp_path)
    writer_tensor = _page_aligned_rand_tensor(4, _BLOCK_ELEMENTS)
    expected = writer_tensor[0].clone()
    writer = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(
            tp_size=2, world_size=2, rank=0, replicated_layout=True
        ),
        primary_kv_view=memoryview(writer_tensor.numpy()),
        tier_type="fs",
        root_dir=root,
        n_read_threads=2,
        n_write_threads=2,
    )
    try:
        writer.submit_store(make_job(1, [key(7)], [0]))
        assert all(r.success for r in drain(writer))
        writer_base = writer.file_mapper.base_path
        writer_path = writer.file_mapper.get_file_name(key(7))
    finally:
        writer.shutdown()

    reader_tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    reader = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(
            tp_size=4, world_size=4, rank=3, replicated_layout=True
        ),
        primary_kv_view=memoryview(reader_tensor.numpy()),
        tier_type="fs",
        root_dir=root,
        n_read_threads=2,
        n_write_threads=2,
    )
    try:
        assert reader.file_mapper.base_path == writer_base
        assert reader.file_mapper.get_file_name(key(7)) == writer_path
        assert lookup_and_wait(reader, [key(7)]) == [LookupResult.HIT]
        reader.submit_load(make_job(2, [key(7)], [1], is_promotion=True))
        assert all(r.success for r in drain(reader))
        assert torch.allclose(reader_tensor[1], expected)
    finally:
        reader.shutdown()


# ---------------------------------------------------------------------------
# Work-stealing pool concurrency stress tests
#
# The queues, work-stealing, and per-block I/O all moved into the fs_io_C
# C++ pool (see csrc/fs_io.cpp). Threading races there are inherently
# intermittent, so these tests use many threads/jobs/blocks and, where
# relevant, repeat across several seeds rather than relying on a single run.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
def test_stress_concurrent_store_load_data_integrity(tmp_path, seed):
    """Round-trip many blocks through many small jobs and many work-stealing
    threads; every byte must match exactly. A race in the C++ pop/steal/push
    paths would show up here as silent data corruption."""
    rng = random.Random(seed)
    n_blocks = 300
    tensor = _page_aligned_zero_tensor(n_blocks, _STRESS_BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=16,
        n_write_threads=16,
    )
    try:
        for bid in range(n_blocks):
            tensor[bid].fill_(float(bid + 1))
        expected = tensor.clone()

        keys = [key(1000 + i) for i in range(n_blocks)]
        block_ids = list(range(n_blocks))
        rng.shuffle(block_ids)

        # Split into many randomly-sized jobs (1-5 blocks each).
        job_specs: list[tuple[int, list[int]]] = []
        i = 0
        job_id = 0
        while i < n_blocks:
            chunk = block_ids[i : i + rng.randint(1, 5)]
            job_specs.append((job_id, chunk))
            i += len(chunk)
            job_id += 1

        for jid, chunk in job_specs:
            tier.submit_store(make_job(jid, [keys[b] for b in chunk], chunk))
        store_results = drain(tier)
        assert len(store_results) == len(job_specs)
        assert all(r.success for r in store_results)

        tensor.zero_()

        # Load each key's block into a permuted (disjoint-from-original-in-
        # general) destination id, to also exercise loading into a
        # different id than it was stored from.
        perm = list(range(n_blocks))
        rng.shuffle(perm)
        for jid, chunk in job_specs:
            tier.submit_load(
                make_job(
                    jid + 1_000_000,
                    [keys[b] for b in chunk],
                    [perm[b] for b in chunk],
                    is_promotion=True,
                )
            )
        load_results = drain(tier)
        assert len(load_results) == len(job_specs)
        assert all(r.success for r in load_results)

        for b in range(n_blocks):
            assert torch.equal(tensor[perm[b]], expected[b]), (
                f"block for key {b} corrupted after store+load round trip"
            )
    finally:
        tier.shutdown()


def test_stress_interleaved_same_key_store_load(tmp_path):
    """Rapidly alternate store/load jobs over the same overlapping keys,
    processed by many work-stealing threads, exercising the "already
    exists" short-circuit in _store_block under real concurrency.

    The store and load queues are independent, so nothing orders a store
    job before a same-round load job beyond submission order; every key is
    seeded once up front (and never deleted) so every load in the
    interleaved loop is guaranteed to target an existing file regardless of
    how the two queues race against each other.
    """
    n_blocks = 8
    tensor = _page_aligned_rand_tensor(n_blocks, _STRESS_BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=16,
        n_write_threads=16,
    )
    try:
        keys = [key(3000 + i) for i in range(4)]
        tier.submit_store(make_job(0, keys, [0, 1, 2, 3]))
        assert all(r.success for r in drain(tier))

        job_id = 1
        for _ in range(40):
            # Re-store of already-existing files: hits the short-circuit.
            tier.submit_store(make_job(job_id, keys, [0, 1, 2, 3]))
            job_id += 1
            tier.submit_load(make_job(job_id, keys, [4, 5, 6, 7], is_promotion=True))
            job_id += 1
        results = drain(tier)
        assert len(results) == 80
        assert all(r.success for r in results)
    finally:
        tier.shutdown()


@pytest.mark.parametrize("trial", range(5))
def test_stress_partial_failure_attribution_exact(tmp_path, trial):
    """Inject failures on a random subset of blocks within a large batch
    processed by many work-stealing threads; the reported successful_keys
    must exactly equal the non-corrupted keys every time. This directly
    targets the out-of-order-completion correctness issue that made the old
    num_succeeded-prefix model wrong under parallel execution."""
    rng = random.Random(trial)
    n_blocks = 64
    tensor = _page_aligned_zero_tensor(n_blocks, _STRESS_BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=16,
        n_write_threads=16,
    )
    try:
        keys = [key(4000 + i) for i in range(n_blocks)]
        tier.submit_store(make_job(1, keys, list(range(n_blocks))))
        assert all(r.success for r in drain(tier))

        n_failures = rng.randint(1, n_blocks - 1)
        failed_idx = set(rng.sample(range(n_blocks), n_failures))
        for i in failed_idx:
            path = tier.file_mapper.get_file_name(keys[i])
            with open(path, "wb") as f:
                f.write(b"x" * 4)  # below block_size -> short read

        tier.submit_load(make_job(2, keys, list(range(n_blocks)), is_promotion=True))
        results = drain(tier)
        assert len(results) == 1
        assert not results[0].success

        expected_successful = {keys[i] for i in range(n_blocks) if i not in failed_idx}
        assert set(results[0].successful_keys or ()) == expected_successful
    finally:
        tier.shutdown()


def test_stress_shutdown_under_load(tmp_path):
    """shutdown() while hundreds of jobs are being raced over by many
    work-stealing worker threads must not crash, hang, or deadlock, and
    join() must return promptly."""
    n_blocks = 4
    tensor = _page_aligned_zero_tensor(n_blocks, _STRESS_BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=16,
        n_write_threads=16,
    )
    keys = [key(5000 + i) for i in range(n_blocks)]
    for i in range(300):
        tier.submit_store(make_job(i, keys, list(range(n_blocks))))

    start = time.monotonic()
    tier.shutdown()
    elapsed = time.monotonic() - start
    assert elapsed < 10.0, f"shutdown() took {elapsed:.2f}s"
    assert all(not t.is_alive() for t in tier._pool._threads)


def test_stress_wait_idle_liveness_under_concurrent_enqueue(tmp_path):
    """wait_idle() must account for every job submitted by several
    concurrent driver threads before returning -- no job may be lost or
    left unaccounted for under concurrent enqueue."""
    block_size = 64
    buf = bytearray(block_size)
    pool = DualQueueThreadPool(
        n_read_threads=4,
        n_write_threads=4,
        n_write_excl_threads=2,
        primary_kv_view=memoryview(buf),
        block_size=block_size,
    )
    n_drivers = 4
    jobs_per_driver = 50

    def driver(driver_id: int) -> None:
        for i in range(jobs_per_driver):
            job_id = driver_id * 100_000 + i
            path = str(tmp_path / f"d{driver_id}_{i}.bin")
            pool.enqueue_store(job_id=job_id, paths=[path], offsets=[0])

    waiter_done = threading.Event()

    def waiter() -> None:
        pool.wait_idle()
        waiter_done.set()

    try:
        drivers = [threading.Thread(target=driver, args=(d,)) for d in range(n_drivers)]
        w = threading.Thread(target=waiter)
        for t in drivers:
            t.start()
        w.start()
        for t in drivers:
            t.join(timeout=10.0)
        w.join(timeout=10.0)
        assert waiter_done.is_set(), "wait_idle() did not return"

        finished = pool.get_finished()
        assert len(finished) == n_drivers * jobs_per_driver
        assert all(success for _, success, _, _ in finished)
    finally:
        pool.shutdown(wait=True)


def test_stress_pool_lifecycle_churn():
    """Repeated create_pool/destroy_pool cycles must not crash or
    double-free -- simulating repeated secondary-tier add/remove."""
    fs_io_C = pytest.importorskip("vllm.fs_io_C")

    block_size = 32
    for _ in range(200):
        buf = bytearray(block_size * 4)
        pool = fs_io_C.create_pool(memoryview(buf), block_size, False)
        fs_io_C.push_store(pool, 1, [], [])  # touch the queue mutex too
        fs_io_C.destroy_pool(pool)
        del pool, buf
        gc.collect()
