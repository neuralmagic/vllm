# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.fs.evictor import (
    EvictorProcess,
    EvictorRuntimeConfig,
    get_disk_usage_from_statvfs,
    register_evictor_process,
    safe_is_evictor_alive,
    unregister_evictor_process,
)


def _make_mock_offloading_spec() -> MagicMock:
    mock_vllm_config = MagicMock()
    mock_vllm_config.model_config.model = "test-model"
    mock_vllm_config.cache_config.block_size = 16
    mock_vllm_config.cache_config.cache_dtype = "torch.float32"
    mock_vllm_config.parallel_config.tensor_parallel_size = 1
    mock_vllm_config.parallel_config.pipeline_parallel_size = 1
    mock_vllm_config.parallel_config.prefill_context_parallel_size = 1
    mock_vllm_config.parallel_config.decode_context_parallel_size = 1
    mock_vllm_config.parallel_config.rank = 0

    mock_kv_cache_config = MagicMock()
    mock_kv_cache_config.kv_cache_groups = []

    mock_offloading_spec = MagicMock()
    mock_offloading_spec.vllm_config = mock_vllm_config
    mock_offloading_spec.kv_cache_config = mock_kv_cache_config
    mock_offloading_spec.block_size_factor = 1
    return mock_offloading_spec


def _touch_bin_files(root_dir: Path, count: int = 4) -> list[str]:
    mapper = FileMapper.from_offloading_spec(
        root_dir=str(root_dir),
        offloading_spec=_make_mock_offloading_spec(),
        gpu_blocks_per_file=1,
    )
    created: list[str] = []
    for i in range(count):
        offload_key = make_offload_key(i.to_bytes(8, "big"), 0)
        path = Path(mapper.get_file_name(offload_key))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"bin-{i}".encode())
        created.append(str(path))
    return created


def _evictor_holder(root_dir: str, run_s: float):
    process_path = None
    fd = None
    try:
        process_path, fd = register_evictor_process(root_dir)
        time.sleep(run_s)
    finally:
        if process_path is not None and fd is not None:
            unregister_evictor_process(process_path, fd)


def test_get_disk_usage_from_statvfs(tmp_path: Path):
    usage = get_disk_usage_from_statvfs(str(tmp_path))
    assert usage is not None

    stat = os.statvfs(tmp_path)
    block_size = stat.f_frsize
    expected_total = stat.f_blocks * block_size
    expected_free = stat.f_bfree * block_size
    expected_used = expected_total - expected_free

    assert usage.total_bytes == expected_total
    assert usage.available_bytes == expected_free
    assert usage.used_bytes == expected_used
    if expected_total > 0:
        assert usage.usage_percent == pytest.approx(
            (expected_used / expected_total) * 100
        )


def test_access_time_updates_after_read(tmp_path: Path):
    f = tmp_path / "atime.bin"
    f.write_bytes(b"abcdef")
    old_ts = time.time() - 7200
    os.utime(f, (old_ts, old_ts))
    before = f.stat().st_atime

    with open(f, "rb") as fp:
        assert fp.read() == b"abcdef"

    after = f.stat().st_atime
    assert after >= before
    assert after > old_ts


def test_register_evictor_filelock_effect_and_unregister_cleanup(tmp_path: Path):
    process_path, fd = register_evictor_process(str(tmp_path))
    assert process_path is not None
    assert fd is not None
    assert os.path.exists(process_path)

    assert safe_is_evictor_alive(process_path)
    unregister_evictor_process(process_path, fd)

    assert not os.path.exists(process_path)
    assert not safe_is_evictor_alive(process_path)
    with pytest.raises(OSError):
        os.fstat(fd)


def test_discover_evictors_updates_after_spawn_and_random_shutdown(tmp_path: Path):
    observer = EvictorProcess(
        root_dir=str(tmp_path),
        runtime_config=EvictorRuntimeConfig(
            storage_size=None,
            storage_threshold_pct=99.99,
            delete_check_interval_s=1,
        ),
    )
    my_path, my_fd = register_evictor_process(str(tmp_path))
    assert my_path is not None
    assert my_fd is not None

    children = []
    try:
        observer.my_evictor_path = my_path
        spawn_count = 8
        for _ in range(spawn_count):
            proc = multiprocessing.Process(
                target=_evictor_holder, args=(str(tmp_path), 15.0)
            )
            proc.start()
            children.append(proc)
            time.sleep(0.05)
            count, my_id = observer.discover_evictors()
            assert count >= 2
            assert 0 <= my_id < count
            observer.num_evictors = count
            observer.my_evictor_id = my_id
            min_hex, max_hex = observer.get_my_hex_ranges()
            if min_hex != -1:
                assert 0 <= min_hex <= max_hex <= 15
                assert (max_hex - min_hex + 1) in {1, 2, 4, 8, 16}

        random_order = list(range(len(children)))
        random.Random(7).shuffle(random_order)
        for idx in random_order:
            children[idx].terminate()
            children[idx].join(timeout=5)
            assert not children[idx].is_alive()
            # discover_evictors should clean stale .evictor files and recompute.
            count, my_id = observer.discover_evictors()
            assert count >= 1
            assert 0 <= my_id < count
    finally:
        for proc in children:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        unregister_evictor_process(my_path, my_fd)


def test_discover_evictors_with_arbitrary_spawn_delays(tmp_path: Path):
    observer = EvictorProcess(
        root_dir=str(tmp_path),
        runtime_config=EvictorRuntimeConfig(
            storage_size=None,
            storage_threshold_pct=99.99,
            delete_check_interval_s=1,
        ),
    )
    my_path, my_fd = register_evictor_process(str(tmp_path))
    assert my_path is not None
    assert my_fd is not None

    children = []
    try:
        observer.my_evictor_path = my_path
        for delay_s in [0.01, 0.2, 0.05, 0.3]:
            time.sleep(delay_s)
            proc = multiprocessing.Process(
                target=_evictor_holder, args=(str(tmp_path), 8.0)
            )
            proc.start()
            children.append(proc)
            count, my_id = observer.discover_evictors()
            observer.num_evictors = count
            observer.my_evictor_id = my_id
            min_hex, max_hex = observer.get_my_hex_ranges()
            if min_hex != -1:
                width = max_hex - min_hex + 1
                assert width in {1, 2, 4, 8, 16}
                assert min_hex % width == 0
    finally:
        for proc in children:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        unregister_evictor_process(my_path, my_fd)


def test_crawler_builds_expected_lru_delete_queue(tmp_path: Path):
    files = _touch_bin_files(tmp_path, count=3)
    now = time.time()
    # Make files have distinct access times; smaller = older.
    os.utime(files[0], (now - 300, now - 300))
    os.utime(files[1], (now - 200, now - 200))
    os.utime(files[2], (now - 100, now - 100))

    evictor = EvictorProcess(
        root_dir=str(tmp_path),
        runtime_config=EvictorRuntimeConfig(
            storage_size=None,
            storage_threshold_pct=99.99,
            access_time_threshold_s=0,
            max_delete_queue_size=2,
            delete_check_interval_s=1,
        ),
    )
    evictor.running = True
    crawler = evictor.crawler(0, 15, timeout_s=0)
    next(crawler)
    for _ in range(30):
        crawler.send(True)
        if len(evictor.file_heapq) == 2:
            break
    assert len(evictor.file_heapq) == 2
    oldest_candidates = {p for _, p in evictor.file_heapq}
    assert files[0] in oldest_candidates
    assert files[1] in oldest_candidates
    with pytest.raises(StopIteration):
        crawler.send(False)


def test_maybe_delete_removes_files_from_queue(tmp_path: Path):
    files = _touch_bin_files(tmp_path, count=2)
    evictor = EvictorProcess(
        root_dir=str(tmp_path),
        runtime_config=EvictorRuntimeConfig(
            storage_size=None,
            storage_threshold_pct=0.0,
            delete_check_interval_s=1,
        ),
    )
    for f in files:
        evictor.file_heapq.append((-time.time(), f))
    evictor.maybe_delete()
    assert all(not os.path.exists(f) for f in files)
    assert evictor.file_heapq == []


def test_crawler_parses_layout_and_ignores_non_bin_files(tmp_path: Path):
    valid_files = _touch_bin_files(tmp_path, count=2)
    non_bin = Path(valid_files[0]).with_suffix(".txt")
    non_bin.write_text("not-a-bin")
    (tmp_path / "random_dir" / "abc").mkdir(parents=True)
    (tmp_path / "random_dir" / "abc" / "other.bin").write_bytes(b"ignored")

    evictor = EvictorProcess(
        root_dir=str(tmp_path),
        runtime_config=EvictorRuntimeConfig(
            storage_size=None,
            storage_threshold_pct=99.99,
            access_time_threshold_s=0,
        ),
    )
    evictor.running = True
    crawler = evictor.crawler(0, 15, timeout_s=0)
    next(crawler)
    for _ in range(30):
        crawler.send(True)
        if len(evictor.file_heapq) >= 2:
            break
    queued_files = {p for _, p in evictor.file_heapq}
    assert set(valid_files).issubset(queued_files)
    assert str(non_bin) not in queued_files
    assert str(tmp_path / "random_dir" / "abc" / "other.bin") not in queued_files
    with pytest.raises(StopIteration):
        crawler.send(False)


def test_empty_or_no_bin_directory_still_runs_discovery_and_delete_paths(
    tmp_path: Path,
):
    evictor = EvictorProcess(
        root_dir=str(tmp_path),
        runtime_config=EvictorRuntimeConfig(
            storage_size=None,
            storage_threshold_pct=99.99,
            access_time_threshold_s=0,
        ),
    )
    my_path, my_fd = register_evictor_process(str(tmp_path))
    assert my_path is not None
    assert my_fd is not None
    evictor.my_evictor_path = my_path
    try:
        count, my_id = evictor.discover_evictors()
        assert count == 1
        assert my_id == 0

        evictor.running = True
        crawler = evictor.crawler(0, 15, timeout_s=0)
        next(crawler)
        for _ in range(8):
            crawler.send(True)
        assert evictor.file_heapq == []
        # no-op delete path should not crash on empty queue.
        evictor.maybe_delete()
        with pytest.raises(StopIteration):
            crawler.send(False)
    finally:
        unregister_evictor_process(my_path, my_fd)


def test_evictor_process_sigterm_cleans_registration(tmp_path: Path):
    process = EvictorProcess(
        root_dir=str(tmp_path),
        runtime_config=EvictorRuntimeConfig(
            storage_size=None,
            storage_threshold_pct=99.99,
            access_time_threshold_s=0,
            delete_check_interval_s=1,
        ),
    )
    process.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            evictor_files = list(tmp_path.glob("*.evictor"))
            if evictor_files:
                break
            time.sleep(0.05)

        os.kill(process.pid, signal.SIGTERM)
        process.join(timeout=10)
        assert not process.is_alive()

        # process shutdown should unregister itself.
        deadline = time.time() + 3
        while time.time() < deadline and list(tmp_path.glob("*.evictor")):
            time.sleep(0.05)
        assert list(tmp_path.glob("*.evictor")) == []
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def test_parent_death_kills_evictor_child(tmp_path: Path):
    child_pid_path = tmp_path / "child.pid"
    script = f"""
import time
from pathlib import Path
from vllm.v1.kv_offload.tiering.fs.evictor import EvictorProcess, EvictorRuntimeConfig

root = Path({str(tmp_path)!r})
child_pid_path = root / "child.pid"
proc = EvictorProcess(
    root_dir=str(root),
    runtime_config=EvictorRuntimeConfig(
        storage_size=None,
        storage_threshold_pct=99.99,
        delete_check_interval_s=1,
    ),
)
proc.start()
child_pid_path.write_text(str(proc.pid))
while True:
    time.sleep(1)
"""
    parent = subprocess.Popen([sys.executable, "-c", script])
    try:
        deadline = time.time() + 8
        child_pid = None
        while time.time() < deadline:
            if child_pid_path.exists():
                child_pid = int(child_pid_path.read_text().strip())
                break
            time.sleep(0.05)
        assert child_pid is not None
        assert os.path.exists(f"/proc/{child_pid}")
        time.sleep(2)

        parent.kill()
        parent.wait(timeout=5)

        deadline = time.time() + 8
        while time.time() < deadline:
            if not os.path.exists(f"/proc/{child_pid}"):
                break
            time.sleep(0.1)
        assert not os.path.exists(f"/proc/{child_pid}")
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
