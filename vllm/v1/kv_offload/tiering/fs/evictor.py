# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Evcitor process for fs tier. Monitors the fs storage directory and evicts
old files when storage thresholds are hit.
"""

import ctypes
import fcntl
import heapq
import json
import math
import multiprocessing
import os
import signal
import threading
import time
import uuid
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from pathlib import Path

from vllm.logger import init_logger

ACCESS_TIME_THRESHOLD_S = 30
DISCOVERY_TIMEOUT_S = 60  # 5 minutes

logger = init_logger(__name__)


@dataclass
class DiskUsage:
    total_bytes: int
    used_bytes: int
    available_bytes: int
    usage_percent: float


@dataclass(frozen=True)
class EvictorRuntimeConfig:
    storage_size: int | None
    storage_threshold_pct: float | None
    access_time_threshold_s: int = ACCESS_TIME_THRESHOLD_S
    max_delete_queue_size: int = 1000
    delete_check_interval_s: int = DISCOVERY_TIMEOUT_S

    @staticmethod
    def from_json_str(json_str: str) -> "EvictorRuntimeConfig":
        data = json.loads(json_str)
        config = EvictorRuntimeConfig(
            storage_size=data.get("storage_size", None),
            storage_threshold_pct=data.get("storage_threshold_pct", None),
            access_time_threshold_s=data.get(
                "access_time_threshold_s", ACCESS_TIME_THRESHOLD_S
            ),
            max_delete_queue_size=data.get("max_delete_queue_size", 1000),
            delete_check_interval_s=data.get(
                "delete_check_interval_s", DISCOVERY_TIMEOUT_S
            ),
        )

        assert config.storage_size is None or config.storage_threshold_pct is None
        return config


def get_disk_usage_from_statvfs(root_dir: str) -> DiskUsage | None:
    """
    Get disk usage using statvfs() - O(1) operation, critical for multi-TB volumes.

    Trade-off: statvfs() provides instant disk usage statistics but is less accurate
    than `du` which would be O(n) and could take hours on large volumes.
    """
    try:
        stat = os.statvfs(root_dir)
        block_size = stat.f_frsize
        total_blocks = stat.f_blocks
        free_blocks = stat.f_bfree

        total_bytes = total_blocks * block_size
        free_bytes = free_blocks * block_size
        used_bytes = total_bytes - free_bytes
        usage_percent = (used_bytes / total_bytes) * 100 if total_bytes > 0 else 0

        return DiskUsage(
            total_bytes=total_bytes,
            used_bytes=used_bytes,
            available_bytes=free_bytes,
            usage_percent=usage_percent,
        )
    except Exception as e:
        logger.error("Cannot fetch disk usage. Failed with %s", e)
        return None


def safe_scandir(path: str) -> Iterator[os.DirEntry]:
    """
    Safely scan a directory, handling filesystem errors.

    Returns an iterator of directory entries, or empty iterator on error.
    This reduces exception handling duplication while maintaining streaming
    behavior.
    """
    try:
        return os.scandir(path)
    except (OSError, PermissionError):
        return iter([])


def safe_remove(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except Exception as e:
        logger.error("Failed for remove path %s : %s", path, e)
    return False


def safe_atime(file_path: Path) -> float | None:
    # Check file access time - skip recently accessed files
    # Note: relatime filesystem may not update atime on every access
    # This can cause false positives (deleting "hot" files)
    try:
        file_stat = file_path.stat()
        return file_stat.st_atime  # Last access time
    except (OSError, AttributeError) as e:
        logger.error("Cannot fetch file atime : %s", e)
    return None


def safe_is_evictor_alive(evictor_path: str) -> bool:
    fd: int | None = None
    try:
        fd = os.open(evictor_path, os.O_RDONLY)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    except BlockingIOError:
        # Lock still held by some Evictor process.
        return True
    except Exception as e:
        # Cannot determine liveness. Consider evictor failed so as not
        # to risk having untracked hash files.
        logger.warning("Cannot determine evictor liveness %s : %s", evictor_path, e)
        return False
    finally:
        if fd is not None:
            os.close(fd)

    # Could successfully acquire and release lock.
    return False


def register_evictor_process(root_dir: str) -> tuple[str | None, int | None]:
    if not os.path.exists(root_dir):
        raise ValueError(f"{root_dir} does not exist.")

    # Try a arbitrary number of times (4 here) to avoid the rare case
    # where multiple evictors generating the same id.
    for _ in range(4):
        process_file_path = f"{root_dir}/__{uuid.uuid4()}.evictor"
        try:
            fd = os.open(
                process_file_path,
                os.O_CREAT | os.O_EXCL,
                0o644,
            )
            # lock file. The lock is what other evictor processes check for liveness.
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return process_file_path, fd
        except Exception as e:
            logger.error("Evictor registration error %s", e)

    # Failure
    return None, None


def unregister_evictor_process(process_file_path: str, fd: int):
    try:
        # unlock file
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        os.remove(process_file_path)
    except Exception as e:
        logger.error("Cannot unregister evictor process %s : %s", process_file_path, e)
        return

    logger.info("Unregistered evictor process %s", process_file_path)
    assert not os.path.exists(process_file_path)


class EvictorProcess(multiprocessing.Process):
    def __init__(
        self,
        root_dir: str,
        runtime_config: EvictorRuntimeConfig,
    ):
        super().__init__(name="Evictor")
        self.root_dir = root_dir
        self.runtime_config = runtime_config

        # disk usage thread for storage_size spawns
        self.du_thread: threading.Thread | None = None
        self.storage_alloc_size_lock = threading.Lock()
        self.storage_alloc_size = 0

        # delete file heapq. list of (access_time * -1, file_path)
        self.file_heapq: list[tuple[float, str]] = []

        # global evictor status
        self.my_evictor_path: str | None = None
        self.my_evictor_path_fd: int | None = None
        self.num_evictors: int = 0
        self.my_evictor_id: int | None = None

    def get_true_disk_usage_thread(self):
        while self.running:
            running_total_size_bytes = 0
            stack: list[Path] = [Path(self.root_dir)]
            while stack:
                entry = stack.pop()
                for sub_entry in safe_scandir(str(entry)):
                    if sub_entry.is_dir(follow_symlinks=False):
                        stack.append(Path(sub_entry.path))
                    elif sub_entry.is_file(follow_symlinks=False):
                        running_total_size_bytes += Path(sub_entry.path).stat().st_size
            with self.storage_alloc_size_lock:
                self.storage_alloc_size = running_total_size_bytes
            # wait before next poll
            time.sleep(2)

    def signal_handler(self, signum, frame):
        self.running = False
        self.shutdown()

    def shutdown(self):
        logger.info("Shutdown evictor ...")
        self.running = False
        if self.du_thread is not None and self.du_thread.is_alive():
            # Wait for the du thread to complete
            self.du_thread.join()
        if self.my_evictor_path is not None:
            assert self.my_evictor_path_fd is not None
            unregister_evictor_process(self.my_evictor_path, self.my_evictor_path_fd)

    def should_delete(self) -> bool:
        """
        True if deletion should be triggered. False otherwise.
        """
        if self.runtime_config.storage_size is not None:
            assert self.du_thread is not None
            assert self.du_thread.is_alive()
            with self.storage_alloc_size_lock:
                return self.runtime_config.storage_size < self.storage_alloc_size
        else:
            assert self.runtime_config.storage_threshold_pct is not None
            disk_usage = get_disk_usage_from_statvfs(self.root_dir)
            if disk_usage is None:
                return False
            return disk_usage.usage_percent > self.runtime_config.storage_threshold_pct

    def maybe_delete(self):
        if self.should_delete():
            logger.debug("Delete %d files ...", len(self.file_heapq))
            while self.file_heapq:
                _, file_path = self.file_heapq.pop()
                safe_remove(file_path)

    def discover_evictors(self) -> tuple[int, int]:
        """
        Returns num live evictors and the id of my evictor.
        """
        assert self.my_evictor_path is not None
        assert os.path.exists(self.my_evictor_path), "My Evictor was unregistered"

        def is_evictor_file(entry: os.DirEntry):
            return (
                entry.is_file(follow_symlinks=False)
                and entry.name.startswith("__")
                and entry.name.endswith(".evictor")
            )

        def is_evictor_alive(evictor_path: str) -> bool:
            if evictor_path == self.my_evictor_path:
                return True
            return safe_is_evictor_alive(evictor_path)

        evictors: list[str] = []
        for entry in safe_scandir(self.root_dir):
            if is_evictor_file(entry):
                evictors.append(entry.path)

        dead_evictors = []
        live_evictors = []
        for evictor in evictors:
            if is_evictor_alive(evictor):
                live_evictors.append(evictor)
            else:
                dead_evictors.append(evictor)

        for evictor in dead_evictors:
            safe_remove(evictor)
        sorted_live = sorted(live_evictors)
        if self.my_evictor_path not in sorted_live:
            raise RuntimeError(
                f"Current evictor file missing from live set: {self.my_evictor_path}"
            )
        return len(sorted_live), sorted_live.index(self.my_evictor_path)

    def get_my_hex_ranges(self) -> tuple[int, int]:
        assert self.num_evictors is not None
        assert self.my_evictor_id is not None
        # Always have power of 2 evictor processes doing work with
        # a maximum of 16 evictors.
        evictor_limit = min(16, 2 ** int(math.log2(self.num_evictors)))
        if self.my_evictor_id > evictor_limit - 1:
            return (-1, -1)

        num_hex_per_process = 16 // evictor_limit
        # split 16 hashes evenly among evictors
        hex_range_start = num_hex_per_process * self.my_evictor_id
        hex_range_end = hex_range_start + num_hex_per_process - 1
        return (hex_range_start, hex_range_end)

    def crawler(self, min_hex: int, max_hex: int, timeout_s: int):
        assert max_hex >= min_hex
        yield
        logger.debug(
            "Crawler triggered at %s  for hex range %s, %s",
            self.root_dir,
            min_hex,
            max_hex,
        )

        def hex_to_int(hex_str: str) -> int | None:
            """Convert hex string to integer."""
            try:
                return int(hex_str, 16)
            except (ValueError, TypeError):
                return None

        def _record_dir(hex3_dir: os.DirEntry, min_hex: int, max_hex: int) -> bool:
            # Apply hex modulo filtering for load balancing across crawlers.
            hex_int = hex_to_int(hex3_dir.name)
            if hex_int is None:
                return False
            hex_mod = hex_int % 16
            return min_hex <= hex_mod <= max_hex

        def _cleanup():
            while self.file_heapq:
                self.file_heapq.pop()

        def _safe_yield_dir(path: str) -> Iterator[os.DirEntry]:
            for entry in safe_scandir(path):
                if entry.is_dir(follow_symlinks=False):
                    yield entry

        def _yield_bin_files() -> Iterator[os.DirEntry]:
            # Layout: <root_dir>/<safe_model_name>_<sha256-prefix>/<hhh>/<hh>_g<group_idx>/<hash>.bin # noqa: E501
            for model_dir in _safe_yield_dir(self.root_dir):
                for hex1 in _safe_yield_dir(model_dir.path):
                    if _record_dir(hex1, min_hex, max_hex):
                        for hex2 in _safe_yield_dir(hex1.path):
                            for bin_file in safe_scandir(hex2.path):
                                if bin_file.is_file(
                                    follow_symlinks=False
                                ) and bin_file.path.endswith(".bin"):
                                    yield bin_file

        deadline = time.monotonic() + timeout_s

        def _maybe_yield() -> Generator[None, bool | None, bool | None]:
            nonlocal deadline
            if time.monotonic() < deadline:
                return True
            should_continue = yield
            if should_continue:
                deadline = time.monotonic() + timeout_s
            return should_continue

        if min_hex == -1 and max_hex == -1:
            # dummy crawler code path code path code path code path
            while True:
                time.sleep(timeout_s)
                # Check timeout and yield
                should_continue = yield
                if not should_continue:
                    _cleanup()
                    return

        while True:
            # Check timeout and yield
            should_continue = yield from _maybe_yield()
            if not should_continue:
                _cleanup()
                return

            for bin_file in _yield_bin_files():
                if not self.running:
                    return

                # Check timeout and yield
                should_continue = yield from _maybe_yield()
                if not should_continue:
                    _cleanup()
                    return

                access_time = safe_atime(Path(bin_file.path))
                if access_time is None:
                    continue
                if (
                    time.time() - access_time
                    < self.runtime_config.access_time_threshold_s
                ):
                    # skip
                    continue
                if len(self.file_heapq) == self.runtime_config.max_delete_queue_size:
                    # slow down
                    time.sleep(0.1)
                    if access_time < -self.file_heapq[0][0]:
                        heapq.heappushpop(
                            self.file_heapq, (-access_time, bin_file.path)
                        )
                else:
                    # by default heapq is does min-heap. We want access times
                    # that are larger to appear in front so we can swap them.
                    heapq.heappush(self.file_heapq, (-access_time, bin_file.path))

    def run(self):
        # handle signals and exit gracefully
        signal.signal(signal.SIGTERM, self.signal_handler)
        # register termination when parent dies
        libc = ctypes.CDLL(None)
        libc.prctl(1, 15)  # sigterm when parent dies

        self.my_evictor_path, self.my_evictor_path_fd = register_evictor_process(
            self.root_dir
        )
        if self.my_evictor_path is None:
            logger.error("Cannot register evictor. Terminating...")
            self.shutdown()
            return

        self.num_evictors, self.my_evictor_id = self.discover_evictors()
        assert self.num_evictors > 0
        min_hex, max_hex = self.get_my_hex_ranges()

        self.running = True
        # start the disk usage thread
        if self.runtime_config.storage_size is not None:
            self.du_thread = threading.Thread(target=self.get_true_disk_usage_thread)
            self.du_thread.start()

        while self.running:
            try:
                # trigger crawler
                crawler_p = self.crawler(
                    min_hex,
                    max_hex,
                    timeout_s=self.runtime_config.delete_check_interval_s,
                )
                crawler_p.send(None)

                while self.running:
                    # check deletes
                    self.maybe_delete()

                    # Gets here after the first timeout
                    # discover evictors
                    new_num_evictors, new_my_evictor_id = self.discover_evictors()
                    assert new_num_evictors > 0
                    if (
                        self.num_evictors == new_num_evictors
                        and self.my_evictor_id == new_my_evictor_id
                    ):
                        # business as usual
                        crawler_p.send(True)
                        continue
                    else:
                        # need reassignment
                        self.num_evictors = new_num_evictors
                        self.my_evictor_id = new_my_evictor_id
                        # update min and max hex
                        min_hex, max_hex = self.get_my_hex_ranges()
                        crawler_p.send(False)  # this will trigger stopitereation
            except StopIteration:
                logger.debug(
                    "Triggered Reassignment as new evictors discovered."
                    "new_num_evictors=%d, new_my_evictor_id=%d",
                    new_num_evictors,
                    new_my_evictor_id,
                )
            continue


class Evictor:
    def __init__(self, root_dir: str, evictor_config: str):
        """
        root_dir: root directory of the storage to monitor.
        """
        self.root_dir = root_dir
        self.evictor_config = EvictorRuntimeConfig.from_json_str(evictor_config)

        logger.info(
            "Evictor created with fields: \n"
            "   : root_dir : %s \n"
            "   : evictor_config : %s \n",
            self.root_dir,
            self.evictor_config,
        )

    def is_alive(
        self,
    ):
        return self.process.is_alive()

    def spawn_evictor(
        self,
    ):
        """
        Spawn evictor process and always monitor
        """
        self.process = EvictorProcess(
            self.root_dir,
            self.evictor_config,
        )
        self.process.start()

    def shutdown(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
