# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Evcitor process for fs tier. Monitors the fs storage directory and evicts
old files when storage thresholds are hit.
"""

import ctypes
import enum
import fcntl
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
from vllm.utils.system_utils import decorate_logs, get_mp_context, set_process_title
from vllm.v1.kv_offload.tiering.fs.evictor_utils import (
    EvictorQueues,
    get_directory_size,
    get_disk_usage_percent_form_statvfs,
    hex_to_int,
    safe_atime,
    safe_remove,
    safe_scandir,
    safe_yield_dir,
)

DEFAULT_EVICTION_THRESHOLD = 0.95
DEFAULT_ACCESS_TIME_THRESHOLD_S = 300  # 5 minutes
DEFAULT_DELETE_CHECK_INTERVAL_S = 30  # 30s
DEFAULT_MAX_DELETE_QUEUE_SIZE = 1000

logger = init_logger(__name__)


class StorageType(enum.Enum):
    # eviction threshold applied against the volume
    Volume = 0
    # eviction threshold applied against some directory in a volume.
    Directory = 1

    @staticmethod
    def from_str(type_str: str):
        if type_str.lower() == "volume":
            return StorageType.Volume
        elif type_str.lower() == "directory":
            return StorageType.Directory
        raise ValueError(f"Unrecognized storage type {type_str}")


@dataclass(frozen=True)
class EvictorRuntimeConfig:
    storage_type: StorageType
    max_directory_size_gb: int | None = None
    eviction_threshold: float = DEFAULT_EVICTION_THRESHOLD
    access_time_threshold_s: int = DEFAULT_ACCESS_TIME_THRESHOLD_S
    max_delete_queue_size: int = DEFAULT_MAX_DELETE_QUEUE_SIZE
    delete_check_interval_s: int = DEFAULT_DELETE_CHECK_INTERVAL_S

    @property
    def max_directory_size_bytes(self):
        assert self.max_directory_size_gb is not None
        return self.max_directory_size_gb * (1024**3)

    @staticmethod
    def from_json_str(json_str: str) -> "EvictorRuntimeConfig":
        data = json.loads(json_str)

        def _type_or_none(key: str, type_fn):
            raw = data.get(key, None)
            return type_fn(raw) if raw is not None else raw

        def _type_or_default(key: str, type_fn, default):
            raw = data.get(key, None)
            return type_fn(raw) if raw is not None else default

        raw_storage_type = data.get("storage_type")
        assert raw_storage_type is not None, "Must specify a storage type"
        storage_type = StorageType.from_str(raw_storage_type)

        config = EvictorRuntimeConfig(
            storage_type=storage_type,
            eviction_threshold=_type_or_default(
                "eviction_threshold", float, DEFAULT_EVICTION_THRESHOLD
            ),
            max_directory_size_gb=_type_or_none("max_directory_size_gb", int),
            access_time_threshold_s=_type_or_default(
                "access_time_threshold_s", int, DEFAULT_ACCESS_TIME_THRESHOLD_S
            ),
            max_delete_queue_size=_type_or_default(
                "max_delete_queue_size", int, DEFAULT_MAX_DELETE_QUEUE_SIZE
            ),
            delete_check_interval_s=_type_or_default(
                "delete_check_interval_s", int, DEFAULT_DELETE_CHECK_INTERVAL_S
            ),
        )

        # sanity check
        if config.max_directory_size_gb is not None:
            assert storage_type == StorageType.Directory, (
                "Invalid set of combinations. Need storage_type=Directory "
                "when used with max_directory_size_gb"
            )
        assert 0.0 <= config.eviction_threshold <= 1.0, (
            f"Invalid eviction threshold : {config.eviction_threshold}"
        )

        return config


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
            logger.info("Register evictor : %s", process_file_path)
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


def get_evictor_hex_range(evictor_id: int, num_evictors: int) -> tuple[int, int]:
    # Always have power of 2 evictor processes doing work with
    # a maximum of 16 evictors.
    evictor_limit = min(16, 2 ** int(math.log2(num_evictors)))
    if evictor_id > evictor_limit - 1:
        return (-1, -1)

    num_hex_per_process = 16 // evictor_limit
    # split 16 hashes evenly among evictors
    hex_range_start = num_hex_per_process * evictor_id
    hex_range_end = hex_range_start + num_hex_per_process - 1
    return (hex_range_start, hex_range_end)


class EvictorProcess(multiprocessing.Process):
    def __init__(
        self,
        root_dir: str,
        cfg: EvictorRuntimeConfig,
    ):
        super().__init__(name="Evictor")
        self.root_dir = root_dir
        self.cfg = cfg

        # disk usage thread for storage_size spawns
        self.du_thread: threading.Thread | None = None
        self.storage_alloc_size = 0

        self.delete_thread: threading.Thread | None = None

        # global evictor status
        self.my_evictor_path: str | None = None
        self.my_evictor_path_fd: int | None = None
        self.num_evictors: int = 0
        self.my_evictor_id: int | None = None

    def delete_thread_fn(self):
        def _should_delete() -> bool:
            """
            True if deletion should be triggered. False otherwise.
            """
            usage_percent = None
            if self.cfg.storage_type == StorageType.Directory:
                dir_size = get_directory_size(self.root_dir)
                usage_percent = dir_size / self.cfg.max_directory_size_bytes
            else:
                usage_percent = get_disk_usage_percent_form_statvfs(self.root_dir)

            if usage_percent is None:
                logger.error("Cannot determine usage percent")
                return False

            return usage_percent > self.cfg.eviction_threshold

        def _delete(pct: float):
            assert 0.0 < pct <= 1.0
            to_delete = self.queues.drain_delete_files(pct)
            logger.info("Delete %d files ...", len(to_delete))
            for file_path in to_delete:
                safe_remove(file_path)

        delete_pct = 0.1  # start with 10% of delete queue
        while self.running:
            if _should_delete():
                _delete(delete_pct)
                delete_pct = min(1.0, delete_pct + 0.1)
            else:
                # backoff
                delete_pct = max(0.1, delete_pct - 0.1)
            time.sleep(2.0)

    def crawler(self, min_hex: int, max_hex: int, yield_timeout_s: int):
        """
        Crawls self.root_dir checking for files to delete.
        Periodically, i.e. when yield_timeout_s runs out, the control
        transfers to the caller where a Discovery step is performed to check
        if this crawler's hex_range should change.
        """
        assert max_hex >= min_hex
        yield
        logger.info(
            "Crawler triggered at %s  for hex range %s, %s",
            self.root_dir,
            min_hex,
            max_hex,
        )

        if min_hex == -1 and max_hex == -1:
            # dummy crawler code path code path code path code path
            while True:
                time.sleep(yield_timeout_s)
                # Check timeout and yield
                should_continue = yield
                if not should_continue:
                    return

        def _record_dir(hex3_dir: os.DirEntry) -> bool:
            # Apply hex modulo filtering for load balancing across crawlers.
            hex_int = hex_to_int(hex3_dir.name)
            if hex_int is None:
                return False
            hex_mod = hex_int % 16
            return min_hex <= hex_mod <= max_hex

        def _yield_bin_files() -> Iterator[os.DirEntry]:
            # Layout: <root_dir>/<safe_model_name>_<sha256-prefix>/<hhh>/<hh>_g<group_idx>/<hash>.bin # noqa: E501
            for model_dir in safe_yield_dir(self.root_dir):
                for hex1 in safe_yield_dir(model_dir.path):
                    if _record_dir(hex1):
                        for hex2 in safe_yield_dir(hex1.path):
                            for bin_file in safe_scandir(hex2.path):
                                if bin_file.is_file(
                                    follow_symlinks=False
                                ) and bin_file.path.endswith(".bin"):
                                    yield bin_file

        def _cleanup():
            self.queues.refresh_file_queue()

        deadline = time.monotonic() + yield_timeout_s

        def _maybe_yield() -> Generator[None, bool | None, bool | None]:
            nonlocal deadline
            if time.monotonic() < deadline:
                return True
            should_continue = yield
            if should_continue:
                deadline = time.monotonic() + yield_timeout_s
            return should_continue

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
                    # skip
                    continue

                if time.time() - access_time < self.cfg.access_time_threshold_s:
                    # skip
                    continue

                self.queues.maybe_put_file_queue(access_time, bin_file.path)
                if self.queues.is_file_queue_full():
                    # slow down
                    time.sleep(0.1)

            self.queues.refresh_file_queue()

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

    def run(self):
        # Mirror vLLM worker process behavior so child logs carry process prefix.
        set_process_title("Evictor")
        decorate_logs("Evictor", skip_if_decorated=True)

        self.queues = EvictorQueues(self.cfg.max_delete_queue_size)

        # handle signals and exit gracefully
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
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
        min_hex, max_hex = get_evictor_hex_range(self.my_evictor_id, self.num_evictors)

        self.running = True

        # launch monitor-delete thread
        self.delete_thread = threading.Thread(target=self.delete_thread_fn)
        self.delete_thread.start()

        while self.running:
            try:
                # trigger crawler
                crawler_p = self.crawler(
                    min_hex,
                    max_hex,
                    yield_timeout_s=self.cfg.delete_check_interval_s,
                )
                crawler_p.send(None)

                while self.running:
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
                        min_hex, max_hex = get_evictor_hex_range(
                            self.my_evictor_id, self.num_evictors
                        )
                        crawler_p.send(False)  # this will trigger stopitereation
            except StopIteration:
                logger.info(
                    "Triggered Reassignment as new evictors discovered."
                    "new_num_evictors=%d, new_my_evictor_id=%d",
                    new_num_evictors,
                    new_my_evictor_id,
                )
            continue

    def signal_handler(self, signum, frame):
        self.running = False
        self.shutdown()

    def shutdown(self):
        logger.info("Shutdown evictor %s ", self.my_evictor_path)
        self.running = False
        if self.delete_thread is not None and self.delete_thread.is_alive():
            self.delete_thread.join()
        if self.my_evictor_path is not None:
            assert self.my_evictor_path_fd is not None
            unregister_evictor_process(self.my_evictor_path, self.my_evictor_path_fd)


def _run_evictor_process(root_dir: str, runtime_config: EvictorRuntimeConfig) -> None:
    EvictorProcess(root_dir, runtime_config).run()


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

    def is_alive(self):
        return self.process.is_alive()

    def spawn_evictor(
        self,
    ):
        """
        Spawn evictor process and always monitor
        """
        mp_ctx = get_mp_context()
        self.process = mp_ctx.Process(
            name="Evictor",
            target=_run_evictor_process,
            args=(self.root_dir, self.evictor_config),
        )
        self.process.start()

    def shutdown(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
