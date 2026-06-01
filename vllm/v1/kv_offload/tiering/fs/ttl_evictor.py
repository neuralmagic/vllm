# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# make a stale deleter
#   - the deleter can exist in eager and lazy mode -
#       - in eager mode - it just deletes what it can find
#       - in lazy mode - it is triggered only when delete thresholds are reached
#   - make a du reporter

# make a evictor manager v1
#   - evictor manager spawns a deleter process.
#   - monitors store fails -- when stores start failing - retries after timeout.

# make a evictor manager v2
#   - spawns a stale deleter process
#   - manages a cache -- starts removing from the cache for LRU if the stores
#      start failing . This is so the new blocks have a shot -
#      is flexible in providing some guarantees

import fcntl
import os
import random
import signal
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

DU_CHECK_INTERVAL_S = 5  # check usage every 5s
DELETE_TRIGGER_POLL_S = 5  # wake up delete trigger every 5s to check shutdown.


def get_disk_usage_fraction_form_statvfs(root_dir: Path) -> float | None:
    try:
        stat = os.statvfs(root_dir)
        block_size = stat.f_frsize
        total_bytes = stat.f_blocks * block_size
        free_bytes = stat.f_bfree * block_size
        used_bytes = total_bytes - free_bytes
        usage = (used_bytes / total_bytes) if total_bytes > 0 else 0
        return usage

    except Exception as e:
        logger.error("Cannot fetch disk usage. Failed with %s", e)
        return None


def safe_scandir(path: str) -> Iterator[os.DirEntry]:
    try:
        return os.scandir(path)
    except (OSError, PermissionError):
        return iter([])


def safe_yield_dir(path: str) -> Iterator[os.DirEntry]:
    for entry in safe_scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield entry


def safe_remove(path: Path):
    try:
        os.remove(path)
    except Exception as e:
        logger.error("Failed for remove path %s : %s", path, e)


def safe_close(fd: int):
    try:
        os.close(fd)
    except Exception as e:
        logger.error("Failed to close fd %d : %s", fd, e)


def safe_atime(path: Path) -> float | None:
    try:
        return os.path.getatime(path)  # Last access time
    except (OSError, AttributeError) as e:
        logger.error("Cannot fetch file atime : %s", e)
    return None


def permute(it: Iterator[os.DirEntry], batch_size: int = 50) -> Iterator[os.DirEntry]:
    batch: list[os.DirEntry] = []
    for e in it:
        batch.append(e)
        if len(batch) == batch_size:
            random.shuffle(batch)
            while batch:
                yield batch.pop()
    # balance
    random.shuffle(batch)
    while batch:
        yield batch.pop()


# Layout: <root_dir>/<safe_model_name>_<sha256-prefix>/<hhh>/<hh>_g<group_idx>/<hash>.bin # noqa: E501
def yield_bin_files(root_dir) -> Iterator[Path]:
    for model_dir in permute(safe_yield_dir(root_dir)):
        for hex1 in permute(safe_yield_dir(model_dir.path)):
            for hex2 in permute(safe_yield_dir(hex1.path)):
                for bin_file in permute(safe_scandir(hex2.path)):
                    if bin_file.is_file(
                        follow_symlinks=False
                    ) and bin_file.path.endswith(".bin"):
                        yield Path(bin_file.path)


class TTLEvictor:
    def __init__(
        self,
        root_dir: Path,
        ttl_s: int = 60 * 60,  # 1 hr
        is_lazy: bool = True,
        lazy_eviction_high_watermark: float = 0.9,
        lazy_eviction_low_watermark: float = 0.8,
        num_active_evictors: int = 8,
        register_retry_interval_s: int = 60,
    ):
        self.root_dir = root_dir
        self.ttl_s = ttl_s
        self.is_lazy = is_lazy
        self.eviction_high_watermark = lazy_eviction_high_watermark
        self.eviction_low_watermark = lazy_eviction_low_watermark
        self.num_active_evictors = num_active_evictors
        self.register_retry_interval_s = register_retry_interval_s

        self.du_thread: threading.Thread | None = None
        self.delete_trigger: threading.Event = threading.Event()

        self.evictor_fd: int | None = None
        self.evictor_path: Path | None = None

        self.shutdown: threading.Event = threading.Event()

        assert self.eviction_high_watermark >= self.eviction_low_watermark

    def signal_handler(self, signum, frame):
        self.shutdown.set()

    @staticmethod
    def is_evictor_alive(evictor_path: Path) -> bool:
        try:
            with open(evictor_path) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except BlockingIOError:
            return True
        except Exception as e:
            # cannot determine liveness. consider unalive.
            logger.error("Cannot determine liveness %s : %s", evictor_path, e)
            return False
        return False

    @staticmethod
    def maybe_register(e_path: Path) -> int | None:
        if TTLEvictor.is_evictor_alive(e_path):
            return None

        # Try reclaiming the dead evictor
        safe_remove(e_path)

        fd = None
        try:
            fd = os.open(e_path, os.O_CREAT | os.O_EXCL, 0o644)
            # lock file. The lock is what other evictor processes check for liveness.
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            if fd is not None:
                safe_close(fd)
                safe_remove(e_path)
            return None

        return fd

    def register(self) -> bool:
        fd = None
        for e in range(self.num_active_evictors):
            e_path = self.root_dir / f"__{e}.evictor"
            fd = self.maybe_register(e_path)
            if fd is not None:
                self.evictor_fd = fd
                self.evictor_path = Path(e_path)
                return True
        return False

    def unregister(self):
        if self.evictor_fd is None:
            return
        assert self.evictor_path is not None
        try:
            fcntl.flock(self.evictor_fd, fcntl.LOCK_UN)
            os.close(self.evictor_fd)
            safe_remove(self.evictor_path)
        except Exception as e:
            logger.error("Cannot unregister safely : %s", e)

    def du_thread_fn(self):
        while not self.shutdown.is_set():
            pct = get_disk_usage_fraction_form_statvfs(self.root_dir)
            if pct is not None:
                if pct >= self.eviction_high_watermark:
                    self.delete_trigger.set()

                if pct < self.eviction_low_watermark:
                    self.delete_trigger.clear()

            self.shutdown.wait(timeout=DU_CHECK_INTERVAL_S)

    def run(self):
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        while not self.register():
            if self.shutdown.wait(timeout=self.register_retry_interval_s):
                break

        if self.shutdown.is_set():
            return self.cleanup()

        logger.info("Evictor registered at %s", self.evictor_path)

        if self.is_lazy:
            self.du_thread = threading.Thread(target=self.du_thread_fn, daemon=True)
            self.du_thread.start()
        else:
            # delete eagerly
            self.delete_trigger.set()

        try:
            while not self.shutdown.is_set():
                if not self.delete_trigger.wait(timeout=DELETE_TRIGGER_POLL_S):
                    # polling failed.
                    continue

                if self.shutdown.is_set():
                    break

                for bin_file in yield_bin_files(self.root_dir):
                    if self.shutdown.is_set() or not self.delete_trigger.is_set():
                        break

                    a_time = safe_atime(bin_file)
                    if a_time is None:
                        continue

                    if a_time + self.ttl_s < time.time():
                        safe_remove(bin_file)

        except Exception as e:
            logger.error("TTLEvictor terminated with %s", e)
        finally:
            self.cleanup()

    def cleanup(self):
        if self.du_thread is not None:
            self.du_thread.join()
        self.unregister()
