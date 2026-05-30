# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
import os
import threading
from collections.abc import Iterator
from pathlib import Path

from vllm.logger import init_logger

DEFAULT_EVICTION_THRESHOLD = 0.95
DEFAULT_ACCESS_TIME_THRESHOLD_S = 300  # 5 minutes
DEFAULT_DELETE_CHECK_INTERVAL_S = 30  # 30s
DEFAULT_MAX_DELETE_QUEUE_SIZE = 1000

logger = init_logger(__name__)


class EvictorQueues:
    def __init__(self, max_queue_size: int):
        self.max_queue_size = max_queue_size
        self.file_queue: list[tuple[float, str]] = []
        self.delete_queue: list[tuple[float, str]] = []
        self._swap_lock = threading.Lock()

    def _swap_queues(self):
        tmp = self.delete_queue
        self.delete_queue = self.file_queue
        self.file_queue = tmp

    def is_file_queue_full(self):
        with self._swap_lock:
            return len(self.file_queue) == self.max_queue_size

    def refresh_file_queue(self):
        with self._swap_lock:
            # logger.info(f"Refresh swap ...")
            self._swap_queues()
            self.file_queue = []

    def drain_delete_files(self, percent: float) -> list[tuple[float, str]]:
        assert 0.0 <= percent <= 1.0, f"got {percent}"
        with self._swap_lock:
            if not self.delete_queue and self.file_queue:
                # swap only if delete_queue empty and file_queue has values
                # logger.info(f"Delete swap - file queue {len(self.file_queue)}...")
                self._swap_queues()
            total_delete = len(self.delete_queue)
            num_delete = int(total_delete * percent)
            to_delete = []
            while num_delete:
                atime, file_path = heapq.heappop(self.delete_queue)
                to_delete.append((-atime, file_path))
                num_delete -= 1
            return to_delete

    def maybe_put_file_queue(self, access_time: float, file_path: str):
        # Most accessed function
        with self._swap_lock:
            if len(self.file_queue) == self.max_queue_size:
                if access_time < -self.file_queue[0][0]:
                    heapq.heappushpop(self.file_queue, (-access_time, file_path))
            else:
                heapq.heappush(self.file_queue, (-access_time, file_path))


def hex_to_int(hex_str: str) -> int | None:
    """Convert hex string to integer."""
    try:
        return int(hex_str, 16)
    except (ValueError, TypeError):
        return None


## Safe os functions


def safe_scandir(path: str) -> Iterator[os.DirEntry]:
    try:
        return os.scandir(path)
    except (OSError, PermissionError):
        return iter([])


def safe_yield_dir(path: str) -> Iterator[os.DirEntry]:
    for entry in safe_scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield entry


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


def get_disk_usage_percent_form_statvfs(root_dir: str) -> float | None:
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
        usage_percent = (used_bytes / total_bytes) if total_bytes > 0 else 0
        return usage_percent

    except Exception as e:
        logger.error("Cannot fetch disk usage. Failed with %s", e)
        return None


def get_directory_size(path: str):
    total_bytes = 0
    stack: list[Path] = [Path(path)]
    while stack:
        entry = stack.pop()
        for sub_entry in safe_scandir(str(entry)):
            if sub_entry.is_dir(follow_symlinks=False):
                stack.append(Path(sub_entry.path))
            elif sub_entry.is_file(follow_symlinks=False):
                total_bytes += Path(sub_entry.path).stat().st_size
    return total_bytes
