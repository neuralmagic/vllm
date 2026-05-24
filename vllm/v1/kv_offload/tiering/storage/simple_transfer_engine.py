# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Callable, Collection
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.base import (
    JobId,
)
from vllm.v1.kv_offload.tiering.storage.common import TransferResult

logger = init_logger(__name__)


class SimpleTransferEngine:
    def __init__(self, primary_kv: np.ndarray, num_threads: int):
        self.primary_kv = primary_kv
        self.num_threads = num_threads
        self._executor = ThreadPoolExecutor(max_workers=num_threads)

    @staticmethod
    def _file_write(file_path: Path, np_arr: np.ndarray) -> bool:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = file_path.with_suffix(".tmp")

        with open(tmp_file, mode="w+b") as f:
            np_arr.tofile(f)
        tmp_file.rename(file_path)  # rename at the end for atomicity
        return True

    @staticmethod
    def _file_read(file_path: Path, np_arr: np.ndarray) -> bool:
        with open(str(file_path), "rb") as f:
            bytes_read = f.readinto(np_arr)
            assert bytes_read == np_arr.nbytes, f"{bytes_read=} != {np_arr.nbytes}"
        return True

    def _do_op(
        self,
        job_id: JobId,
        file_paths: Collection[Path],
        block_ids: Collection[int],
        op: Callable[[Path, np.ndarray], bool],
    ) -> TransferResult:
        assert len(file_paths) == len(block_ids)
        start = time.perf_counter()

        results = []
        try:
            futures = [
                self._executor.submit(op, fp, self.primary_kv[bid])
                for fp, bid in zip(file_paths, block_ids)
            ]
            wait(futures, return_when=ALL_COMPLETED)
            results = [f.result() for f in futures]
        except Exception as e:
            logger.info("Job %s failed: %s", job_id, e)
            return TransferResult()

        return TransferResult(
            success=all(results),
            transfer_size=sum([self.primary_kv[b].nbytes for b in block_ids]),
            transfer_time=time.perf_counter() - start,
        )

    def load(
        self, job_id: JobId, file_paths: Collection[Path], block_ids: Collection[int]
    ) -> TransferResult:
        return self._do_op(job_id, file_paths, block_ids, self._file_read)

    def store(
        self, job_id: JobId, file_paths: Collection[Path], block_ids: Collection[int]
    ) -> TransferResult:
        return self._do_op(job_id, file_paths, block_ids, self._file_write)
