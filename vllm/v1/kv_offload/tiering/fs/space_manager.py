# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from typing import Any

from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.fs.ttl_evictor import TTLEvictor, TTLEvictorHandle


class SpaceManager:
    def __init__(
        self,
        root_dir: str,
        file_mapper: FileMapper,
        ttl_evictor_args: dict[str, Any] | None = None,
    ):
        self.root_dir = root_dir
        self.file_mapper = file_mapper

        self._evictor_handle: TTLEvictorHandle | None = None
        if ttl_evictor_args is not None:
            self._evictor_handle = TTLEvictor.spawn(
                root_dir=Path(root_dir),
                **ttl_evictor_args,
            )

    def shutdown(self) -> None:
        if self._evictor_handle is not None:
            self._evictor_handle.stop()
            self._evictor_handle = None
