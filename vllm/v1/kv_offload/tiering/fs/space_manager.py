# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.file_mapper import FileMapper


class SpaceManager:
    def __init__(self, root_dir: str, file_mapper: FileMapper):
        self.root_dir = root_dir
        self.file_mapper = file_mapper

    def lookup(self, key: OffloadKey) -> bool:
        return os.path.exists(self.file_mapper.get_file_name(key))
