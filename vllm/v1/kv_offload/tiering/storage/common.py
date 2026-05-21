# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass
class TransferResult:
    success: bool = False
    transfer_size: int = 0
    transfer_time: float = 0
