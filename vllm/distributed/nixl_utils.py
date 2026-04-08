# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import os
import sys
from importlib import import_module
from types import ModuleType

from vllm.platforms import current_platform

_UCX_RCACHE_MAX_UNRELEASED = "1024"


def prepare_nixl_import(logger: logging.Logger) -> None:
    if "UCX_RCACHE_MAX_UNRELEASED" in os.environ:
        return

    # Avoid a memory leak in UCX when using NIXL on some models.
    # See: https://github.com/vllm-project/vllm/issues/24264
    # This must happen before the first nixl/rixl import to affect the
    # library initialization path.
    if "nixl" in sys.modules or "rixl" in sys.modules:
        logger.warning(
            "NIXL was already imported, we can't reset "
            "UCX_RCACHE_MAX_UNRELEASED. Please set it to '%s' manually.",
            _UCX_RCACHE_MAX_UNRELEASED,
        )
        return

    logger.info(
        "Setting UCX_RCACHE_MAX_UNRELEASED to '%s' to avoid a rare "
        "memory leak in UCX when using NIXL.",
        _UCX_RCACHE_MAX_UNRELEASED,
    )
    os.environ["UCX_RCACHE_MAX_UNRELEASED"] = _UCX_RCACHE_MAX_UNRELEASED


def import_nixl_module(module_suffix: str, logger: logging.Logger) -> ModuleType:
    prepare_nixl_import(logger)
    package_name = "rixl" if current_platform.is_rocm() else "nixl"
    return import_module(f"{package_name}.{module_suffix}")
