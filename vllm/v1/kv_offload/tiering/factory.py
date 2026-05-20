# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Factory for creating secondary tier implementations.
"""

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.tiering.base import PrimaryTierMetadata, SecondaryTierManager
from vllm.v1.kv_offload.tiering.example import ExampleSecondaryTier
from vllm.v1.kv_offload.tiering.storage.storage_tier import StorageSecondaryTier

if TYPE_CHECKING:
    from vllm.config import VllmConfig

SUPPORTED_TIERS: tuple[type[SecondaryTierManager], ...] = (
    ExampleSecondaryTier,
    StorageSecondaryTier,
)

_TIER_REGISTRY: dict[str, type[SecondaryTierManager]] = {
    cls.get_tier_type(): cls for cls in SUPPORTED_TIERS
}

logger = init_logger(__name__)


def create_secondary_tier(
    tier_config: dict,
    primary_tier_meta: PrimaryTierMetadata,
    vllm_config: "VllmConfig",
    kv_cache_config: KVCacheConfig,
) -> SecondaryTierManager:
    """
    Create a secondary tier from configuration.

    Args:
        tier_config: Dictionary with tier configuration containing:
            - type (required): Type of secondary tier (e.g., "example")
            - Additional tier-specific parameters are passed directly
              to the tier constructor
        primary_tier_meta: Primary Tier's metadata information.
        vllm_config: Global vLLM configuration.
        kv_cache_config: Global KV Cache config.

    Returns:
        SecondaryTierManager instance

    Raises:
        ValueError: If tier type is unknown or configuration is invalid
    """
    config = tier_config.copy()

    tier_type = config.pop("type", None)
    if not tier_type:
        raise ValueError("Secondary tier configuration must include 'type'")

    cls = _TIER_REGISTRY.get(tier_type)
    if cls is None:
        raise ValueError(
            f"Unknown secondary tier type: {tier_type!r}. "
            f"Supported types: {list(_TIER_REGISTRY)}"
        )

    logger.info("Making Secondary Tier: %s with config %s", cls, config)
    return cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        primary_tier_meta=primary_tier_meta,
        **config,
    )
