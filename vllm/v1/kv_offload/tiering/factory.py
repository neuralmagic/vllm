# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import regex as re

from vllm.v1.kv_offload.tiering.base import SecondaryTierManager

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingMetricMetadata, OffloadingSpec

# tier_name is embedded directly in Prometheus metric names, which require
# [a-zA-Z0-9_]. Do not sanitize silently — reject invalid names immediately
# so users fix their config rather than end up with unexpected metric names.
_TIER_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")


def _resolve_tier_name(tier_type: str, tier_config: dict) -> str:
    """Resolve the effective name for a secondary tier instance.

    Uses ``tier_config["name"]`` when provided; otherwise falls back to
    ``tier_type`` (e.g. ``"fs"``), which preserves the original metric names
    for single-tier deployments.

    Args:
        tier_type: Registered tier type string (e.g. ``"fs"``).
        tier_config: Raw tier configuration dict from extra_config.

    Returns:
        Validated tier name string safe for use in Prometheus metric names.

    Raises:
        ValueError: If the user-supplied name contains characters outside
            ``[a-zA-Z0-9_]``.
    """
    name = tier_config.get("name")
    if name is not None:
        if not _TIER_NAME_RE.match(name):
            raise ValueError(
                f"tier name {name!r} contains invalid characters. "
                "Use only letters, digits, and underscores."
            )
        return name
    return tier_type


class SecondaryTierFactory:
    _registry: dict[str, Callable[[], type[SecondaryTierManager]]] = {}

    @classmethod
    def register_tier(cls, tier_type: str, module_path: str, class_name: str) -> None:
        if tier_type in cls._registry:
            raise ValueError(f"Tier '{tier_type}' is already registered.")

        def loader() -> type[SecondaryTierManager]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[tier_type] = loader

    @classmethod
    def get_metric_definitions(
        cls, tier_configs: list[dict]
    ) -> "dict[str, OffloadingMetricMetadata]":
        """Collect Prometheus metric definitions from all configured secondary tiers.

        Loads each tier class and calls its ``build_metric_definitions``
        classmethod, passing the full ``tier_config`` so each tier can resolve
        its own optional ``name`` for metric namespacing.

        Args:
            tier_configs: List of tier configuration dicts from extra_config
                (each must contain a ``type`` key).
        """
        definitions: dict[str, OffloadingMetricMetadata] = {}
        for tier_config in tier_configs:
            tier_type = tier_config.get("type")
            if tier_type and tier_type in cls._registry:
                tier_cls = cls._registry[tier_type]()
                tier_name = _resolve_tier_name(tier_type, tier_config)
                definitions.update(
                    tier_cls.build_metric_definitions(tier_config, tier_name)
                )
        return definitions

    @classmethod
    def create_secondary_tier(
        cls,
        tier_config: dict,
        primary_kv_view: memoryview,
        offloading_spec: "OffloadingSpec",
    ) -> SecondaryTierManager:
        config = tier_config.copy()

        tier_type = config.pop("type", None)
        if not tier_type:
            raise ValueError("Secondary tier configuration must include 'type'")

        if tier_type not in cls._registry:
            raise ValueError(
                f"Unknown secondary tier type: {tier_type!r}. "
                f"Supported types: {list(cls._registry)}"
            )

        tier_name = _resolve_tier_name(tier_type, config)
        config.pop("name", None)

        tier_cls = cls._registry[tier_type]()
        return tier_cls(
            offloading_spec=offloading_spec,
            primary_kv_view=primary_kv_view,
            tier_name=tier_name,
            **config,
        )


SecondaryTierFactory.register_tier(
    "example",
    "vllm.v1.kv_offload.tiering.example.manager",
    "ExampleSecondaryTierManager",
)

SecondaryTierFactory.register_tier(
    "fs",
    "vllm.v1.kv_offload.tiering.fs.manager",
    "FileSystemTierManager",
)

SecondaryTierFactory.register_tier(
    "obj",
    "vllm.v1.kv_offload.tiering.obj.manager",
    "ObjectStoreSecondaryTierManager",
)
