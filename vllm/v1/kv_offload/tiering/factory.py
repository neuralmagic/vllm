# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.v1.kv_offload.tiering.base import SecondaryTierManager

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingMetricMetadata, OffloadingSpec


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
        classmethod. Definitions from multiple tiers of the same type are
        merged (last writer wins for duplicate names, though duplicates should
        not occur in practice).

        Args:
            tier_configs: List of tier configuration dicts from extra_config
                (each must contain a ``type`` key).
        """
        definitions: dict[str, OffloadingMetricMetadata] = {}
        for tier_config in tier_configs:
            tier_type = tier_config.get("type")
            if tier_type and tier_type in cls._registry:
                tier_cls = cls._registry[tier_type]()
                definitions.update(tier_cls.build_metric_definitions(tier_config))
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

        tier_cls = cls._registry[tier_type]()
        return tier_cls(
            offloading_spec=offloading_spec,
            primary_kv_view=primary_kv_view,
            tier_type=tier_type,
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
