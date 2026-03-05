# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import Any

from transformers import PretrainedConfig

from vllm.transformers_utils.configs.speculators.algos import (
    SUPPORTED_SPECULATORS_TYPES,
)

__all__ = ["SpeculatorsConfig"]


class SpeculatorsConfig(PretrainedConfig):
    model_type = "speculators"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **kwargs,
    ) -> "SpeculatorsConfig":
        config_dict, _ = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls(**cls.extract_transformers_pre_trained_config(config_dict))

    @classmethod
    def extract_transformers_pre_trained_config(
        cls, config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        speculators_model_type = config_dict.get("speculators_model_type")
        if speculators_model_type not in SUPPORTED_SPECULATORS_TYPES:
            raise ValueError(
                f"Expected one of: {SUPPORTED_SPECULATORS_TYPES}. "
                "Please ensure you're loading a speculators-format model."
            )

        pre_trained_config = dict(
            config_dict.get("transformer_layer_config")
            or config_dict.get("transformer_config", {})
        )
        algo_updater = SUPPORTED_SPECULATORS_TYPES[speculators_model_type]
        algo_updater(config_dict=config_dict, pre_trained_config=pre_trained_config)
        return pre_trained_config

    @classmethod
    def extract_vllm_speculative_config(
        cls, config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        # TODO: @dsikka - use speculators pydantic model to validate
        cls.validate_speculators_config(config_dict=config_dict)
        return cls.build_vllm_speculative_config(config_dict=config_dict)

    @classmethod
    def validate_speculators_config(cls, config_dict: dict[str, Any]) -> None:
        try:
            spec_config = config_dict["speculators_config"]
            methods = spec_config["proposal_methods"]
            first_method = methods[0]
            _ = first_method["speculative_tokens"]
            _ = spec_config["verifier"]["name_or_path"]
            _ = config_dict["speculators_model_type"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError("Invalid speculators config structure") from e

        if (
            "transformer_layer_config" not in config_dict
            and "transformer_config" not in config_dict
        ):
            raise ValueError(
                "Must provide transformer_layer_config or transformer_config"
            )

        transformer_cfg = config_dict.get(
            "transformer_layer_config"
        ) or config_dict.get("transformer_config")
        if not isinstance(transformer_cfg, dict):
            raise TypeError(
                "'transformer_layer_config'/'transformer_config' must be a dictionary"
            )

    @classmethod
    def build_vllm_speculative_config(
        cls, config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        spec_config = config_dict["speculators_config"]

        proposal_methods = spec_config.get("proposal_methods")
        if not proposal_methods:
            raise ValueError("No proposal methods found in speculators config")

        first_method = proposal_methods[0]
        num_speculative_tokens = first_method.get("speculative_tokens")
        if num_speculative_tokens is None:
            raise ValueError(
                f"Missing 'speculative_tokens' in proposal method. Got: {first_method}"
            )

        return {
            "method": config_dict.get("speculators_model_type"),
            "num_speculative_tokens": num_speculative_tokens,
        }
