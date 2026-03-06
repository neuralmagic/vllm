# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

SUPPORTED_SPECULATORS_TYPES = {}


def register_speculator(name):
    def decorator(fn):
        SUPPORTED_SPECULATORS_TYPES[name] = fn
        return fn

    return decorator


@register_speculator("eagle3")
def update_eagle3(config_dict: dict, pre_trained_config: dict) -> None:
    """
    Apply Eagle-3 specific configuration transformations to the `dict` used to
    construct the Transformers PreTrainedConfig.

    Eagle-3 specific fields:
    - draft_vocab_size: Size of the draft model's vocabulary
    - target_hidden_size: Hidden size of the target model
    - norm_before_residual: Whether to apply norm before residual connection
    - eagle_aux_hidden_state_layer_ids: List of layer indices from the base
        model to use as auxiliary inputs for the Eagle3 drafter. These layers
        provide intermediate hidden states that help the drafter make better
        predictions. This is the standard field used in Eagle3 checkpoints.
    """

    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    if config_dict.get("target_hidden_size") is not None:
        pre_trained_config["target_hidden_size"] = config_dict["target_hidden_size"]
    pre_trained_config["norm_before_residual"] = config_dict.get(
        "norm_before_residual", True
    )
    pre_trained_config["architectures"] = ["Eagle3LlamaForCausalLM"]
    if config_dict.get("eagle_aux_hidden_state_layer_ids"):
        pre_trained_config["eagle_aux_hidden_state_layer_ids"] = config_dict[
            "eagle_aux_hidden_state_layer_ids"
        ]


@register_speculator("mtp")
def update_mtp(config_dict: dict, pre_trained_config: dict) -> None:
    """Apply MTP-specific config fields for standalone speculators MTP checkpoints.

    hf_config_override() uses architectures[0] (MiMo) or model_type (Qwen3-Next)
    to remap to the vLLM MTP model type, so both must be correctly populated.
    n_predict is consumed by SpeculativeConfig to set num_speculative_tokens.
    num_nextn_predict_layers is consumed by the MTP model constructors.
    """
    # Set architectures from verifier so hf_config_override detects the right MTP type:
    #   "MiMoForCausalLM"     → mimo_mtp (num_hidden_layers also set to 0)
    #   "Qwen3MoeForCausalLM" → model_type="qwen3_next" (already in pre_trained_config)
    #                           → qwen3_next_mtp
    verifier_archs = (
        config_dict.get("speculators_config", {})
        .get("verifier", {})
        .get("architectures", [])
    )
    if verifier_archs:
        pre_trained_config["architectures"] = verifier_archs

    # Number of MTP layers — consumed by MiMoMultiTokenPredictor
    # and Qwen3NextMultiTokenPredictor.
    pre_trained_config.setdefault(
        "num_nextn_predict_layers", config_dict.get("num_nextn_predict_layers", 1)
    )

    # n_predict — consumed by SpeculativeConfig to derive num_speculative_tokens
    # when not explicitly provided.
    proposal_methods = config_dict.get("speculators_config", {}).get(
        "proposal_methods", []
    )
    if proposal_methods:
        pre_trained_config["n_predict"] = proposal_methods[0].get(
            "speculative_tokens", pre_trained_config.get("num_nextn_predict_layers", 1)
        )
