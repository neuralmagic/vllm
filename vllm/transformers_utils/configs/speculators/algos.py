# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

SUPPORTED_SPECULATORS_TYPES = {}


def register_speculator(name):
    def decorator(fn):
        SUPPORTED_SPECULATORS_TYPES[name] = fn
        return fn

    return decorator


@register_speculator("mtp")
def update_mtp(config_dict: dict, pre_trained_config: dict) -> None:
    """Configure vLLM to load a speculators FastMTP checkpoint.

    Extracts num_nextn_predict_layers from the top-level config (not
    transformer_layer_config) and sets the architecture to Qwen3NextMTP
    for qwen3_next base models.
    """
    tl_cfg = config_dict.get("transformer_layer_config", {})
    if tl_cfg.get("model_type") == "qwen3_next":
        pre_trained_config["architectures"] = ["Qwen3NextMTP"]
    pre_trained_config["speculators_model_type"] = "mtp"
    pre_trained_config["num_nextn_predict_layers"] = config_dict.get(
        "num_nextn_predict_layers", 1
    )
    pre_trained_config["n_predict"] = pre_trained_config["num_nextn_predict_layers"]


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
