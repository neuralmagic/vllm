# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import uuid

import torch

MASK_64_BITS = (1 << 64) - 1


def random_uuid() -> str:
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars


def length_from_prompt_token_ids_or_embeds(
    prompt_token_ids: list[int] | torch.Tensor | None,
    prompt_embeds: torch.Tensor | None,
) -> int:
    """Calculate the request length (in number of tokens) give either
    prompt_token_ids or prompt_embeds.
    """
    prompt_token_len = None if prompt_token_ids is None else len(prompt_token_ids)
    prompt_embeds_len = None if prompt_embeds is None else len(prompt_embeds)

    if prompt_token_len is None:
        if prompt_embeds_len is None:
            raise ValueError("Neither prompt_token_ids nor prompt_embeds were defined.")
        return prompt_embeds_len
    else:
        if prompt_embeds_len is not None and prompt_embeds_len != prompt_token_len:
            raise ValueError(
                "Prompt token ids and prompt embeds had different lengths"
                f" prompt_token_ids={prompt_token_len}"
                f" prompt_embeds={prompt_embeds_len}"
            )
        return prompt_token_len


def is_moe_layer(module: torch.nn.Module) -> bool:
    # TODO(bnell): Should use isinstance but can't.  Maybe search for
    # presence of quant_method.maybe_init_modular_kernel?
    # return (hasattr(module, "quant_method")
    #         and hasattr(module.quant_method, "moe_kernel"))
    return (
        module.__class__.__name__ == "FusedMoE"
        or module.__class__.__name__ == "SharedFusedMoE"
        or module.__class__.__name__ == "DefaultMoERunner"
        or module.__class__.__name__ == "ChunkingMoERunner"
    )
