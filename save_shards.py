import os
import json
import torch
from torch.distributed import ReduceOp
from safetensors.torch import save_file
from vllm.distributed import get_tensor_model_parallel_rank, tensor_model_parallel_all_gather, get_tp_group


def remap_attention_substrings(state_dict):
    """
    Replace substrings in state_dict keys:

        self_attn.q_b_proj  -> attention.wq_b
        self_attn.kv_b_proj -> attention.wkv_b
        self_attn.o_proj    -> attention.wo

    Returns the modified state_dict.
    """

    replacements = {
        "self_attn.q_b_proj": "attention.wq_b",
        "self_attn.kv_b_proj": "attention.wkv_b",
        "self_attn.o_proj": "attention.wo",
    }

    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements.items():
            if old in new_key:
                new_key = new_key.replace(old, new)
        new_state_dict[new_key] = value

    return new_state_dict


def split_gate_up(state_dict: dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("language_model.model.", "")

        if "shared_expert" in new_key:
            new_key = new_key.replace("mlp.", "")
        else:
            new_key = new_key.replace("mlp", "feed_forward")

        if "gate_up" in new_key:
            gate_key = new_key.replace("gate_up_proj", "w1")
            up_key   = new_key.replace("gate_up_proj", "w3")

            # Add new keys only
            new_state_dict[gate_key] = value.clone()
            new_state_dict[up_key]   = value.clone()
            continue

        if "down_proj" in new_key:
            new_key = new_key.replace("down_proj", "w2")

        new_state_dict[new_key] = value

    return new_state_dict

def split_expert_input_global_scales(state_dict):
    keys_to_process = [
        k for k in state_dict
        if ".experts." in k and (
            k.endswith("w13_input_global_scale") or
            k.endswith("w2_input_global_scale")
        )
    ]

    new_entries = {}
    keys_to_delete = []

    for key in keys_to_process:
        tensor = state_dict[key]

        # Extract layer index
        parts = key.split(".")
        layer_id = parts[1]

        if key.endswith("w13_input_global_scale"):
            if tensor.dim() != 2 or tensor.size(1) != 2:
                raise ValueError(f"{key} expected [N,2], got {tuple(tensor.shape)}")

            num_experts = tensor.size(0)
            w1_vals = tensor[:, 0]
            w3_vals = tensor[:, 1]

            for i in range(num_experts):
                new_entries[f"layers.{layer_id}.experts.{i}.w1.input_global_scale"] = w1_vals[i].clone().reshape([1])
                new_entries[f"layers.{layer_id}.experts.{i}.w3.input_global_scale"] = w3_vals[i].clone().reshape([1])

        else:  # w2_input_global_scale
            if tensor.dim() != 2 or tensor.size(1) != 1:
                raise ValueError(f"{key} expected [N,1], got {tuple(tensor.shape)}")

            num_experts = tensor.size(0)
            w2_vals = tensor[:, 0]

            for i in range(num_experts):
                new_entries[f"layers.{layer_id}.experts.{i}.w2.input_global_scale"] = w2_vals[i].clone().reshape([1])

        # mark original for deletion
        keys_to_delete.append(key)

    for k in keys_to_delete:
        del state_dict[k]

    state_dict.update(new_entries)

    return state_dict


def save_shards_if_rank0(model):
    """
    Save the full model as multiple .safetensors shard files,
    but only on rank 0 (to avoid multi-GPU overwrite issues).
    """
    rank = get_tensor_model_parallel_rank()

    # Collect all tensors
    state_dict = {}
    for prefix, module in model.named_modules():
        updated = {
            f"{prefix}.{name}" if prefix else name: param
            for name, param in module.named_parameters(recurse=False)
            if "input_global_scale" in name

        }
        state_dict.update(updated)

    # for key, value in state_dict.items():
    #     gathered = tensor_model_parallel_all_gather(value.unsqueeze(-1), dim=-1)
    #     minned = torch.min(gathered, dim=-1).values
    #     if rank == 0:
    #         print(key)
    #         print(value.shape)
    #         print(gathered.shape)
    #         print(minned.shape)
    # return

    # all gather min
    state_dict = {
        key: torch.min(tensor_model_parallel_all_gather(value.unsqueeze(-1), dim=-1), dim=-1).values
        for key, value in state_dict.items()
    }

    if rank != 0:
        return
    
    remapped = remap_attention_substrings(state_dict)
    remapped_split = split_gate_up(remapped)
    remapped_final = split_expert_input_global_scales(remapped_split)

    PATH = "/raid/engine/kylesayrs/mistral-large-3-NVFP4"
    os.makedirs(PATH, exist_ok=True)
    max_shard_size = 2 * 1024**3  # 2 GB

    current_shard = {}
    current_size = 0
    shard_index = 0
    index_meta = {}

    for name, tensor in remapped_final.items():
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > max_shard_size and current_shard:
            file_path = os.path.join(PATH, "consolidated-00273-of-00273.safetensors")
            save_file(current_shard, file_path)
            index_meta["weight_map"] = {key: "consolidated-00273-of-00273.safetensors" for key in current_shard.keys()}
            print(f"[Rank 0] Saved shard {shard_index} ({len(current_shard)} tensors).")

            shard_index += 1
            current_shard.clear()
            current_size = 0

        current_shard[name] = tensor
        current_size += tensor_size

    # Final shard
    if current_shard:
        file_path = os.path.join(PATH, "consolidated-00273-of-00273.safetensors")
        save_file(current_shard, file_path)
        index_meta["weight_map"] = {key: "consolidated-00273-of-00273.safetensors" for key in current_shard.keys()}
        print(f"[Rank {rank}] Saved final shard {shard_index} ({len(current_shard)} tensors).")

    # Write global index JSON
    # index_path = os.path.join(PATH, "model.safetensors.index.json")
    # with open(index_path, "w") as f:
    #     json.dump(index_meta, f, indent=2)

    print(f"[Rank {rank}] Model save complete.")
    return "saved"
