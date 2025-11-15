import os
import json
import torch
from safetensors.torch import save_file
from vllm.distributed import get_tensor_model_parallel_rank

def save_shards_if_rank0(model):
    """
    Save the full model as multiple .safetensors shard files,
    but only on rank 0 (to avoid multi-GPU overwrite issues).
    """
    rank = get_tensor_model_parallel_rank()
    if rank != 0:
        print(f"[Rank {rank}] Skipping save.")
        return "skipped"

    print("[Rank 0] Starting model save...")

    PATH = "/raid/engine/hub_cache/ml3-nvfp4"
    os.makedirs(PATH, exist_ok=True)
    max_shard_size = 2 * 1024**3  # 2 GB

    current_shard = {}
    current_size = 0
    shard_index = 0
    index_meta = {}

    # Collect all tensors
    state_dict = {}
    for prefix, module in model.named_modules():
        updated = {
            f"{prefix}.{name}" if prefix else name: param
            for name, param in module.named_parameters(recurse=False)
            if "input_global_scale" in name

        }
        state_dict.update(updated)
    
    #for k, v in state_dict.items():
    #    print(k, v.shape)

    # Write shards
    for name, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > max_shard_size and current_shard:
            file_path = os.path.join(PATH, f"model-{shard_index:05d}.safetensors")
            save_file(current_shard, file_path)
            index_meta[file_path] = list(current_shard.keys())
            print(f"[Rank 0] Saved shard {shard_index} ({len(current_shard)} tensors).")

            shard_index += 1
            current_shard.clear()
            current_size = 0

        current_shard[name] = tensor
        current_size += tensor_size

    # Final shard
    if current_shard:
        file_path = os.path.join(PATH, f"model-{shard_index:05d}.safetensors")
        save_file(current_shard, file_path)
        index_meta[file_path] = list(current_shard.keys())
        print(f"[Rank 0] Saved final shard {shard_index} ({len(current_shard)} tensors).")

    # Write global index JSON
    index_path = os.path.join(PATH, "model_index.json")
    with open(index_path, "w") as f:
        json.dump(index_meta, f, indent=2)

    print("[Rank 0] Model save complete.")
    return "saved"
