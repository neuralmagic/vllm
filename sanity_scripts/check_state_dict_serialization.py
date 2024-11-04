"""
This script was created to check if the compressed state dict
is the same when saved and loaded using safetensors

Modified save_pretrained on llmcompressor side to return the compressed state dict
when saving the model
"""

from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from transformers import AutoTokenizer
import torch
from safetensors.torch import load_file


model_path = "nm-testing/TinyLlama-1.1B-Chat-v1.0-pruned_50.2of4-uncompressed"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = SparseAutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
save_dir = "TinyLlama-1.1B-Chat-v1.0-pruned_50.2of4-uncompressed-compressed"

oneshot(
    model=model,
    output_dir=save_dir,
)

compressed_state = model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# Load model.safetensors in save_dir using safetensors and compare with compressed_state

load_path = save_dir + "/model.safetensors"

loaded_state_dict = load_file(load_path)

# Compare loaded_state_dict with compressed_state
for key in compressed_state.keys():
    exp = compressed_state[key]
    actual = loaded_state_dict[key]
    assert torch.equal(exp, actual), "Mismatch in " + key
    assert exp.dtype == actual.dtype, "Mismatch in dtype of " + key
    # to cuda and back to cpu to avoid device mismatch
    actual = actual.cuda().cpu()
    assert torch.equal(exp, actual), "Mismatch in " + key
