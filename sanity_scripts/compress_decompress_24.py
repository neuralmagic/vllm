"""
This test script is used to check sanity of compression.

Uses:
- llmcompressor
- compressed_tensors
- transformers
- torch
(NO VLLM)

The script does the following:
1) Load a compressible (but not yet compressed) model from Hugging Face model hub or local directory
    - Let's call this model the `uncompressed_model`

2) Compress the uncompressed model using the ModelCompressor class
    - Entails calling save_pretrained on the uncompressed model, the sparse
      compressor is inferred from the model, and the compressed model is saved
      to disk
    - Let's call this directory the compressed_save_directory

3) Initialize a new base model from Hugging Face model hub or local directory,
    load in compressed weights from the `compressed_save_directory` using
    ModelCompressor.decompress (To bypass HFQuantizer);
    - Let's call this model the decompressed model

4) Check that the parameters in the decompressed model are the same as the
    uncompressed_model

5) Run inference on the decompressed model
"""

import torch
from llmcompressor.transformers import SparseAutoModelForCausalLM
from compressed_tensors.compressors import ModelCompressor
from transformers import AutoTokenizer

hf_model_stub = "nm-testing/TinyLlama-1.1B-Chat-v1.0-pruned_50.2of4-uncompressed"
uncompressed_model = SparseAutoModelForCausalLM.from_pretrained(hf_model_stub, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(hf_model_stub)

compressed_save_dir = "temp-model"
uncompressed_model.save_pretrained(save_directory=compressed_save_dir)
tokenizer.save_pretrained(save_directory=compressed_save_dir)

base_stub = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
decompressed_model = SparseAutoModelForCausalLM.from_pretrained(base_stub, torch_dtype="auto", device_map="auto")
# decompressed_model at this point is just the base model, with dense weights

compressor: ModelCompressor = ModelCompressor.from_pretrained(compressed_save_dir)
compressor.decompress(model_path=compressed_save_dir, model=decompressed_model)
# now decompressed_model has decompressed weights



decompressed_state_dict = decompressed_model.state_dict()
uncompressed_state_dict = uncompressed_model.state_dict()

# check that the params in decompressed model is the same as the original model
for key in decompressed_state_dict.keys():
    assert key in uncompressed_state_dict.keys()
    decompressed_tensor = decompressed_state_dict[key]
    uncompressed_tensor = uncompressed_state_dict[key]

    assert torch.equal(decompressed_tensor, uncompressed_tensor), f"Tensor {key} is not equal."

print("All parameters in the decompressed model are the same as the original model.")
print("Inference on the decompressed model")

# Confirm generations of the quantized model look sane.

print("\n\n")
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = decompressed_model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")


"""
Output from above:

All parameters in the decompressed model are the same as the original model.
Inference on the decompressed model



========== SAMPLE GENERATION ==============
<s> Hello my name is John. I am a student at the University of California. I am a student at the University of California. I am a student at the University of California. I am a student at the University of California. I am a student at the University of California. I am a student at the University of California. I am a student at the University of California. I am a student at the University of California. I am a student at the University of California. I am a student at the University of
==========================================
"""