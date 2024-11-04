from vllm import LLM, SamplingParams
import torch

# model_path = "/home/rahul/vllm/TinyLlama-1.1B-Chat-v1.0-pruned_50.2of4-compressed"
# model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model_path = "/home/rahul/vllm/TinyLlama-1.1B-Chat-v1.0-pruned_50.2of4-FP8-compressed"
# model_path = "/home/rahul/vllm/TinyLlama-1.1B-Chat-v1.0-pruned_50.2of4-uncompressed"

model_path = "/home/rahul/llm-compressor/TinyLlama-1.1B-Chat-v1.0-pruned_50.2of4-uncompressed-compressed"
model = LLM(
    model=model_path, 
    enforce_eager=True,
    dtype=torch.bfloat16,
    tensor_parallel_size=1,
    )

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    )
outputs = model.generate(
    "Hello my name is,",
    sampling_params=sampling_params,
    )

print(outputs[0].outputs[0].text)
