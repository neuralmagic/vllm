
import numpy
import torch

from vllm import LLM, SamplingParams

prompts = [
    "The Swiss Alps are", "The president of the USA is",
    "The Boston Bruins are"
]

# Create a sampling params object for greedy sampling
sampling_params = SamplingParams(temperature=0.80, top_p=0.95, max_tokens=40, min_tokens=10)
#llm  = LLM('nm-testing/Llama-3.1-8B-Instruct-FP4-Weight')
#llm = LLM("/home/dsikka/llm-compressor/examples/quantization_w4a16_fp4/TinyLlama-1.1B-Chat-v1.0-FP4")
#llm = LLM("/home/dsikka/llm-compressor/examples/quantization_w4a16_fp4/Llama-3.1-8B-Instruct-NVFP4A16")
#llm = LLM("/home/dsikka/llm-compressor/examples/quantization_w4a16_fp4/Llama-3.1-8B-Instruct-NVFP4A16-MSE")
#llm = LLM("nm-testing/Llama-3.3-70B-Instruct-NVFP4A16", max_model_len=4096)
# Print the outputs.
llm = LLM("nvidia/Llama-3.3-70B-Instruct-FP4", max_model_len=4096, quantization="nvfp4", enforce_eager=True)
output = llm.generate(prompts, sampling_params)
for o in output:
    print(o.outputs[0].text)
    print("\n")
