# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

prompts = [
    "The Swiss Alps are", "The president of the USA is",
    "The Boston Bruins are"
]

# Create a sampling params object for greedy sampling
sampling_params = SamplingParams(temperature=0.90,
                                 max_tokens=40,
                                 min_tokens=10)
#llm = LLM('/home/dsikka/llm-compressor/examples/quantization_w4a16_fp4/TinyLlama-1.1B-Chat-v1.0-FP4', enforce_eager=True)
llm = LLM(
    "/home/dsikka/llm-compressor/examples/quantization_w4a16_fp4/Llama-3.1-8B-Instruct-FP4",
    enforce_eager=True)

#llm = LLM("nvidia/Llama-3.3-70B-Instruct-FP4", quantization='nvfp4', max_model_len=2048, enforce_eager=True)
#llm = LLM("nm-testing/llama2.c-stories110M-FP4", enforce_eager=True)
# Print the outputs.
output = llm.generate(prompts, sampling_params)
for o in output:
    print(o.outputs[0].text)
    print("\n")
