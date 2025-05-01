import numpy
import torch

from vllm import LLM, SamplingParams

prompts = ["The Swiss Alps are", "The president of the USA is", "The Boston Bruins are"]

# Create a sampling params object for greedy sampling
sampling_params = SamplingParams(temperature=0.80, top_p=0.95, max_tokens=40, min_tokens=10)
llm  = LLM('nm-testing/llama2.c-stories110M-FP4', enforce_eager=True)


# Print the outputs.
output = llm.generate(prompts, sampling_params)
for o in output:
    print(o.outputs[0].text)
    print("\n")
    
