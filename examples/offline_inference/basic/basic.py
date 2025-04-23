# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
#llm = LLM(model="facebook/opt-125m")
model = "deepseek-ai/DeepSeek-V2-Lite"
llm = LLM(model=model,
          #model="mgoin/DeepSeek-Coder-V2-Lite-Instruct-FP8",
          #data_parallel_size = 1,
          #enable_expert_parallel = True,
          data_parallel_size = 2,
          enable_expert_parallel = False,
          #tensor_parallel_size=2,
          trust_remote_code=True,
          enforce_eager=True,
          #tensor_parallel_size=2,
          gpu_memory_utilization=0.60)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
print("\nGenerated Outputs:\n" + "-" * 60)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt:    {prompt!r}")
    print(f"Output:    {generated_text!r}")
    print("-" * 60)