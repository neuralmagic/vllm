from vllm import LLM, SamplingParams

prompts = [
    "A robot may not injure a human being",
    "It is only with the heart that one can see rightly;",
    "The greatest glory in living lies not in never falling,",
]
answers = [
    " or, through inaction, allow a human being to come to harm.",
    " what is essential is invisible to the eye.",
    " but in rising every time we fall.",
]
N = 1
# Currently, top-p sampling is disabled. `top_p` should be 1.0.
sampling_params = SamplingParams(temperature=0.7,
                                 top_p=1.0,
                                 n=N,
                                 max_tokens=16)

# Set `enforce_eager=True` to avoid ahead-of-time compilation.
# In real workloads, `enforce_eager` should be `False`.
llm = LLM(
    # model="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
    # model="neuralmagic/gemma-2-2b-it-quantized.w8a16",
    model="neuralmagic/SmolLM-1.7B-Instruct-quantized.w8a16",
    enforce_eager=True,
    max_model_len=1024)
outputs = llm.generate(prompts, sampling_params)
for output, answer in zip(outputs, answers):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    # assert generated_text.startswith(answer)
