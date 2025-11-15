from save_shards import save_shards_if_rank0

if __name__ == '__main__':
    from vllm import LLMEngine, SamplingParams, EngineArgs
    import torch
    from datasets import load_dataset

    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"

    # Select number of samples
    NUM_CALIBRATION_SAMPLES = 20

    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    ds = ds.shuffle(seed=42)

    # Create a sampling params object for greedy sampling
    sampling_params = SamplingParams(temperature=0.80, top_p=0.95, max_tokens=40, min_tokens=10)
    engine_args = EngineArgs(
        #model="/raid/engine/ml3-nvfp4a16",
        model="nm-testing/Llama-4-Scout-17B-16E-NVFP4A16",
        #model="nm-testing/TinyLlama-1.1B-Chat-v1.0-NVFP4A16-e2e",
        tensor_parallel_size=2,
        enforce_eager=True,
        max_model_len=2048,
        gpu_memory_utilization=0.95,
        #tokenizer_mode="mistral", 
        #config_format="mistral",
        #load_format="mistral"
    )
    llm = LLMEngine.from_engine_args(engine_args)
    counter = 0
    for item in ds:
        llm.add_request(
            request_id=str(counter),
            prompt=item["prompt"],
            params=sampling_params
        )
        counter += 1
    
    # Process requests
    outputs = {}
    step_count = 0
    while llm.has_unfinished_requests():
        step_outputs = llm.step()
        step_count += 1
        for output in step_outputs:
            if output.finished:
                outputs[output.request_id] = output

                generated_text = output.outputs[0].text
                #print(f"Generated: {generated_text}\n")

    # Save updated model
    llm.apply_model(save_shards_if_rank0)


