import argparse
import torch
import sys

from vllm import LLM, SamplingParams
from vllm.profiler import nm_profile

BATCH_SIZE_DEFAULT = 1
PROMPT_LEN_DEFAULT = 256
MAX_SEQ_LEN_DEFAULT = 1024


def run_profile(model_name, model_revision, csv_output, is_sparse,
                quant_method, max_seq_len, prompt_len, batch_size, num_gpus,
                allow_cuda_graphs):
    print("Run profile with:")
    print(f"  model_name = {model_name}")
    print(f"  model_revision = {model_revision}")
    print(f"  is_sparse = {is_sparse}")
    print(f"  quant_method = {quant_method}")
    print(f"  max_seq_len = {max_seq_len}")
    print(f"  prompt_len = {prompt_len}")
    print(f"  batch_size = {batch_size}")
    print(f"  num_gpus = {num_gpus}")
    print(f"  allow_cuda_graphs = {allow_cuda_graphs}")

    # Create sampling params
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=8)

    # Create LLM
    llm = LLM(
        model=model_name,
        revision=model_revision,
        sparsity="sparse_w16a16" if is_sparse else None,
        enforce_eager=not allow_cuda_graphs,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=0.9,
        max_model_len=max_seq_len,
        quantization=quant_method,
    )

    max_num_batched_tokens = llm.llm_engine.scheduler_config.max_num_batched_tokens
    max_num_seqs = llm.llm_engine.scheduler_config.max_num_seqs
    if batch_size * prompt_len > max_num_batched_tokens:
        print(
            f"ERROR: chosen batch_size * prompt_len ({batch_size} * {prompt_len} = {batch_size * prompt_len}) is larger than max_num_batched_tokens ({max_num_batched_tokens}) and therefore cannot be run in a single profile step, please choose a smaller batch size or prompt length"
        )
        sys.exit(-1)
    if batch_size >= max_num_seqs:
        print(
            f"ERROR: chosen batch_size ({batch_size}) is larger than max_num_seqs ({max_num_seqs}) and therefore cannot be run in a single profile step, please choose a smaller batch size"
        )
        sys.exit(-1)

    for i in range(batch_size):
        llm.llm_engine.add_request(
            request_id=f"seq{i}",
            prompt=None,
            prompt_token_ids=torch.randint(
                0,
                llm.llm_engine.model_config.get_vocab_size() // 2,
                size=(prompt_len, )).tolist(),
            sampling_params=sampling_params)

    with nm_profile() as prefill_prof:
        llm.llm_engine.step()  # First step is prefill

    with nm_profile() as decode_prof:
        llm.llm_engine.step()

    prefill_results = prefill_prof.results
    decode_results = decode_prof.results

    print("=" * 80)
    print(
        f"= Prefill Model Table (prompt_len={prompt_len}, batch_size={batch_size})"
    )
    print("=" * 80)
    print()
    prefill_results.print_model_table()
    print()
    print("=" * 80)
    print(
        f"= Decode Model Table (prompt_len={prompt_len}, batch_size={batch_size})"
    )
    print("=" * 80)
    print()
    decode_results.print_model_table()
    print()
    print("=" * 80)
    print(
        f"= Prefill Summary Table (prompt_len={prompt_len}, batch_size={batch_size})"
    )
    print("=" * 80)
    print()
    prefill_results.print_summary_table()
    print()
    print("=" * 80)
    print(
        f"= Decode Summary Table (prompt_len={prompt_len}, batch_size={batch_size})"
    )
    print("=" * 80)
    print()
    decode_results.print_summary_table()

    csv_filename_base = csv_output.replace(".csv", "")
    if csv_output:
        prefill_results.export_model_table_csv(csv_filename_base +
                                               "_prefill_model_table.csv")
        prefill_results.export_summary_table_csv(csv_filename_base +
                                                 "_prefill_summary_table.csv")
        decode_results.export_model_table_csv(csv_filename_base +
                                              "_decode_model_table.csv")
        decode_results.export_summary_table_csv(csv_filename_base +
                                                "_decode_summary_table.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument('--is_sparse', action='store_true')
    parser.add_argument("--quant_method", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN_DEFAULT)
    parser.add_argument("--prompt_len", type=int, default=PROMPT_LEN_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--allow_cuda_graphs', action='store_true')

    args = parser.parse_args()

    run_profile(model_name=args.model_name,
                model_revision=args.model_revision,
                csv_output=args.csv,
                is_sparse=args.is_sparse,
                quant_method=args.quant_method,
                max_seq_len=args.max_seq_len,
                prompt_len=args.prompt_len,
                batch_size=args.batch_size,
                num_gpus=args.num_gpus,
                allow_cuda_graphs=args.allow_cuda_graphs)
