import argparse
import torch
import sys
import json
import inspect

from dataclasses import dataclass, asdict
from typing import Optional
from vllm import LLM, SamplingParams
from vllm.profiler import nm_profile

BATCH_SIZE_DEFAULT = 1
PROMPT_LEN_DEFAULT = 256
MAX_SEQ_LEN_DEFAULT = 1024


@dataclass
class ProfileContext:
    model_name: str
    model_revision: str
    is_sparse: bool
    quant_method: str
    max_seq_len: int
    max_num_batched_tokens: int
    prompt_len: int
    batch_size: int
    num_gpus: int
    allow_cuda_graphs: bool


def run_profile(context: ProfileContext, csv_output: Optional[str],
                json_output: Optional[str]):
    print("Run profile with:")
    for key, value in asdict(context).items():
        print(f"  {key} = {value}")

    # Create sampling params
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=8)

    # Create LLM
    llm = LLM(
        model=context.model_name,
        revision=context.model_revision,
        sparsity="sparse_w16a16" if context.is_sparse else None,
        enforce_eager=not context.allow_cuda_graphs,
        tensor_parallel_size=context.num_gpus,
        gpu_memory_utilization=0.9,
        max_model_len=context.max_seq_len,
        quantization=context.quant_method,
        max_num_batched_tokens=context.max_num_batched_tokens,
    )

    batch_size = context.batch_size
    prompt_len = context.prompt_len

    max_num_batched_tokens = llm.llm_engine.scheduler_config.max_num_batched_tokens
    max_num_seqs = llm.llm_engine.scheduler_config.max_num_seqs
    if batch_size * prompt_len > max_num_batched_tokens:
        print(
            f"ERROR: chosen batch_size * prompt_len "
            f"({batch_size} * {prompt_len} = {batch_size * prompt_len}) is larger "
            f"than max_num_batched_tokens ({max_num_batched_tokens}) and therefore "
            f"cannot be run in a single profile step, please choose a smaller batch "
            f"size or prompt length, or increase --max_num_batched_tokens")
        sys.exit(-1)
    if batch_size >= max_num_seqs:
        print(
            f"ERROR: chosen batch_size ({batch_size}) is larger than max_num_seqs "
            f"({max_num_seqs}) and therefore cannot be run in a single profile step"
            f", please choose a smaller batch size")
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

    if csv_output:
        csv_filename_base = csv_output.rstrip(".csv")
        prefill_results.export_model_stats_table_csv(
            csv_filename_base + "_prefill_model_table.csv")
        prefill_results.export_summary_stats_table_csv(
            csv_filename_base + "_prefill_summary_table.csv")
        decode_results.export_model_stats_table_csv(\
            csv_filename_base + "_decode_model_table.csv")
        decode_results.export_summary_stats_table_csv(
            csv_filename_base + "_decode_summary_table.csv")

    if json_output:
        cuda_devices = [
            torch.cuda.get_device_properties(dev_idx)
            for dev_idx in range(torch.cuda.device_count())
        ]

        json_dict = {
            "context": {
                "python_version": f"{sys.version}",
                "torch_version": f"{torch.__version__}",
                "torch_cuda_version": f"{torch.version.cuda}",
                "cuda_devices": f"{cuda_devices}",
                **asdict(context)
            },
            "prefill": prefill_results.convert_stats_to_dict(),
            "decode": decode_results.convert_stats_to_dict()
        }

        with open(json_output.rstrip(".json") + ".json", "w+") as f:
            json.dump(json_dict, f, indent=2)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument('--is_sparse', action='store_true')
    parser.add_argument("--quant_method", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN_DEFAULT)
    parser.add_argument("--max_num_batched_tokens", type=int, default=None)
    parser.add_argument("--prompt_len", type=int, default=PROMPT_LEN_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--allow_cuda_graphs', action='store_true')

    args = parser.parse_args()

    context = ProfileContext(
        **{
            k: v
            for k, v in vars(args).items()
            if k in inspect.signature(ProfileContext).parameters
        })
    run_profile(context, csv_output=args.csv, json_output=args.json)
