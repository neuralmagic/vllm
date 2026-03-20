# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance-rate evaluation for MTP speculative decoding.

Supports both base-model-embedded MTP heads and standalone speculators
FastMTP checkpoints.

Usage examples::

    # Baseline: embedded MTP (no separate model)
    python examples/offline_inference/spec_decode_mtp_acceptance.py \\
        --model Qwen/Qwen3-Next-80B-A3B-Instruct --num-spec-tokens 3 --tp 8

    # Standalone speculators checkpoint
    python examples/offline_inference/spec_decode_mtp_acceptance.py \\
        --model Qwen/Qwen3-Next-80B-A3B-Instruct \\
        --mtp-model inference-optimization/Qwen3-Next-80B-A3B-Instruct-MTP-ultrachat-epoch1 \\
        --num-spec-tokens 3 --tp 8
"""

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.benchmarks.datasets import add_dataset_parser, get_samples
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.metrics.reader import Counter, Vector


def parse_args():
    parser = FlexibleArgumentParser()
    add_dataset_parser(parser)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base model path or HuggingFace repo ID.",
    )
    parser.add_argument(
        "--mtp-model",
        type=str,
        default=None,
        help=(
            "Speculators FastMTP checkpoint path or HuggingFace repo ID. "
            "If omitted, uses the MTP head embedded in the base model."
        ),
    )
    parser.add_argument("--num-spec-tokens", type=int, default=3)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-output-len", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--print-output", action="store_true")
    return parser.parse_args()


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    prompts = get_samples(args, tokenizer)
    llm_prompts = [
        {
            "prompt_token_ids": tokenizer.encode(
                p.prompt, add_special_tokens=False
            )
        }
        for p in prompts
    ]

    speculative_config: dict = {
        "method": "mtp",
        "num_speculative_tokens": args.num_spec_tokens,
    }
    if args.mtp_model is not None:
        speculative_config["model"] = args.mtp_model

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=speculative_config,
        disable_log_stats=False,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temp, max_tokens=args.max_output_len
    )
    outputs = llm.generate(llm_prompts, sampling_params=sampling_params)

    if args.print_output:
        for i, output in enumerate(outputs):
            print("-" * 50)
            print(f"prompt: {prompts[i].prompt}")
            print(f"generated text: {output.outputs[0].text}")

    metrics = llm.get_metrics()

    total_output_tokens = sum(
        len(o.outputs[0].token_ids) for o in outputs
    )
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * args.num_spec_tokens

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    print("-" * 50)
    mtp_label = args.mtp_model if args.mtp_model else "(embedded)"
    print(f"model:              {args.model}")
    print(f"mtp_model:          {mtp_label}")
    print(f"num_spec_tokens:    {args.num_spec_tokens}")
    print(f"total_output_tokens:{total_output_tokens}")
    print(f"num_drafts:         {num_drafts}")
    print(f"num_draft_tokens:   {num_draft_tokens}")
    print(f"num_accepted_tokens:{num_accepted_tokens}")
    acceptance_length = (
        1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1.0
    )
    print(f"mean acceptance length: {acceptance_length:.3f}")
    print("-" * 50)
    for i in range(len(acceptance_counts)):
        rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0.0
        print(f"  acceptance @ token {i}: {rate:.3f}")

    return acceptance_length


if __name__ == "__main__":
    args = parse_args()
    args.enable_multimodal_chat = False
    main(args)
