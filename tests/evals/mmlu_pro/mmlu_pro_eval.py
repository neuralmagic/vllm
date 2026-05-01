#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Isolated MMLU-Pro evaluation script for vLLM serve endpoint.
"""

import argparse
import asyncio
import json
import os
import time

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm

INVALID = -9999999


def download_and_cache_file(url: str, filename: str | None = None) -> str:
    """Download and cache a file from a URL."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)

    return filename


def load_mmlu_pro_data(
    category: str | None = None,
    cache_dir: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load MMLU-Pro validation and test data.

    Args:
        category: Optional category to filter (e.g., 'math', 'physics', etc.)
                 If None, loads all categories.
        cache_dir: Optional cache directory for datasets.

    Returns:
        Tuple of (validation_data, test_data) lists
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package is required for MMLU-Pro evaluation. "
            "Install with: pip install datasets"
        ) from None

    cache_dir = cache_dir or os.path.join("/tmp", "mmlu_pro_cache")

    # Load the dataset from Hugging Face
    dataset = load_dataset(
        "TIGER-Lab/MMLU-Pro",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # MMLU-Pro has 'validation' and 'test' splits
    validation_data = list(dataset["validation"])
    test_data = list(dataset["test"])

    # Filter by category if specified
    if category:
        validation_data = [d for d in validation_data if d.get("category") == category]
        test_data = [d for d in test_data if d.get("category") == category]

        if not test_data:
            available_cats = sorted(set(d.get("category") for d in dataset["test"]))
            raise ValueError(
                f"Category '{category}' not found. "
                f"Available categories: {available_cats}"
            )

    return validation_data, test_data


def get_available_categories(cache_dir: str | None = None) -> list[str]:
    """Get list of available MMLU-Pro categories."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    cache_dir = cache_dir or os.path.join("/tmp", "mmlu_pro_cache")
    dataset = load_dataset(
        "TIGER-Lab/MMLU-Pro",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    categories = sorted(set(d.get("category") for d in dataset["test"]))
    return categories


def format_mmlu_pro_question(example: dict) -> str:
    """Format MMLU-Pro question with options."""
    question = example["question"]
    options = example["options"]

    formatted = f"{question}\n\n"
    for i, option in enumerate(options):
        formatted += f"{chr(65 + i)}. {option}\n"

    return formatted


def get_answer_letter(example: dict) -> str:
    """Get the answer letter (A, B, C, ...) from the example."""
    answer_idx = example["answer_index"]
    return chr(65 + answer_idx)


def extract_answer_letter(response: str) -> str | int:
    """Extract answer letter from model response.

    Looks for patterns like:
    - "The answer is A"
    - "Answer: B"
    - Just "C" at the end
    """
    response = response.strip().upper()

    # Look for explicit answer patterns
    import regex as re

    patterns = [
        r"ANSWER\s*IS\s*([A-J])",
        r"ANSWER\s*:\s*([A-J])",
        r"THE\s+ANSWER\s+IS\s+([A-J])",
        r"\b([A-J])\s*$",  # Single letter at end
        r"^\s*([A-J])\s*\.",  # Letter at start with period
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)

    # If no pattern found, look for any single capital letter
    letters = re.findall(r"\b([A-J])\b", response)
    if letters:
        return letters[-1]  # Take the last one

    return INVALID


async def call_vllm_api(
    session: aiohttp.ClientSession,
    prompt: str,
    temperature: float,
    max_tokens: int,
    stop: list[str] | None = None,
    url: str | None = None,
    seed: int | None = None,
) -> tuple[str, int]:
    """Call vLLM's OpenAI-compatible completions endpoint.

    Returns:
        Tuple of (response_text, completion_tokens)
    """
    data = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
    }
    if seed is not None:
        data["seed"] = seed

    try:
        async with session.post(f"{url}/v1/completions", json=data) as response:
            response.raise_for_status()
            result = await response.json()
            text = result["choices"][0]["text"]
            completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
            return text, completion_tokens
    except Exception as e:
        print(f"Error calling vLLM API: {e}")
        return "", 0


def _build_mmlu_pro_prompts(
    num_questions: int = 0,
    num_shots: int = 5,
    category: str | None = None,
) -> tuple[list[str], list[str]]:
    """Build few-shot MMLU-Pro completion prompts and ground-truth labels.

    Args:
        num_questions: Number of questions to evaluate (0 = all)
        num_shots: Number of few-shot examples
        category: Optional category filter

    Returns:
        Tuple of (prompts, labels) where labels are answer letters (A, B, C, ...)
    """
    validation_data, test_data = load_mmlu_pro_data(category=category)

    if num_questions == 0:
        num_questions = len(test_data)
    else:
        num_questions = min(num_questions, len(test_data))

    if num_shots > len(validation_data):
        print(
            f"Warning: Requested {num_shots} shots but only {len(validation_data)} "
            f"validation examples available. Using {len(validation_data)} shots."
        )
        num_shots = len(validation_data)

    # Build few-shot examples from validation set
    few_shot_examples = ""
    for i in range(num_shots):
        example = validation_data[i]
        question = format_mmlu_pro_question(example)
        answer = get_answer_letter(example)
        few_shot_examples += f"{question}\nAnswer: {answer}\n\n"

    # Build prompts and labels for test set
    prompts = []
    labels = []
    for i in range(num_questions):
        example = test_data[i]
        question = format_mmlu_pro_question(example)
        prompts.append(few_shot_examples + f"{question}\nAnswer:")
        labels.append(get_answer_letter(example))

    return prompts, labels


def _score_mmlu_pro(
    states: list[str],
    output_tokens: list[int],
    labels: list[str],
    num_shots: int,
    max_tokens: int,
    latency: float,
    category: str | None = None,
) -> dict[str, float | int | str]:
    """Score MMLU-Pro responses and return a results dict."""
    num_questions = len(labels)
    preds = [extract_answer_letter(state) for state in states]

    # Calculate accuracy
    correct = [pred == label for pred, label in zip(preds, labels)]
    accuracy = np.mean(correct)
    invalid_rate = np.mean([pred == INVALID for pred in preds])
    total_output_tokens = sum(output_tokens)
    tokens_per_second = total_output_tokens / latency if latency > 0 else 0.0

    return {
        "accuracy": accuracy,
        "invalid_rate": invalid_rate,
        "latency": latency,
        "questions_per_second": num_questions / latency if latency > 0 else 0.0,
        "total_output_tokens": total_output_tokens,
        "tokens_per_second": tokens_per_second,
        "num_questions": num_questions,
        "num_shots": num_shots,
        "max_tokens": max_tokens,
        "category": category or "all",
        "timestamp": time.time(),
    }


def evaluate_mmlu_pro(
    num_questions: int = 0,
    num_shots: int = 5,
    max_tokens: int = 256,
    host: str = "http://127.0.0.1",
    port: int = 8000,
    temperature: float = 0.0,
    seed: int | None = 42,
    category: str | None = None,
) -> dict[str, float | int | str]:
    """
    Evaluate MMLU-Pro accuracy using vLLM serve endpoint.

    Args:
        num_questions: Number of questions to evaluate (0 = all)
        num_shots: Number of few-shot examples
        max_tokens: Max tokens for generation
        host: Server host
        port: Server port
        temperature: Generation temperature
        seed: Random seed
        category: Optional category filter (e.g., 'math', 'physics')

    Returns dict with accuracy, invalid_rate, latency, etc.
    """
    base_url = f"{host}:{port}"
    prompts, labels = _build_mmlu_pro_prompts(num_questions, num_shots, category)
    num_questions = len(prompts)

    async def run_async_evaluation():
        states: list[str] = [""] * num_questions
        output_tokens: list[int] = [0] * num_questions

        async def get_answer(session: aiohttp.ClientSession, i: int) -> tuple[str, int]:
            answer, tokens = await call_vllm_api(
                session=session,
                prompt=prompts[i],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["\n\n", "Question:", "<|separator|>"],
                url=base_url,
                seed=seed,
            )
            states[i] = answer
            output_tokens[i] = tokens
            return answer, tokens

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)
        ) as session:
            tasks = [get_answer(session, i) for i in range(num_questions)]
            await tqdm.gather(*tasks, desc="Evaluating")

        return states, output_tokens

    category_str = f" (category: {category})" if category else ""
    print(
        f"Running MMLU-Pro evaluation: {num_questions} questions, "
        f"{num_shots}-shot{category_str}"
    )

    tic = time.perf_counter()
    states, output_tokens = asyncio.run(run_async_evaluation())
    latency = time.perf_counter() - tic

    return _score_mmlu_pro(
        states, output_tokens, labels, num_shots, max_tokens, latency, category
    )


def evaluate_mmlu_pro_offline(
    llm,
    num_questions: int = 0,
    num_shots: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.0,
    category: str | None = None,
) -> dict[str, float | int | str]:
    """Evaluate MMLU-Pro accuracy using an offline vllm.LLM object.

    Same prompts and scoring as evaluate_mmlu_pro(), but runs generation
    directly via llm.generate() instead of calling a server over HTTP.
    """
    from vllm import SamplingParams

    prompts, labels = _build_mmlu_pro_prompts(num_questions, num_shots, category)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["\n\n", "Question:", "<|separator|>"],
    )

    category_str = f" (category: {category})" if category else ""
    print(
        f"Running offline MMLU-Pro evaluation: {len(prompts)} questions, "
        f"{num_shots}-shot{category_str}"
    )

    tic = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    latency = time.perf_counter() - tic

    states = [o.outputs[0].text for o in outputs]
    output_tokens = [len(o.outputs[0].token_ids) for o in outputs]

    return _score_mmlu_pro(
        states, output_tokens, labels, num_shots, max_tokens, latency, category
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MMLU-Pro evaluation for vLLM serve")
    parser.add_argument(
        "--num-shots", type=int, default=5, help="Number of few-shot examples"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=0,
        help="Number of questions to evaluate (0 = all)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Max tokens for generation"
    )
    parser.add_argument("--host", type=str, default="http://127.0.0.1", help="Host URL")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for generation"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help=(
            "MMLU-Pro category to evaluate (e.g., 'math', 'physics'). "
            "If not specified, evaluates all categories."
        ),
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available MMLU-Pro categories and exit",
    )
    parser.add_argument("--save-results", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    if args.list_categories:
        categories = get_available_categories()
        print("Available MMLU-Pro categories:")
        for cat in categories:
            print(f"  - {cat}")
        return

    result = evaluate_mmlu_pro(
        num_questions=args.num_questions,
        num_shots=args.num_shots,
        max_tokens=args.max_tokens,
        host=args.host,
        port=args.port,
        temperature=args.temperature,
        seed=args.seed,
        category=args.category,
    )

    # Print results to terminal
    print("\nResults:")
    print(f"Category: {result['category']}")
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(f"Invalid responses: {result['invalid_rate']:.3f}")
    print(f"Total latency: {result['latency']:.3f} s")
    print(f"Questions per second: {result['questions_per_second']:.3f}")
    print(f"Total output tokens: {result['total_output_tokens']}")
    print(f"Output tokens per second: {result['tokens_per_second']:.3f}")

    # Optional file saving
    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.save_results}")


if __name__ == "__main__":
    main()
