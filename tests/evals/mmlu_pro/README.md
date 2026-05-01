# MMLU-Pro Evaluation

This directory contains a helper script for MMLU-Pro evaluation with vLLM.

## Features

- **Category Selection**: Evaluate on all MMLU-Pro categories or specific ones (e.g., math, physics, chemistry)
- **Few-shot Learning**: Configurable number of few-shot examples
- **Online and Offline Modes**: Test with vLLM server or offline LLM object

## Quick Start

### List Available Categories

```bash
python tests/evals/mmlu_pro/mmlu_pro_eval.py --list-categories
```

### Run Evaluation (Server Mode)

First, start a vLLM server:

```bash
vllm serve Qwen/Qwen3-4B --max-model-len 4096
```

Then run evaluation:

```bash
# Evaluate all categories
python tests/evals/mmlu_pro/mmlu_pro_eval.py \
    --num-questions 100 \
    --num-shots 5

# Evaluate specific category (e.g., math)
python tests/evals/mmlu_pro/mmlu_pro_eval.py \
    --num-questions 50 \
    --num-shots 5 \
    --category math

# Save results to file
python tests/evals/mmlu_pro/mmlu_pro_eval.py \
    --num-questions 100 \
    --category physics \
    --save-results results.json
```

## Available Categories

Common MMLU-Pro categories include:

- `math` - Mathematics
- `physics` - Physics
- `chemistry` - Chemistry
- `biology` - Biology
- `computer science` - Computer Science
- `economics` - Economics
- `engineering` - Engineering
- `health` - Health
- `history` - History
- `law` - Law
- `philosophy` - Philosophy
- `psychology` - Psychology
- And more...

Use `--list-categories` to see the full list.

## API Reference

### Standalone Evaluation

```python
from tests.evals.mmlu_pro.mmlu_pro_eval import evaluate_mmlu_pro, evaluate_mmlu_pro_offline

# Server mode
results = evaluate_mmlu_pro(
    num_questions=100,
    num_shots=5,
    category="math",
    host="http://127.0.0.1",
    port=8000,
)

# Offline mode
from vllm import LLM
llm = LLM(model="Qwen/Qwen3-4B")
results = evaluate_mmlu_pro_offline(
    llm,
    num_questions=100,
    num_shots=5,
    category="math",
)

print(f"Accuracy: {results['accuracy']:.3f}")
```

## Requirements

Install the datasets library:

```bash
pip install datasets
```

## Comparison with lm-eval-harness

This implementation provides:

- Better control over evaluation parameters
- Support for category-specific evaluation
- Faster execution through async requests
- Direct integration with vLLM server and offline LLM
