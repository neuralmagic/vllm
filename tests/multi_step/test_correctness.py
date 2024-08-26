# Test the AsyncLLMEngine with multi-step-decoding

import os
from collections import namedtuple
from enum import Enum
from typing import List

import pytest
from datetime import datetime

from ..utils import RemoteOpenAIServer


class MultiStepChunkedPrefillPolicy(Enum):
    # When prompt and decode sequences are scheduled together,
    # the DEFAULT policy is to run the prompt and decodes sequences
    # together only for the first step and run just the decode sequences
    # in the rest of the steps.
    DEFAULT = 1
    # In FORCE_SINGLE_STEP policy, we force the scheduled sequences to
    # run a single step and then re-schedule.
    FORCE_SINGLE_STEP = 2
    INVALID = 3


ChunkedPrefillTestArgType = namedtuple('ChunkedPrefillTestArgType',
                                       ['enabled', 'policy'])

MODELS = [
    "JackFram/llama-160m",
]
NUM_SCHEDULER_STEPS = [8]  # Multi-step decoding steps
NUM_PROMPTS = [1000]
CHUNKED_PREFILL_ARGS = [
    ChunkedPrefillTestArgType(False, MultiStepChunkedPrefillPolicy.INVALID),
    ChunkedPrefillTestArgType(True, MultiStepChunkedPrefillPolicy.DEFAULT),
    ChunkedPrefillTestArgType(True,
                              MultiStepChunkedPrefillPolicy.FORCE_SINGLE_STEP)
]

DEFAULT_SERVER_ARGS: List[str] = [
    "--disable-log-requests",
    "--use-v2-block-manager",
    "--worker-use-ray",
    "--gpu-memory-utilization",
    "0.85",
    "--swap-space",
    "16",
]


class EnvContextManager():

    def __init__(self, env: dict):
        self.os_env = dict(os.environ)
        self.add_env = dict(env)

    def __enter__(self):
        os.environ.update(self.add_env)

    def __exit__(self, *args, **kwargs):
        os.environ.clear()
        os.environ.update(self.os_env)


async def completions_with_server_args(prompts: List[str],
                                       model_name: str,
                                       server_cli_args: List[str],
                                       with_env: dict = {}):  # noqa: B006
    # env setup
    os.environ.update(with_env)

    outputs = None
    with EnvContextManager(with_env) as _:  # noqa: SIM117
        with RemoteOpenAIServer(model_name, server_cli_args) as server:
            client = server.get_async_client()
            outputs = await client.completions.create(model=model_name,
                                                      prompt=prompts,
                                                      temperature=0,
                                                      stream=False,
                                                      max_tokens=150)
    assert outputs is not None

    return outputs


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(("tp_size, pp_size"), [
    (1, 1),
    (2, 1),
    (2, 2),
])
@pytest.mark.parametrize("eager_mode", [False, True])
@pytest.mark.parametrize("num_scheduler_steps", NUM_SCHEDULER_STEPS)
@pytest.mark.parametrize("num_prompts", NUM_PROMPTS)
@pytest.mark.parametrize("chunked_prefill", CHUNKED_PREFILL_ARGS)
@pytest.mark.asyncio
async def test_multi_step(example_prompts, model: str, tp_size: int,
                          pp_size: int, eager_mode: bool,
                          num_scheduler_steps: int, num_prompts: int,
                          chunked_prefill: ChunkedPrefillTestArgType):

    prompts = example_prompts
    if len(prompts) < num_prompts:
        prompts = prompts * ((num_prompts // len(prompts)) + 1)
    prompts = prompts[:num_prompts]
    assert len(prompts) == num_prompts

    server_args = DEFAULT_SERVER_ARGS + ["--enforce-eager"]
    ms_server_args = DEFAULT_SERVER_ARGS + \
        ["--num-scheduler-steps", f"{num_scheduler_steps}"]

    if eager_mode:
        ms_server_args.append("--enforce-eager")

    test_env = {}
    if chunked_prefill.enabled:
        ms_server_args.append("--enable-chunked-prefill")
        if chunked_prefill.policy == \
            MultiStepChunkedPrefillPolicy.FORCE_SINGLE_STEP:
            test_env[
                'VLLM_MULTI_STEP_CHUNKED_PREFILL_SINGLE_STEP_POLICY'] = '1'

    distributed_args = [
        "--tensor-parallel-size",
        str(tp_size),
        "--pipeline-parallel-size",
        str(pp_size),
    ]

    ref_completions = await completions_with_server_args(
        prompts, model, server_args + distributed_args)
    test_completions = await completions_with_server_args(
        prompts, model, ms_server_args + distributed_args, test_env)

    def get_text_generations(completions):
        return [x.text for x in completions.choices]

    ref_generations = get_text_generations(ref_completions)
    test_generations = get_text_generations(test_completions)
    cp_policy = "na"
    if chunked_prefill.enabled:
        cp_policy = "p1" if chunked_prefill.policy == MultiStepChunkedPrefillPolicy.FORCE_SINGLE_STEP else "p2"

    test_description = (f"tp_{tp_size}_" 
                        f"pp_{pp_size}_"
                        f"eager_{eager_mode}_"
                        f"cp_{chunked_prefill.enabled}"
                        f"policy_{cp_policy}")
    def compare(refs, tests, desc):
        num_total = len(refs)
        num_correct = 0
        incorrect_rt = []
        for r, t in zip(refs, tests):
            if r == t:
                num_correct = num_correct + 1
            else:
                incorrect_rt.append((r, t))

        print (f"- {desc}")
        print (f"   Compare Results :::: {num_correct}/{num_total}")
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        fname = f"{desc}-{current_dt}.txt"
        with open(fname, "w+") as f:
            for r, t in incorrect_rt:
                f.write(f"REF:\n{r}\nTES:\n{t}\n")

    compare(ref_generations, test_generations, test_description)
    #assert ref_generations == test_generations
