import enum
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import msgspec
import numpy.typing as npt

from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalKwargs, MultiModalPlaceholderDict
from vllm.sampling_params import RequestOutputKind, SamplingParams


@dataclass
class DetokenizerRequest:

    request_id: str
    prompt: Optional[str]
    prompt_token_ids: List[int]
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    output_kind: RequestOutputKind

    stop: List[str]
    include_stop_str_in_output: bool

    # Per-request logprobs & prompt logprobs
    # counts; None is equivalent to 0
    logprobs: Optional[int]
    prompt_logprobs: Optional[int]


@dataclass
class EngineCoreRequest:

    # NOTE: prompt and prompt_token_ids should be DecoderOnlyInput,
    # but this object is currently not playing well with msgspec
    # due to circular imports and typing we have in data.py

    request_id: str
    #NOTE(Nick): I don't think we need to pass prompt here since it should
    # always be tokenized?
    prompt: Optional[str]
    prompt_token_ids: List[int]
    mm_inputs: Optional[List[Optional[MultiModalKwargs]]]
    mm_hashes: Optional[List[str]]
    mm_placeholders: Optional[MultiModalPlaceholderDict]
    sampling_params: SamplingParams
    eos_token_id: Optional[int]
    arrival_time: float
    lora_request: Optional[LoRARequest]


class EngineCoreOutput(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    request_id: str
    new_token_ids: List[int]
    finished: bool
    logprobs: Optional[List[Tuple[npt.NDArray, npt.NDArray]]]
    prompt_logprobs: Optional[npt.NDArray]
    prompt_logprobs_token_ids: Optional[npt.NDArray]
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None


class EngineCoreOutputs(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False):  # type: ignore[call-arg]

    #NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout and using an int enum for finish/stop reason

    # [num_reqs]
    outputs: List[EngineCoreOutput]


@dataclass
class EngineCoreProfile:
    is_start: bool


class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """
    ADD = b'\x00'
    ABORT = b'\x01'
    PROFILE = b'\x02'


EngineCoreRequestUnion = Union[EngineCoreRequest, EngineCoreProfile, List[str]]
