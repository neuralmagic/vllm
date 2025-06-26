# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests compute_expert_num_tokens kernels
"""

import dataclasses
from typing import Optional

import pytest
import torch

from vllm import _custom_ops as ops


@dataclasses.dataclass
class TestTensors:

    topk_ids: torch.Tensor
    expert_map: Optional[torch.Tensor] = None

    def to_device(self, device: str):
        self.topk_ids = self.topk_ids.to(device=device)
        if self.expert_map is not None:
            self.expert_map = self.expert_map.to(device=device)

    @staticmethod
    def make(num_tokens: int, num_topk: int, num_experts: int,
             device: str) -> "TestTensors":

        # make topk ids
        topk_ids = torch.empty((num_tokens, num_topk),
                               device=device,
                               dtype=torch.int64)
        for x in range(num_tokens):
            topk_ids[x] = torch.randperm(num_experts)[:num_topk]
        topk_ids = topk_ids.to(dtype=torch.int64)
        return TestTensors(topk_ids=topk_ids)

    def with_ep_rank(self, ep_rank: int, num_global_experts: int,
                     num_local_experts: int, device: str):
        # make an expert map
        expert_map = torch.empty((num_global_experts),
                                 device=device,
                                 dtype=torch.int32)
        expert_map.fill_(-1)
        s = ep_rank * num_local_experts
        e = s + num_local_experts
        expert_map[s:e] = torch.tensor(list(range(num_local_experts)),
                                       device=device)

        # mark topk ids -1
        topk_ids = expert_map[self.topk_ids.clone()]
        return TestTensors(topk_ids=topk_ids, expert_map=expert_map)


def ref_impl(tt: TestTensors, expert_num_tokens: torch.Tensor,
             total_num_tokens: torch.Tensor):
    # do the reference in cpu
    tt.to_device("cpu")
    expert_ids, counts = tt.topk_ids.unique(return_counts=True)

    for eid, count in zip(expert_ids, counts):
        if eid == -1:
            continue
        if tt.expert_map is not None and tt.expert_map[eid] == -1:
            continue
        expert_num_tokens[eid] += count
        total_num_tokens[0] += count


#@pytest.mark.parametrize("num_tokens", [1,3,5])
#@pytest.mark.parametrize("num_topk", [list(range(1, 8192))])
#@pytest.mark.parametrize("num_experts", [32])
#@pytest.mark.parametrize("ep_size", [1, 2, 4])

#@pytest.mark.parametrize("num_tokens", [1, 128, 512])
#@pytest.mark.parametrize("num_topk", [2, 6, 12])
#@pytest.mark.parametrize("num_experts", [32])
#@pytest.mark.parametrize("ep_size", [1, 2, 4])


@pytest.mark.parametrize("num_tokens", [4])
@pytest.mark.parametrize("num_topk", [6])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("ep_size", [1])
def test_compute_expert_num_tokens(num_tokens: int, num_topk: int,
                                   num_experts: int, ep_size: int):

    assert num_topk <= num_experts

    tt = TestTensors.make(num_tokens, num_topk, num_experts, device="cpu")

    num_global_experts = num_experts
    assert num_global_experts % ep_size == 0
    num_local_experts = num_global_experts // ep_size
    for ep_rank in range(ep_size):
        tt_rank = tt.with_ep_rank(ep_rank, num_global_experts,
                                  num_local_experts, "cpu")

        ref_expert_num_tokens = torch.zeros((num_local_experts),
                                            device="cpu",
                                            dtype=torch.int32)
        ref_total_num_tokens = torch.zeros((1),
                                           device="cpu",
                                           dtype=torch.int32)

        ref_impl(tt_rank, ref_expert_num_tokens, ref_total_num_tokens)

        tt_rank.to_device("cuda")
        impl_expert_num_tokens = ref_expert_num_tokens.clone().to("cuda")
        impl_total_num_tokens = ref_total_num_tokens.clone().to("cuda")

        ops.compute_expert_num_tokens(tt_rank.topk_ids,
                                      impl_expert_num_tokens,
                                      impl_total_num_tokens,
                                      local_num_experts=num_local_experts,
                                      expert_map=tt_rank.expert_map)

        torch.cuda.synchronize()
        print(f"tt_rank {tt_rank.topk_ids} {tt_rank.expert_map}")
        print(f"ref_expert_num_tokens {ref_expert_num_tokens}")
        print(f"impl_expert_num_tokens {impl_expert_num_tokens}")

        torch.testing.assert_close(ref_expert_num_tokens,
                                   impl_expert_num_tokens.to("cpu"),
                                   atol=0,
                                   rtol=0)
        torch.testing.assert_close(ref_total_num_tokens,
                                   impl_total_num_tokens.to("cpu"),
                                   atol=0,
                                   rtol=0)
