# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional
from importlib.util import find_spec

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.config import VllmConfig
import vllm.envs as envs
from vllm.distributed import (get_tp_group, tensor_model_parallel_all_reduce)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank)
from vllm.logger import init_logger

from .vllm_inductor_pass import VllmInductorPass

if envs.VLLM_USE_FLASHINFER_ALLREDUCE and find_spec("flashinfer"):
    import flashinfer.comm as flashinfer_comm
    flashinfer_comm = flashinfer_comm if hasattr(flashinfer_comm, "trtllm_allreduce_fusion") else None
else:
    flashinfer_comm = None

logger = init_logger(__name__)

ALLREDUCE_OP = torch.ops.vllm.all_reduce.default
RMS_OP = torch.ops._C.rms_norm.default
RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default

class BasePattern:

    def __init__(self, dtype: torch.dtype, device: str):
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()


class GEMMReduceScatterPattern(BasePattern):

    def get_inputs(self):
        mul = torch.empty([16, 4], device=self.device, dtype=self.dtype)
        mm_weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        return [mul, mm_weight]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(mul: torch.Tensor, mm_weight: torch.Tensor):
            mm = torch.ops.aten.mm.default(mul, mm_weight)
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name)
            return reduce_scatter

        def replacement(mul: torch.Tensor, mm_weight: torch.Tensor):
            gemm_rs = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                mul,
                mm_weight,
                "avg",
                scatter_dim=0,
                group_name=self.tp.device_group.group_name,
            )

            return gemm_rs

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class AllGatherGEMMPattern(BasePattern):

    def get_inputs(self):
        x = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        return [x, weight]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_gather = torch.ops.vllm.all_gather.default(
                x,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name)

            return torch.ops.aten.mm.default(all_gather, weight)

        def replacement(
                x: torch.Tensor,
                weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
                x,
                [weight],
                gather_dim=0,
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class AsyncTPPass(VllmInductorPass):

    def __init__(self, config: VllmConfig):
        super().__init__(config)

        # Enable symmetric memory for the TP process group
        enable_symm_mem_for_group(get_tp_group().device_group.group_name)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="async_tp_pass")
        GEMMReduceScatterPattern(self.model_dtype,
                                 self.device).register(self.patterns)

        AllGatherGEMMPattern(self.model_dtype,
                             self.device).register(self.patterns)

    def is_applicable_for_shape(self, shape: Optional[int]) -> bool:
        # only do replace for specific shapes
        tp_size = get_tensor_model_parallel_world_size()
        return shape is not None and shape % tp_size == 0

    def __call__(self, graph: fx.Graph):
        self.begin()
        self.dump_graph(graph, "before_async_tp_pass")
        count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", count)
        self.dump_graph(graph, "after_async_tp_pass")
        self.end_and_log()


class FlashInferAllReduceFusionParams:
    """Parameters for FlashInfer allreduce fusion operations."""
    
    def __init__(self,
                 rank: int,
                 world_size: int,
                 hidden_dim: int,
                 workspace_tensor: torch.Tensor,
                 use_fp32_lamport: bool = False):
        self.rank = rank
        self.world_size = world_size
        self.hidden_dim = hidden_dim
        self.use_fp32_lamport = use_fp32_lamport
        self.workspace_tensor = workspace_tensor
        self.trigger_completion_at_end = True
        self.launch_with_pdl = True
        self.use_oneshot = True
        self.fp32_acc = False
        self.pattern_code = 0

    def get_trtllm_fusion_kwargs(self):
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "hidden_dim": self.hidden_dim,
            "workspace_tensor": self.workspace_tensor,
            "launch_with_pdl": self.launch_with_pdl,
            "use_oneshot": self.use_oneshot,
            "trigger_completion_at_end": self.trigger_completion_at_end,
            "fp32_acc": self.fp32_acc,
            "pattern_code": self.pattern_code,
        }


class AllReduceRMSNORMPattern(BasePattern):
    def __init__(self,
                 epsilon: float,
                 dtype: torch.dtype,
                 device: str,
                 allreduce_params: "FlashInferAllReduceFusionParams"):
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.allreduce_params = allreduce_params
        

    def get_inputs(self):
        input = torch.empty([1, 8, 4], device=self.device, dtype=self.dtype)
        rms_result = torch.empty([1, 8, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4], device=self.device, dtype=self.dtype)

        return [input, rms_result, weight]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(input: torch.Tensor,
                    rms_result: torch.Tensor,
                    weight: torch.Tensor):
            all_reduce_output = tensor_model_parallel_all_reduce(input)
            rms = auto_functionalized(RMS_OP,
                                     result=rms_result,
                                     input=all_reduce_output,
                                     weight=weight,
                                     epsilon=self.epsilon)
            return rms[1], all_reduce_output

        def replacement(input: torch.Tensor, rms_result: torch.Tensor,
                        weight: torch.Tensor):
            residual_in = torch.zeros_like(input)
            residual_out = torch.empty_like(residual_in)
            ar_rms = flashinfer_comm.trtllm_allreduce_fusion(
                allreduce_in=input,
                token_num=input.numel() // self.allreduce_params.hidden_dim,
                residual_in=residual_in,
                residual_out=residual_out,
                norm_out=rms_result,
                rms_w=weight,
                rms_eps=self.epsilon,
                **self.allreduce_params.get_trtllm_fusion_kwargs())
            # return residual_out as allreduce_out with zeroed residual_in
            # as flashinfer does not support rms_norm + allreduce_out

            return rms_result, residual_out

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)

class AllReduceFusedAddRMSNormPattern(BasePattern):
    def __init__(self,
                 epsilon: float,
                 dtype: torch.dtype,
                 device: str,
                 allreduce_params: "FlashInferAllReduceFusionParams"):
        super().__init__(dtype, device) 
        self.epsilon = epsilon
        self.allreduce_params = allreduce_params
    
    def get_inputs(self):
        input = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4],
                                       device=self.device,
                                       dtype=self.dtype)
        return [
            residual,
            input,
            weight,
        ]
    
    def register(self, pm_pass: PatternMatcherPass):
        def pattern(residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor):
            all_reduce_output = tensor_model_parallel_all_reduce(input)
            rms = auto_functionalized(RMS_ADD_OP,
                                     input=all_reduce_output,
                                     residual=residual, 
                                     weight=weight,
                                     epsilon=self.epsilon)
            return rms[1], rms[2]

        def replacement(residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor):
            residual_out = torch.empty_like(residual)
            rms_out = torch.empty_like(residual)
            flashinfer_comm.trtllm_allreduce_fusion(
                allreduce_in=input,
                token_num=input.numel() // self.allreduce_params.hidden_dim,
                residual_in=residual,
                residual_out=residual_out,
                norm_out=rms_out,
                rms_w=weight,
                rms_eps=self.epsilon,
                **self.allreduce_params.get_trtllm_fusion_kwargs())
            return rms_out, residual_out


        pm.register_replacement(pattern, replacement, self.get_inputs(),
                        pm.fwd_only, pm_pass)
    


class AllReduceFusionPass(VllmInductorPass):
    
    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.disabled = True
        if flashinfer_comm is None:
            return
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size <= 1:
            return
        
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="all_reduce_fusion_pass")
        hidden_dim = config.model_config.get_hidden_size() if config.model_config else 4096
        self.group = get_tp_group().device_group
        max_token_num = config.model_config.max_model_len
        use_fp32_lamport = self.model_dtype == torch.float32
        rank = get_tensor_model_parallel_rank()
        self.ipc_handles, workspace_tensor = (
            flashinfer_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank,
                tp_size,
                max_token_num,
                hidden_dim,
                group=self.group,
                use_fp32_lamport=use_fp32_lamport,
            ))
        self.allreduce_params = FlashInferAllReduceFusionParams(
            rank=rank,
            world_size=tp_size,
            hidden_dim=hidden_dim,
            workspace_tensor=workspace_tensor,
            use_fp32_lamport=use_fp32_lamport)
        for epsilon in [1e-5, 1e-6]:
            AllReduceRMSNORMPattern(
                epsilon, self.model_dtype, self.device,
                self.allreduce_params).register(self.patterns)
            AllReduceFusedAddRMSNormPattern(
                epsilon, self.model_dtype, self.device,
                self.allreduce_params).register(self.patterns)
        self.disabled = False


    def __call__(self, graph: fx.Graph):
        if self.disabled:
            return
        self.begin()
        self.dump_graph(graph, "before_all_reduce_fusion_pass")
        count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", count)
        self.dump_graph(graph, "after_all_reduce_fusion_pass")
        self.end_and_log()
    
    def __del__(self):
        if self.disabled:
            return
        flashinfer_comm.trtllm_destroy_ipc_workspace(self.ipc_handles, self.group)