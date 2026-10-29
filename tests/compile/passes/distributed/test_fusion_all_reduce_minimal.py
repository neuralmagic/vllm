# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal all-reduce + RMSNorm fusion test.

This is the stripped-down core of ``test_fusion_all_reduce.py``: one model
with a single ``all_reduce -> fused_add_rms_norm`` site, one flashinfer
backend, and only the two passes the fusion structurally requires
(``AllReduceFusionPass`` to rewrite the pattern and
``FixFunctionalizationPass`` to defunctionalize the mutating fused op).

What it proves, on 2 real GPUs:
1. the pattern matcher fires exactly once (``matched_count == 1``),
2. the fused op replaces the unfused ``torch.ops.vllm.all_reduce`` node,
3. the fused output matches the unfused eager output numerically.

Why 2 GPUs are irreducible: ``AllReduceFusionPass`` disables itself for
``tp_size <= 1`` and its workspace creation is collective over the TP group,
so the fusion cannot be exercised single-GPU or single-process.
"""

from importlib.util import find_spec

import pytest
import torch

from tests.compile.backend import TestBackend
from tests.utils import has_module_attribute, multi_gpu_test
from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
    AllReduceFusionPass,
)
from vllm.compilation.passes.utility.fix_functionalization import (
    FixFunctionalizationPass,
)
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed

DEVICE_TYPE = current_platform.device_type


class SingleSiteAllReduceRMSNormModel(torch.nn.Module):
    """One ``all_reduce -> fused_add_rms_norm`` site.

    Mirrors the o_proj -> post_attention_layernorm boundary of a TP decoder:
    ``partial`` stands in for a row-parallel linear's un-reduced output. The
    ``relu``s ensure no graph input feeds the pattern directly (the fused op
    mutates ``allreduce_in``/``residual``, which must be graph-internal).
    """

    def __init__(self, hidden_size: int = 64, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)
        self.w = torch.rand(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        partial = torch.mm(torch.relu(x), self.w)
        residual = torch.relu(x)
        reduced = tensor_model_parallel_all_reduce(partial)
        return self.norm(reduced, residual)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test on CUDA")
@pytest.mark.skipif(
    not find_spec("flashinfer")
    or not has_module_attribute("flashinfer.comm", "allreduce_fusion")
    or not has_module_attribute("flashinfer.comm", "create_allreduce_fusion_workspace"),
    reason="flashinfer is not found or flashinfer "
    "is not compiled with allreduce_fusion",
)
def test_all_reduce_fusion_minimal(dtype: torch.dtype):
    master_port = get_open_port()
    torch.multiprocessing.spawn(
        _run_minimal_ar_fusion_test,
        args=(2, master_port, dtype),
        nprocs=2,
    )


def _run_minimal_ar_fusion_test(
    local_rank: int,
    world_size: int,
    master_port: int,
    dtype: torch.dtype,
):
    set_random_seed(0)

    device = torch.device(f"{DEVICE_TYPE}:{local_rank}")
    torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
            # Single-node NVSwitch backend; the natural default here.
            "VLLM_FLASHINFER_ALLREDUCE_BACKEND": "trtllm",
        }
    )
    init_distributed_environment()

    # The pass reads model_config (hidden size / dtype for workspace sizing);
    # the model name is only used to construct that config, as in the
    # full-scale test.
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, custom_ops=["+rms_norm"]
        )
    )
    vllm_config.compilation_config.pass_config = PassConfig(
        fuse_allreduce_rms=True, eliminate_noops=True
    )
    vllm_config.device_config = DeviceConfig(device=torch.device(DEVICE_TYPE))
    vllm_config.parallel_config.rank = local_rank
    vllm_config.model_config = ModelConfig(
        model="RedHatAI/Llama-3.2-1B-Instruct-FP8",
        trust_remote_code=True,
        dtype=dtype,
        seed=42,
    )

    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        fusion_pass = AllReduceFusionPass(vllm_config)
        func_pass = FixFunctionalizationPass(vllm_config)
        backend = TestBackend(fusion_pass, func_pass)

        model = SingleSiteAllReduceRMSNormModel(hidden_size=64)
        hidden_states = torch.randn(8, 64, requires_grad=False)

        compiled_model = torch.compile(model, backend=backend)
        compiled_model(hidden_states)  # trigger compilation

        results_unfused = model(hidden_states)
        results_fused = compiled_model(hidden_states)
        torch.testing.assert_close(results_unfused, results_fused, atol=1e-2, rtol=1e-2)

        # The single site fused: pattern fired once, unfused AR is gone,
        # fused op is present. (Resolved here, not at module level: the fused
        # op is only registered when flashinfer is importable.)
        fused_op = torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm.default
        unfused_ar_op = torch.ops.vllm.all_reduce.default
        assert fusion_pass.matched_count == 1, f"{fusion_pass.matched_count=}"
        backend.check_before_ops([unfused_ar_op], fully_replaced=True)
        backend.check_after_ops([fused_op])
        del fusion_pass


# ---------------------------------------------------------------------------
# Manual-fusion variant (model-level ResidualStream).
#
# Manual fusion moves AR + residual-add + RMSNorm (+ activation-quant) out of
# the compiler pass exercised above and into model code: a per-forward
# ResidualStream with each decoder layer doing
# prepare -> attn -> record -> prepare -> mlp -> record, and finalize_norm
# owning the last decoder's deferred all-reduce. This test fills the gap the
# pass-level test leaves: one decoder layer whose row-linears defer their
# reduce, run in eager with every compiler fusion off, checked for
# (a) the fused op firing exactly at the two deferred sites on the no-quant
# AR+RMSNorm pattern, (b) no fallback to an unfused all-reduce inside the
# manual path, and (c) numerical equivalence with an explicit-AR reference.
#
# The stream is constructed per forward (it carries the running residual and
# the pending contribution), and the layer declares its own output
# distribution state, exactly as the migrated Qwen3 decoder does.
# ---------------------------------------------------------------------------


def _flashinfer_fused_ar_available() -> bool:
    return bool(
        find_spec("flashinfer")
        and has_module_attribute("flashinfer.comm", "allreduce_fusion")
        and has_module_attribute("flashinfer.comm", "create_allreduce_fusion_workspace")
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test on CUDA")
@pytest.mark.skipif(
    not _flashinfer_fused_ar_available(),
    reason="flashinfer is not found or flashinfer "
    "is not compiled with allreduce_fusion",
)
def test_manual_all_reduce_fusion_minimal(dtype: torch.dtype):
    master_port = get_open_port()
    torch.multiprocessing.spawn(
        _run_manual_ar_fusion_test,
        args=(2, master_port, dtype),
        nprocs=2,
    )


def _run_manual_ar_fusion_test(
    local_rank: int,
    world_size: int,
    master_port: int,
    dtype: torch.dtype,
):
    # Manual-fusion modules; imported here so a missing optional dependency
    # cannot break collection of the pass-level test above.
    from vllm.model_executor.layers.fusion import ar_rms_quant
    from vllm.model_executor.layers.fusion.residual_stream import (
        ResidualStream,
        Scatter,
        finalize_norm,
    )

    set_random_seed(0)

    device = torch.device(f"{DEVICE_TYPE}:{local_rank}")
    torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
            "VLLM_FLASHINFER_ALLREDUCE_BACKEND": "trtllm",
        }
    )
    init_distributed_environment()

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, custom_ops=["+rms_norm"]
        )
    )
    # The manual fusion must not depend on the compiler pass; run eager.
    vllm_config.compilation_config.pass_config = PassConfig(
        fuse_allreduce_rms=False, eliminate_noops=True
    )
    vllm_config.device_config = DeviceConfig(device=torch.device(DEVICE_TYPE))
    vllm_config.parallel_config.rank = local_rank
    vllm_config.model_config = ModelConfig(
        model="RedHatAI/Llama-3.2-1B-Instruct-FP8",
        trust_remote_code=True,
        dtype=dtype,
        seed=42,
    )

    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)

        hidden_size = 64

        class ManualFusionLayer(torch.nn.Module):
            """One decoder layer shaped like a migrated Qwen3 forward.

            ``prepare -> attn -> record -> prepare -> mlp -> record`` with
            row-linears that defer their reduce, so the stream owes (and
            fuses) the all-reduce at the post-attention norm; ``finalize_norm``
            owns the MLP output's deferred all-reduce.
            """

            def __init__(self) -> None:
                super().__init__()
                self.input_layernorm = RMSNorm(hidden_size, eps=1e-6)
                self.post_attention_layernorm = RMSNorm(hidden_size, eps=1e-6)
                self.final_norm = RMSNorm(hidden_size, eps=1e-6)
                # Real row-parallel linears with their reduce deferred. Only
                # their reduce_results / quant-key metadata is read here: the
                # matmuls below stand in for the attention and MLP kernels.
                self.qkv_proj = RowParallelLinear(
                    hidden_size, hidden_size, bias=False, reduce_results=False
                )
                self.o_proj = RowParallelLinear(
                    hidden_size, hidden_size, bias=False, reduce_results=False
                )
                self.gate_up_proj = RowParallelLinear(
                    hidden_size, hidden_size, bias=False, reduce_results=False
                )
                self.down_proj = RowParallelLinear(
                    hidden_size, hidden_size, bias=False, reduce_results=False
                )
                self.w_attn = torch.rand(hidden_size, hidden_size)
                self.w_mlp = torch.rand(hidden_size, hidden_size)
                # Declared distribution state of this layer's output, as the
                # migrated decoders do.
                self.output_scatter = (
                    Scatter.PARTIAL
                    if not self.down_proj.reduce_results
                    else Scatter.FULL
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Per-forward stream: it carries the running residual, and the
                # fused kernel updates the residual/allreduce_in in place.
                stream = ResidualStream(vllm_config=vllm_config)
                hidden, residual = stream.prepare(
                    x.clone(), self.input_layernorm, self.qkv_proj
                )
                assert isinstance(hidden, torch.Tensor)  # no activation quant
                hidden = torch.mm(hidden, self.w_attn)  # o_proj: PARTIAL
                stream.record(
                    hidden,
                    Scatter.PARTIAL if stream.defer_attn(self.o_proj) else Scatter.FULL,
                )
                hidden, residual = stream.prepare(
                    hidden, self.post_attention_layernorm, self.gate_up_proj
                )
                assert isinstance(hidden, torch.Tensor)  # no activation quant
                hidden = torch.mm(hidden, self.w_mlp)  # down_proj: PARTIAL
                stream.record(
                    hidden,
                    Scatter.PARTIAL
                    if stream.defer_mlp(self.down_proj)
                    else Scatter.FULL,
                )
                return finalize_norm(
                    self.final_norm, hidden, residual, incoming=self.output_scatter
                )

        def unfused_reference(x: torch.Tensor) -> torch.Tensor:
            """Explicit all-reduce + RMSNorm at each boundary."""
            x = x.clone()
            hidden = model.input_layernorm(x)
            partial = torch.mm(hidden, model.w_attn)
            hidden, residual = model.post_attention_layernorm(
                tensor_model_parallel_all_reduce(partial), x
            )
            partial = torch.mm(hidden, model.w_mlp)
            out, _ = model.final_norm(
                tensor_model_parallel_all_reduce(partial), residual
            )
            return out

        set_random_seed(0)
        model = ManualFusionLayer()
        # Per-rank partial-producer weights so a dropped all-reduce (the bug
        # class this test guards) changes the output.
        torch.manual_seed(1000 + local_rank)
        model.w_attn = torch.rand(hidden_size, hidden_size)
        model.w_mlp = torch.rand(hidden_size, hidden_size)

        set_random_seed(7)
        hidden_states = torch.randn(8, hidden_size, requires_grad=False)

        # Spy on the manual path's kernel dispatch: patch the module-level
        # names ar_rms_quant's impls resolve at call time (the PR's suggested
        # mechanism -- TorchDispatchMode cannot see this dispatch).
        fused_op = ar_rms_quant.flashinfer_trtllm_fused_allreduce_norm
        assert fused_op is not None, "flashinfer fused AR op not registered"
        patterns = ar_rms_quant.ar_fusion_patterns
        assert patterns is not None
        unfused_ar = ar_rms_quant.tensor_model_parallel_all_reduce

        fused_op_patterns: list = []
        fallback_ar_calls = 0

        def spy_fused_op(*args, **kwargs):
            fused_op_patterns.append(kwargs.get("pattern_code"))
            return fused_op(*args, **kwargs)

        def spy_fallback_ar(*args, **kwargs):
            nonlocal fallback_ar_calls
            fallback_ar_calls += 1
            return unfused_ar(*args, **kwargs)

        ar_rms_quant.flashinfer_trtllm_fused_allreduce_norm = spy_fused_op
        ar_rms_quant.tensor_model_parallel_all_reduce = spy_fallback_ar
        try:
            results_fused = model(hidden_states)
            torch.accelerator.synchronize()
        finally:
            ar_rms_quant.flashinfer_trtllm_fused_allreduce_norm = fused_op
            ar_rms_quant.tensor_model_parallel_all_reduce = unfused_ar

        results_unfused = unfused_reference(hidden_states)
        torch.accelerator.synchronize()

        torch.testing.assert_close(results_unfused, results_fused, atol=1e-2, rtol=1e-2)

        # Exactly the two deferred sites fused (post-attention norm and the
        # final norm), on the no-quant AR+RMSNorm pattern, with no fallback
        # to an unfused all-reduce inside the manual path.
        expected_pattern = patterns.kARResidualRMSNorm
        assert fused_op_patterns == [expected_pattern, expected_pattern], (
            fused_op_patterns
        )
        assert fallback_ar_calls == 0, (
            "manual fusion fell back to an unfused all-reduce"
        )
