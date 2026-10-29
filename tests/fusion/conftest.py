# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixtures for manual-fusion tests.

The counters here are installed *inside* each worker through ``collective_rpc``
rather than in the test process: with the V1 engine the model forward runs in
spawned worker processes, so a monkeypatch applied in the test process never
reaches the code under test and every count reads zero.

Manual fusion on this branch dispatches through a Python-level helper
(``fused_ar_rms_norm_quant``), which a ``TorchDispatchMode`` cannot observe, so
the counter wraps the module-level Python symbols instead.
"""

import os
from collections import Counter
from contextlib import contextmanager

import pytest

# Required so collective_rpc can ship our counter install/fetch closures to
# workers (vLLM defaults to a strict msgpack codec that rejects functions).
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _install_ar_counter(worker_self):
    """Wrap the manual-fusion entry points on this worker. Runs via RPC."""
    import vllm.model_executor.layers.fusion.ar_rms_quant as arq
    import vllm.model_executor.layers.fusion.residual_stream as rs
    import vllm.model_executor.models.qwen2 as qwen2

    state: Counter[str] = Counter()
    orig_fused = arq.fused_ar_rms_norm_quant
    orig_finalize = rs.finalize_norm
    orig_all_reduce = rs.tensor_model_parallel_all_reduce

    def counting_fused(hidden_states, residual, norm, consumer_linear, *, do_allreduce):
        state["prepare_calls"] += 1
        if do_allreduce:
            state["owed_allreduce"] += 1
        return orig_fused(
            hidden_states,
            residual,
            norm,
            consumer_linear,
            do_allreduce=do_allreduce,
        )

    def counting_finalize(norm, hidden_states, residual, *, incoming):
        # Once per model forward for every decoder, migrated or not, so it is
        # the forward counter the other counts are normalized against.
        state["forwards"] += 1
        return orig_finalize(norm, hidden_states, residual, incoming=incoming)

    def counting_all_reduce(x):
        state["explicit_allreduce"] += 1
        return orig_all_reduce(x)

    # ``residual_stream.py`` imported the symbol directly, so every binding must
    # be patched for all call sites to be counted.
    arq.fused_ar_rms_norm_quant = counting_fused
    rs.fused_ar_rms_norm_quant = counting_fused
    rs.finalize_norm = counting_finalize
    qwen2.finalize_norm = counting_finalize
    rs.tensor_model_parallel_all_reduce = counting_all_reduce
    worker_self._ar_counter = state


def _fetch_ar_counter(worker_self) -> dict[str, int]:
    """Return this worker's counts. Runs via RPC."""
    state = getattr(worker_self, "_ar_counter", None)
    if state is None:
        return {}
    del worker_self._ar_counter
    return dict(state)


@pytest.fixture
def ar_counter_session():
    """Aggregate manual-fusion counts from every worker of an ``LLM``.

    Usage::

        with ar_counter_session(llm) as counts:
            llm.generate(...)
        assert counts["owed_allreduce"] > 0
    """

    @contextmanager
    def session(llm):
        llm.collective_rpc(_install_ar_counter)
        aggregated: Counter = Counter()
        try:
            yield aggregated
        finally:
            for per_worker in llm.collective_rpc(_fetch_ar_counter):
                aggregated.update(per_worker)

    return session
