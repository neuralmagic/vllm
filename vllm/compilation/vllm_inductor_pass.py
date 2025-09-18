# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import operator
import time
from pathlib import Path
from typing import ClassVar, Optional

import regex as re
import torch
from torch._dynamo.utils import lazy_format_graph_code
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.logger import init_logger

from ..utils import unique_filepath
from .inductor_pass import InductorPass

logger = init_logger(__name__)


class VllmInductorPass(InductorPass):
    """
    An inductor pass with access to vLLM PassConfig.
    It provides timing, logging, and dumping utilities.
    """
    dump_prefix: ClassVar[Optional[int]] = None
    """Keep track of pass index for debug dump ordering."""

    def __init__(self, config: VllmConfig):
        self.pass_config = config.compilation_config.pass_config
        self.model_dtype = config.model_config.dtype if config.model_config \
            else None
        self.device = config.device_config.device if config.device_config \
            else None
        self.pass_name = self.__class__.__name__

    @staticmethod
    def time_and_log(call_fn):

        @functools.wraps(call_fn)
        def wrapped(self: VllmInductorPass, graph: torch.fx.Graph):
            self.begin()
            self.dump_graph(graph, "before")
            call_fn(self, graph)
            self.dump_graph(graph, "after")
            self.end_and_log()

        return wrapped

    def dump_graph(self, graph: torch.fx.Graph, stage: str):
        i = VllmInductorPass.dump_prefix
        i_str = "" if i is None else f".{i}"
        lazy_format_graph_code(f"post_grad{i_str}.{self.pass_name}.{stage}",
                               graph.owning_module)

    def begin(self):
        self._start_time = time.perf_counter_ns()

    def end_and_log(self):
        self._end_time = time.perf_counter_ns()
        duration_ms = float(self._end_time - self._start_time) / 1.0e6
        logger.debug("%s completed in %.1f ms", self.pass_name, duration_ms)


class VllmPatternMatcherPass(VllmInductorPass):
    """
    A VllmInductorPass that uses the Inductor pattern matcher.
    Its main use is providing the dump_patterns utility that dumps the
    Inductor pattern matcher patterns into a file, which greatly aids debugging.

    TODO(luka) move more utilities to this pass.
    """
    matched_count: int = 0
    """The number of matched patterns in the pass."""

    _OP_OVERLOAD_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"<OpOverload\(op='([^']*)', overload='([^']*)'\)>")

    _FUNC_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"<function ([^ ]+) at 0x[0-9a-fA-F]+>")

    def _replace_op_overloads(self, string: str) -> str:
        """Replace <OpOverload(..., ...)> with nicer formulations"""
        return self._OP_OVERLOAD_PATTERN.sub(
            lambda m: f"torch.ops.{m.group(1)}.{m.group(2)}",
            string,
        )

    def _replace_func_repr(self, string: str) -> str:
        # Replace <function a.<locals>.b at 0x75d...> with 'a.b'
        return self._FUNC_PATTERN.sub(
            lambda m: f"'{m.group(1).replace('<locals>.', '')}'",
            string,
        )

    def dump_patterns(self, config: VllmConfig, pm_pass: PatternMatcherPass):
        """
        If debug dumping is enabled, dump the Inductor pattern-matcher patterns
        into the debug_dump_path folder next to the dumped fx graphs.

        This method does its best to print something that looks like Python code
        for easier debugging and potentially navigation. If any errors appear in
        the output, please add to this method.

        TODO PatternPrettyPrinter, return pattern, manually trace graph to print.
        Also TODO: maybe add pattern base class with utilities.
        """
        debug_dump_path = config.compilation_config.debug_dump_path
        if not debug_dump_path:
            return

        rank = config.parallel_config.rank
        debug_dump_path = Path(debug_dump_path) / f"rank_{rank}"
        debug_dump_path.mkdir(parents=True, exist_ok=True)

        file_path = unique_filepath(
            lambda i: debug_dump_path / f"patterns.{self.pass_name}.{i}.py")

        with file_path.open("w") as f:
            print(
                f'# This file was produced by VllmPatternMatcherPass.'
                f'dump_patterns for {self.pass_name}.\n'
                f'# It does its best to produce valid-Python-looking code but'
                f' please add to dump_patterns if there are any errors.\n\n'
                f'from torch._higher_order_ops.auto_functionalize import '
                f'auto_functionalized\n'
                f'from torch._inductor.pattern_matcher import *',
                file=f)
            print("patterns = {", file=f)
            for node, patterns in pm_pass.patterns.items():
                # fix the operator.getitem repr
                if node[1] == operator.getitem:
                    node_repr = f"({repr(node[0])}, operator.getitem)"
                else:
                    node_repr = repr(node)

                node_repr = self._replace_op_overloads(node_repr)

                print(f"{node_repr}: [", file=f)
                for pattern in patterns:
                    pattern_repr = self._replace_op_overloads(repr(pattern))
                    pattern_repr = self._replace_func_repr(pattern_repr)

                    # also wrap wildcards (and make yapf happy)
                    pattern_repr = pattern_repr.replace('*', '\'*\'')

                    print(f"  {pattern_repr},", file=f)
                print("]", file=f)
            print("}\n\n\n", file=f)
            print(
                "# ===============\n"
                "# PATTERN GRAPHS:\n"
                "# A pattern might have multiple graphs associated with it.\n"
                "# That's because search patterns ignore scalars and shapes,\n"
                "# so different traced patterns could produce the same search pattern.\n"
                "# ===============\n",
                file=f)

            for i, (pattern,
                    graphs) in enumerate(pm_pass.seen_patterns.items()):
                print("\n")
                print(f"# Pattern {i}:", file=f)
                print(pattern, file=f)
                print("\n\n")
                for j, graph in enumerate(graphs):
                    print(f"# Pattern {i}, graph {j}:", file=f)
                    print('"""', file=f)
                    print(graph, file=f)
                    print('"""\n', file=f)


class PrinterInductorPass(VllmInductorPass):

    def __init__(self, name: str, config: VllmConfig):
        super().__init__(config)
        self.name = name

    def __call__(self, graph: torch.fx.Graph):
        self.dump_graph(graph, self.name)
