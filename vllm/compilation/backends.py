import ast
import copy
import dataclasses
import os
import pprint
import time
from collections import defaultdict
from contextlib import ExitStack
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from unittest.mock import patch

import torch
import torch.fx as fx

import vllm.envs as envs
from vllm.config import CompilationConfig, VllmConfig
from vllm.logger import init_logger
from vllm.utils import weak_ref_tensors

from .counter import compilation_counter
from .inductor_pass import InductorPass, pass_context
from .monitor import end_monitoring_torch_compile
from .pass_manager import PostGradPassManager
from .utils import dump_graph
from .shape_prop import ShapeProp

logger = init_logger(__name__)


ID = 0
PASSES = {}
def get_dummy_pass(id: int):
    def dummy_pass(g: fx.Graph):
        print(f"DUMMY PASS {id}")

    pass_fn = PASSES.get(id)
    if pass_fn is None:
        pass_fn = dummy_pass
        PASSES[id] = pass_fn

    return pass_fn


def add_dummy_pass(fn):
    # Disable for now
    if fn is None:
        return fn
    else:
        global ID
        dummy = get_dummy_pass(ID)
        ID = ID + 1
        def compose(graph: fx.GraphModule):
            dummy(graph)
            fn(graph)

    return compose


def wrap_inductor(graph: fx.GraphModule,
                  example_inputs: Sequence[Any],
                  additional_inductor_config: Optional[Dict] = None,
                  compilation_config: CompilationConfig,
                  graph_index: int = 0,
                  num_graphs: int = 1,
                  runtime_shape: Optional[int] = None,
                  use_inductor: bool = True) -> Any:
    if graph_index == 0:
        # before compiling the first graph, record the start time
        global compilation_start_time
        compilation_start_time = time.time()

    if not use_inductor:
        return graph

    compilation_counter.num_inductor_compilations += 1

    logger.info("Inputs: %s", ", ".join([str((inp.shape, inp.dtype) if isinstance(inp, torch.Tensor) else inp) for inp in example_inputs][0:3]))

    from torch._inductor import config
    from torch._inductor.compile_fx import compile_fx

    current_config = config.get_config_copy()

    # Enable support for symmetric memory ops in the inductor.
    current_config._micro_pipeline_tp = True

    if additional_inductor_config is not None:
        current_config.update(additional_inductor_config)

    current_config["post_grad_custom_post_pass"] = \
        add_dummy_pass(current_config["post_grad_custom_post_pass"])

    if isinstance(runtime_shape, int):
        # for a specific batchsize, tuning triton kernel parameters
        # can be beneficial
        current_config["max_autotune"] = True
        current_config["coordinate_descent_tuning"] = True

    # inductor can inplace modify the graph, so we need to copy it
    # see https://github.com/pytorch/pytorch/issues/138980
    graph = copy.deepcopy(graph)

    cache_data = compilation_config.inductor_hash_cache
    if (runtime_shape, graph_index) in cache_data:
        # we compiled this graph before
        # so we can directly lookup the compiled graph via hash
        hash_str = cache_data[(runtime_shape, graph_index)]
        if graph_index == 0:
            # adds some info logging for the first graph
            logger.info(
                "Directly lookup the graph for shape %s from the cache",
                str(runtime_shape))  # noqa
        logger.debug(
            "directly lookup the %s-th graph for shape %s via hash %s",
            graph_index, str(runtime_shape), hash_str)
        from torch._inductor.codecache import FxGraphCache
        with patch("torch._inductor.codecache.FxGraphCache._get_shape_env",
                   lambda *args, **kwargs: AlwaysHitShapeEnv()):
            inductor_compiled_graph = FxGraphCache._lookup_graph(
                hash_str, example_inputs, True, False)
            assert inductor_compiled_graph is not None, (
                "Inductor cache lookup failed. Please remove"
                f"the cache file {compilation_config.inductor_hash_cache.cache_file_path} and try again."  # noqa
            )

        # Inductor calling convention (function signature):
        # f(list) -> tuple
        # Dynamo calling convention (function signature):
        # f(*args) -> Any

        # need to know if the graph returns a tuple
        from torch._inductor.compile_fx import graph_returns_tuple
        returns_tuple = graph_returns_tuple(graph)

        # this is the callable we return to Dynamo to run
        def compiled_graph(*args):
            # convert args to list
            list_args = list(args)
            graph_output = inductor_compiled_graph(list_args)
            # unpack the tuple if needed
            if returns_tuple:
                return graph_output
            else:
                return graph_output[0]
    else:
        # it's the first time we compile this graph
        # the assumption is that we don't have nested Inductor compilation.
        # compiled_fx_graph_hash will only be called once, and we can hook
        # it to get the hash of the compiled graph directly.
        from torch._inductor.codecache import compiled_fx_graph_hash

        def hijack_compiled_fx_graph_hash(*args, **kwargs):
            out = compiled_fx_graph_hash(*args, **kwargs)
            # store the hash in the cache
            nonlocal cache_data
            cache_data[(runtime_shape, graph_index)] = out[0]
            if graph_index == 0:
                # adds some info logging for the first graph
                logger.info("Cache the graph of shape %s for later use",
                            str(runtime_shape))
            logger.debug("store the %s-th graph for shape %s via hash %s",
                         graph_index, str(runtime_shape), out[0])
            return out

        def _check_can_cache(*args, **kwargs):
            # no error means it can be cached.
            # Inductor refuses to cache the graph outside of Dynamo
            # tracing context, and also disables caching for graphs
            # with high-order ops.
            # For vLLM, in either case, we want to cache the graph.
            # see https://github.com/pytorch/pytorch/blob/9f5ebf3fc609105a74eab4ccc24932d6353ff566/torch/_inductor/codecache.py#L1221 # noqa
            return

        def _get_shape_env() -> AlwaysHitShapeEnv:
            return AlwaysHitShapeEnv()

        with patch(# for hijacking the hash of the compiled graph
                "torch._inductor.codecache.compiled_fx_graph_hash",
                hijack_compiled_fx_graph_hash), \
            patch(# for providing a dummy shape environment
                "torch._inductor.codecache.FxGraphCache._get_shape_env",
                 _get_shape_env), \
            patch(# for forcing the graph to be cached
                "torch._inductor.codecache.FxGraphCache._check_can_cache",
                _check_can_cache):
            with pass_context(runtime_shape):
                compiled_graph = compile_fx(graph,
                                            example_inputs,
                                            config_patches=current_config)

    # after compiling the last graph, record the end time
    if graph_index == num_graphs - 1:
        now = time.time()
        elapsed = now - compilation_start_time
        compilation_config.compilation_time += elapsed
        if runtime_shape is None:
            logger.info("Compiling a graph for general shape takes %.2f s",
                        elapsed)
        else:
            logger.info("Compiling a graph for shape %s takes %.2f s",
                        runtime_shape, elapsed)


    return compiled_graph


@dataclasses.dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule


def find_last(nodes, op):
    last = None
    found = False
    for n in nodes:
        if n.op == op:
            found = True
        if found and n.op != op:
            break
        last = n
    return last


def add_input_output_to_graph(gm: fx.GraphModule, arg_name, add_output: bool = False):
    last_input = find_last(gm.graph.nodes, 'placeholder')
    assert last_input is not None
    with gm.graph.inserting_after(last_input):
        input_node = gm.graph.placeholder(arg_name, "f16[s0, 4096]", None) # type_expr?

    if add_output:
        last_output = find_last(gm.graph.nodes, 'output')
        assert last_output is not None
        if isinstance(last_output.args[0], tuple):
            last_output.args = (last_output.args[0] + (input_node, ), )
        else:
            last_output.args = (last_output.args + (input_node, ), )

    return input_node


def add_residual_args(split_gm: fx.GraphModule):
    names = [name for (name, module) in split_gm.named_modules()]
    count = 0
    new_arg = f"my_residual_{count}"
    new_input = add_input_output_to_graph(split_gm, new_arg, False)

    for n in split_gm.graph.nodes:
        if n.op == 'call_module' and n.target in names:
            count += 1
            n.args = n.args + (new_input,)
            submod = getattr(split_gm, n.target)
            new_arg = f"my_residual_{count}"
            add_input_output_to_graph(submod, new_arg, True)

            output_node = find_last(submod.graph.nodes, 'output')
            assert output_node is not None
            outputs = output_node.args[0]

            if isinstance(outputs, tuple) and len(outputs) > 2:
                with split_gm.graph.inserting_after(n):
                    new_input = split_gm.graph.call_function(operator.getitem, (n, len(outputs) - 1))
            else:
                with split_gm.graph.inserting_after(n):
                    new_input = split_gm.graph.call_function(operator.getitem, (n, 1))
                    old_output = split_gm.graph.call_function(operator.getitem, (n, 0))
                del n.users[new_input]
                del n.users[old_output]
                n.replace_all_uses_with(old_output)
                n.users[old_output] = None
                n.users[new_input] = None
            submod.recompile()

    split_gm.recompile()


def split_graph(graph: fx.GraphModule,
                ops: List[str]) -> Tuple[fx.GraphModule, List[SplitItem]]:
    # split graph by ops
    subgraph_id = 0
    node_to_subgraph_id = {}
    split_op_graphs = []
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if node.op == 'call_function' and str(node.target) in ops:
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    # `keep_original_order` is important!
    # otherwise pytorch might reorder the nodes and
    # the semantics of the graph will change when we
    # have mutations in the graph
    split_gm = torch.fx.passes.split_module.split_module(
        graph,
        None,
        lambda node: node_to_subgraph_id[node],
        keep_original_order=True)

    outputs = []

    names = [name for (name, module) in split_gm.named_modules()]

    for name in names:
        if "." in name or name == "":
            # recursive child module or the root module
            continue

        module = getattr(split_gm, name)

        graph_id = int(name.replace("submod_", ""))
        outputs.append(
            SplitItem(name, graph_id, (graph_id in split_op_graphs), module))

    # sort by intetger graph_id, rather than string name
    outputs.sort(key=lambda x: x.graph_id)

    add_residual_args(split_gm)

    return split_gm, outputs


# we share the global graph pool among all the backends
global_graph_pool = None

compilation_start_time = 0.0


class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    """Code adapted from `torch.fx.passes.shape_prop.ShapeProp`.
    It runs the given graph with fake inputs, and compile some
    submodules specified by `compile_submod_names` with the given
    compilation configs.

    NOTE: the order in `compile_submod_names` matters, because
    it will be used to determine the order of the compiled piecewise
    graphs. The first graph will handle logging, and the last graph
    has some special cudagraph output handling.
    """

    def __init__(self, module: torch.fx.GraphModule,
                 compile_submod_names: List[str], vllm_config: VllmConfig,
                 graph_pool):
        super().__init__(module)
        from torch._guards import detect_fake_mode
        self.fake_mode = detect_fake_mode()
        self.compile_submod_names = compile_submod_names
        self.compilation_config = vllm_config.compilation_config
        self.graph_pool = graph_pool
        self.vllm_config = vllm_config

    def run(self, *args):
        self.fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]
        with self.fake_mode:
            return super().run(*self.fake_args)

    def call_module(self, target: torch.fx.node.Target,
                    args: Tuple[torch.fx.node.Argument,
                                ...], kwargs: Dict[str, Any]) -> Any:
        assert isinstance(target, str)
        output = super().call_module(target, args, kwargs)

        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)
            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]
            global compilation_start_time
            compiled_graph_for_general_shape = wrap_inductor(
                submod,
                args,
                self.compilation_config.inductor_compile_config,
                self.compilation_config,
                graph_index=index,
                num_graphs=len(self.compile_submod_names),
                runtime_shape=None,
                use_inductor=self.compilation_config.use_inductor)

            self.module.__dict__[target] = PiecewiseBackend(
                submod, self.vllm_config, self.graph_pool, index,
                len(self.compile_submod_names), sym_shape_indices,
                compiled_graph_for_general_shape)

            compilation_counter.num_piecewise_capturable_graphs_seen += 1

        return output


class VllmBackend:
    """The compilation backend for `torch.compile` with VLLM.
    It is used for compilation level of `CompilationLevel.PIECEWISE`,
    where we customize the compilation.

    The major work of this backend is to split the graph into
    piecewise graphs, and pass them to the piecewise backend.

    This backend also adds the PostGradPassManager to Inductor config,
    which handles the post-grad passes.
    """

    vllm_config: VllmConfig
    compilation_config: CompilationConfig
    graph_pool: Any
    _called: bool = False
    # the graph we compiled
    graph: fx.GraphModule
    # the stiching graph module for all the piecewise graphs
    split_gm: fx.GraphModule
    piecewise_graphs: List[SplitItem]
    returned_callable: Callable
    # Inductor passes to run on the graph pre-defunctionalization
    post_grad_passes: Sequence[Callable]
    sym_tensor_indices: List[int]
    input_buffers: List[torch.Tensor]

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        global global_graph_pool
        if global_graph_pool is None:
            global_graph_pool = torch.cuda.graph_pool_handle()

        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        self.graph_pool = global_graph_pool

        # Passes to run on the graph post-grad.
        self.post_grad_pass_manager = PostGradPassManager()

        self.sym_tensor_indices = []
        self.input_buffers = []

        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config

        # `torch.compile` is JIT compiled, so we don't need to
        # do anything here

    def configure_post_pass(self):
        config = self.compilation_config
        self.post_grad_pass_manager.configure(config.pass_config)

        # Post-grad custom passes are run using the post_grad_custom_post_pass
        # hook. If a pass for that hook exists, add it to the pass manager.
        inductor_config = config.inductor_compile_config
        PASS_KEY = "post_grad_custom_post_pass"
        if PASS_KEY in inductor_config:
            # Config should automatically wrap all inductor passes
            assert isinstance(inductor_config[PASS_KEY], InductorPass)
            self.post_grad_pass_manager.add(inductor_config[PASS_KEY])
        inductor_config[PASS_KEY] = self.post_grad_pass_manager

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:

        # when dynamo calls the backend, it means the bytecode
        # transform and analysis are done
        compilation_counter.num_graphs_seen += 1
        from .monitor import torch_compile_start_time
        dynamo_time = time.time() - torch_compile_start_time
        logger.info("Dynamo bytecode transform time: %.2f s", dynamo_time)
        self.compilation_config.compilation_time += dynamo_time

        # we control the compilation process, each instance can only be
        # called once
        assert not self._called, "VllmBackend can only be called once"

        self.graph = graph
        self.configure_post_pass()

        if ("before_split_graph"
                in self.compilation_configs.pass_config.dump_graph_stages):
            dump_graph(self.compilation_configs.pass_config, graph.graph,
                       "before_split_graph")

        self.split_gm, self.piecewise_graphs = split_graph(
            graph, self.compilation_config.splitting_ops)

        if ("after_split_graph"
                in self.compilation_configs.pass_config.dump_graph_stages):
            dump_graph(self.compilation_configs.pass_config,
                       self.split_gm.graph, "after_split_graph")

        from torch._dynamo.utils import lazy_format_graph_code

        # depyf will hook lazy_format_graph_code and dump the graph
        # for debugging, no need to print the graph here
        lazy_format_graph_code("before split", self.graph)
        lazy_format_graph_code("after split", self.split_gm)

        compilation_counter.num_piecewise_graphs_seen += len(
            self.piecewise_graphs)
        submod_names_to_compile = [
            item.submod_name for item in self.piecewise_graphs
            if not item.is_splitting_graph
        ]

        # propagate the split graph to the piecewise backend,
        # compile submodules with symbolic shapes
        PiecewiseCompileInterpreter(self.split_gm, submod_names_to_compile,
                                    self.vllm_config,
                                    self.graph_pool).run(*example_inputs)

        self._called = True

        if not self.compilation_config.use_cudagraph or \
            not self.compilation_config.cudagraph_copy_inputs:
            return self.split_gm

        # if we need to copy input buffers for cudagraph
        from torch._guards import detect_fake_mode
        fake_mode = detect_fake_mode()
        fake_args = [
            fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in example_inputs
        ]

        # index of tensors that have symbolic shapes (batch size)
        self.sym_tensor_indices = [
            i for i, x in enumerate(fake_args)
            if isinstance(x, torch._subclasses.fake_tensor.FakeTensor)
        ]

        # compiler managed cudagraph input buffers
        # we assume the first run with symbolic shapes
        # has the maximum size among all the tensors
        self.input_buffers = [
            example_inputs[x].clone() for x in self.sym_tensor_indices
        ]

        # this is the callable we return to Dynamo to run
        def copy_and_call(*args):
            list_args = list(args)
            for i, index in enumerate(self.sym_tensor_indices):
                runtime_tensor = list_args[index]
                runtime_shape = runtime_tensor.shape[0]
                static_tensor = self.input_buffers[i][:runtime_shape]

                # copy the tensor to the static buffer
                static_tensor.copy_(runtime_tensor)

                # replace the tensor in the list_args to the static buffer
                list_args[index] = static_tensor
            return self.split_gm(*list_args)

        return copy_and_call


@dataclasses.dataclass
class ConcreteSizeEntry:
    runtime_shape: int
    need_to_compile: bool  # the size is in compile_sizes
    use_cudagraph: bool  # the size is in capture_sizes

    compiled: bool = False
    runnable: Callable = None  # type: ignore
    num_finished_warmup: int = 0
    cudagraph: Optional[torch.cuda.CUDAGraph] = None
    output: Optional[Any] = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[List[int]] = None


class PiecewiseBackend:

    def __init__(self, graph: fx.GraphModule, vllm_config: VllmConfig,
                 graph_pool: Any, piecewise_compile_index: int,
                 total_piecewise_compiles: int, sym_shape_indices: List[int],
                 compiled_graph_for_general_shape: Callable):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation and cudagraph capturing.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.

        Independently, we will capture cudagraph for different shapes.

        If a shape needs both compilation and cudagraph, we will
        compile it first, and then capture cudagraph.
        """
        self.graph = graph
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.graph_pool = graph_pool
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = (
            piecewise_compile_index == total_piecewise_compiles - 1)

        self.compile_sizes: Set[int] = set(
            self.compilation_config.compile_sizes)
        self.capture_sizes: Set[int] = set(
            self.compilation_config.capture_sizes
        ) if self.compilation_config.use_cudagraph else set()

        self.first_run_finished = False

        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape  # noqa

        self.sym_shape_indices = sym_shape_indices

        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # the entries for different shapes that we need to either
        # compile or capture cudagraph
        self.concrete_size_entries: Dict[int, ConcreteSizeEntry] = {}

        # to_be_compiled_sizes tracks the remaining sizes to compile,
        # and updates during the compilation process, so we need to copy it
        self.to_be_compiled_sizes: Set[int] = self.compile_sizes.copy()
        for shape in self.compile_sizes.union(self.capture_sizes):
            self.concrete_size_entries[shape] = ConcreteSizeEntry(
                runtime_shape=shape,
                need_to_compile=shape in self.compile_sizes,
                use_cudagraph=shape in self.capture_sizes,
            )

    def check_for_ending_compilation(self):
        if self.is_last_graph and not self.to_be_compiled_sizes:
            # no specific sizes to compile
            # save the hash of the inductor graph for the next run
            self.compilation_config.inductor_hash_cache.save_to_file()
            end_monitoring_torch_compile(self.vllm_config)

    def __call__(self, *args) -> Any:
        if not self.first_run_finished:
            self.first_run_finished = True
            self.check_for_ending_compilation()
            return self.compiled_graph_for_general_shape(*args)

        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            # we don't need to do anything for this shape
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]

        if entry.runnable is None:
            entry.runnable = self.compiled_graph_for_general_shape

        if entry.need_to_compile and not entry.compiled:
            entry.compiled = True
            self.to_be_compiled_sizes.remove(runtime_shape)
            # args are real arguments
            entry.runnable = wrap_inductor(
                self.graph,
                args,
                self.compilation_config.inductor_compile_config,
                self.compilation_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                runtime_shape=runtime_shape,
                use_inductor=self.compilation_config.use_inductor)

            # finished compilations for all required shapes
            if self.is_last_graph and not self.to_be_compiled_sizes:
                self.check_for_ending_compilation()

        if not entry.use_cudagraph:
            return entry.runnable(*args)

        if entry.cudagraph is None:
            if entry.num_finished_warmup < self.compilation_config.cudagraph_num_of_warmups:  # noqa
                entry.num_finished_warmup += 1
                if self.is_first_graph:
                    logger.debug(
                        "Warming up %s/%s for shape %s",
                        entry.num_finished_warmup,
                        self.compilation_config.cudagraph_num_of_warmups,
                        runtime_shape)
                return entry.runnable(*args)

            if self.is_first_graph:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every shape.
                # We only log it in the debug mode.
                logger.debug("Capturing a cudagraph for shape %s",
                             runtime_shape)

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            cudagraph = torch.cuda.CUDAGraph()

            with ExitStack() as stack:
                if not self.is_first_graph:
                    # during every model forward, we will capture
                    # many pieces of cudagraphs (roughly one per layer).
                    # running gc again and again across layers will
                    # make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(
                        patch("torch.cuda.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.
                with torch.cuda.graph(cudagraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's cudagraph pool
                    output = entry.runnable(*args)
                    if self.is_last_graph:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph, because the output of the last graph
                        # will not be used by any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

            compilation_counter.num_cudagraph_caputured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                "Input addresses for cudagraphs are different during replay."
                f" Expected {entry.input_addresses}, got {new_input_addresses}"
            )

        entry.cudagraph.replay()
        return entry.output
