from collections import defaultdict
from dataclasses import dataclass, field
from vllm.profiler.utils import (indent_string, TablePrinter, event_has_module,
                                 event_is_torch_op, event_module_repr,
                                 event_torch_op_stack_trace)
from typing import Dict, List, Union, Optional, Tuple
from torch.profiler import profile, ProfilerActivity
from torch.autograd.profiler import EventList, FunctionEvent
from torch._C._autograd import _ProfilerResult, _KinetoEvent, DeviceType
from torch._C._profiler import _EventType, _ProfilerEvent, _ExperimentalConfig


@dataclass
class _ModuleTreeNode:
    event: _ProfilerEvent
    parent: Optional['_ModuleTreeNode'] = None
    children: List['_ModuleTreeNode'] = field(default_factory=list)
    trace: str = ""

    @property
    def is_leaf(self):
        return (self.event.children is None or len(self.event.children) == 0)

    @property
    def is_torch_op(self):
        return event_is_torch_op(self.event)

    @property
    def is_cuda(self):
        return (self.event.tag == _EventType.Kineto
                and self.event.typed[1].device_type == DeviceType.CUDA)


@dataclass
class NMProfileResults(profile):
    _kineto_results: _ProfilerResult
    _event_list: EventList
    _kineto_event_correlation_map: Dict[int,
                                        List[_KinetoEvent]] = field(init=False)
    _event_correlation_map: Dict[int, List[FunctionEvent]] = field(init=False)
    _module_tree: List[_ModuleTreeNode] = field(init=False)

    def __post_init__(self):
        self._build_correlation_maps()
        self._build_module_tree()

    def _build_correlation_maps(self):
        self._kineto_event_correlation_map = defaultdict(list)
        self._event_correlation_map = defaultdict(list)
        for event in self._kineto_results.events():
            self._kineto_event_correlation_map[event.correlation_id()].append(
                event)
        for event in self._event_list:
            self._event_correlation_map[event.id].append(event)

    def _build_module_tree(self):
        self._module_tree = []
        event_tree = self._kineto_results.experimental_event_tree()

        def _df_traversal(event: _ProfilerEvent,
                          curr_node: Optional[_ModuleTreeNode] = None):
            if event_has_module(event):
                node = _ModuleTreeNode(event=event, parent=curr_node)
                if curr_node:
                    curr_node.children.append(node)
                else:
                    self._module_tree.append(node)
                curr_node = node

            is_leaf = (event.children is None or len(event.children) == 0)
            if is_leaf and curr_node:
                node = _ModuleTreeNode(
                    event=event,
                    parent=curr_node,
                    trace=event_torch_op_stack_trace(
                        event, until=lambda x: event_has_module(x)))
                curr_node.children.append(node)
                curr_node = node

            for child in event.children:
                _df_traversal(child, curr_node)

        for root in event_tree:
            _df_traversal(root)

    def _get_kineto_gpu_event(self, node: _ModuleTreeNode):
        correlated_kineto_events = self._kineto_event_correlation_map.get(
            node.event.correlation_id, [])
        iterator = (x for x in correlated_kineto_events
                    if x.device_type() == DeviceType.CUDA)
        return next(iterator, None)

    def _cumulative_cuda_time(self, node: _ModuleTreeNode):

        def _cumulative_cuda_time_recursive(node: _ModuleTreeNode):
            if node.is_leaf and (gpu_kineto_event :=
                                 self._get_kineto_gpu_event(node)):
                return gpu_kineto_event.duration_us()
            else:
                cumulative_cuda_time = 0
                for child in node.children:
                    cumulative_cuda_time += _cumulative_cuda_time_recursive(
                        child)
                return cumulative_cuda_time

        return _cumulative_cuda_time_recursive(node)

    def _total_cuda_time(self):
        return sum(
            [self._cumulative_cuda_time(root) for root in self._module_tree])

    def print_model_table(self):

        @dataclass
        class ModelRow:
            name: str
            cpu_time_us: float
            cuda_time_us: float
            pct_cuda_time: float
            trace: str

        column_widths = dict(name=60,
                             cpu_time_us=12,
                             cuda_time_us=12,
                             pct_cuda_time=12,
                             trace=60)

        model_table_printer = TablePrinter(ModelRow, column_widths)
        rows: List[ModelRow] = []

        total_cuda_time = self._total_cuda_time()

        def _df_traversal(node: _ModuleTreeNode, indent: int = 0):
            row = None
            if event_has_module(node.event):
                name = indent_string(event_module_repr(node.event), indent)
                cumulative_cuda_time = self._cumulative_cuda_time(node)
                row = ModelRow(
                    name=name,
                    cpu_time_us=node.event.duration_time_ns / 1000,
                    cuda_time_us=cumulative_cuda_time,
                    pct_cuda_time=(cumulative_cuda_time / total_cuda_time) *
                    100,
                    trace="")
                rows.append(row)
                indent += 1
            elif node.is_leaf and (gpu_kineto_event :=
                                   self._get_kineto_gpu_event(node)):
                name = indent_string(gpu_kineto_event.name(), indent)
                row = ModelRow(name=name,
                               cpu_time_us=0,
                               cuda_time_us=gpu_kineto_event.duration_us(),
                               pct_cuda_time=(gpu_kineto_event.duration_us() /
                                              total_cuda_time) * 100,
                               trace=node.trace)
                rows.append(row)

            for child in node.children:
                _df_traversal(child, indent=indent)

        for root in self._module_tree:
            _df_traversal(root)

        model_table_printer.print_table(
            rows,
            filter=lambda row: row.cuda_time_us == 0 and row.cpu_time_us == 0)

    def print_summary_table(self):

        @dataclass
        class SummaryEntry:
            cuda_time_us: float = 0
            invocations: int = 0

        summary = {}

        def _df_traversal(node: _ModuleTreeNode,
                          indent=0,
                          module_trace: Tuple[str] = ()):
            if event_has_module(node.event):
                module_trace = module_trace + (event_module_repr(node.event), )
                indent += 1
            elif node.is_leaf and (gpu_kineto_event :=
                                   self._get_kineto_gpu_event(node)):
                summary_entry = summary
                for module_name in module_trace:
                    summary_entry = summary_entry.setdefault(module_name, {})
                summary_entry = summary_entry.setdefault(
                    gpu_kineto_event.name(), SummaryEntry())
                summary_entry.cuda_time_us += gpu_kineto_event.duration_us()
                summary_entry.invocations += 1

            for child in node.children:
                _df_traversal(child, indent=indent, module_trace=module_trace)

        for root in self._module_tree:
            _df_traversal(root)

        total_cuda_time = self._total_cuda_time()

        @dataclass
        class SummaryRow:
            name: str
            cuda_time_us: float
            pct_cuda_time: float
            invocations: int

        column_widths = dict(name=80,
                             cuda_time_us=12,
                             pct_cuda_time=12,
                             invocations=15)

        summary_table_printer = TablePrinter(SummaryRow, column_widths)
        rows: List[SummaryRow] = []

        def _construct_summary_rows(summary: Dict[str, Union[Dict,
                                                             SummaryEntry]],
                                    indent=0):

            cumulative_cuda_time_us = 0
            for key, value in summary.items():
                if isinstance(value, dict):
                    row = SummaryRow(name=indent_string(key, indent),
                                     cuda_time_us=0,
                                     pct_cuda_time=0,
                                     invocations="")
                    rows.append(row)
                    row.cuda_time_us = _construct_summary_rows(value,
                                                               indent=indent +
                                                               1)
                    row.pct_cuda_time = (row.cuda_time_us /
                                         total_cuda_time) * 100
                    cumulative_cuda_time_us += row.cuda_time_us
                elif isinstance(value, SummaryEntry):
                    row = SummaryRow(
                        name=indent_string(key, indent),
                        cuda_time_us=value.cuda_time_us,
                        pct_cuda_time=(value.cuda_time_us / total_cuda_time) *
                        100,
                        invocations=value.invocations)
                    cumulative_cuda_time_us += value.cuda_time_us
                    rows.append(row)

            return cumulative_cuda_time_us

        _construct_summary_rows(summary)

        summary_table_printer.print_table(
            rows, filter=lambda row: row.cuda_time_us == 0)


class nm_profile(profile):

    def __init__(self):
        super().__init__(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            with_modules=True,
            experimental_config=_ExperimentalConfig(verbose=True))

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.results = NMProfileResults(self.profiler.kineto_results,
                                        self.events())
