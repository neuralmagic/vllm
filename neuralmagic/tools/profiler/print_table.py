import argparse
import json
import pandas as pd
import copy

from collections import defaultdict
from dataclasses import dataclass, field, asdict
from vllm.profiler.nm_profile import SummaryStatsEntry, ModelStatsEntry
from vllm.profiler.utils import (indent_string, TablePrinter, event_has_module,
                                 event_is_torch_op, event_module_repr,
                                 event_torch_op_stack_trace)
from typing import Dict, List, Union, Optional, Tuple, Callable, TypeAlias
from torch.profiler import profile, ProfilerActivity
from torch.autograd.profiler import FunctionEvent
from torch._C._autograd import _ProfilerResult, _KinetoEvent, DeviceType
from torch._C._profiler import _EventType, _ProfilerEvent, _ExperimentalConfig


def flatten_entries(entry_cls, profile_dict: Dict):
    entries_and_depth = []

    def get_entries(node, curr_depth=0):
        entries_and_depth.append((entry_cls(**node["entry"]), curr_depth))

        for child in node["children"]:
            get_entries(
                child,
                curr_depth=curr_depth + 1,
            )

    for root in profile_dict:
        get_entries(root)

    return entries_and_depth


def print_summary_table(rows, column_widths: Dict[str, int] = None):
    _column_widths = dict(name=80,
                          cuda_time_us=12,
                          pct_cuda_time=12,
                          invocations=15)
    if column_widths:
        _column_widths.update(**column_widths)
    filtered_summary_table = [
        (depth, row)
        for depth, row in self._flatten_stats_tree(self._summary_stats_tree)
        if row.cuda_time_us > 0
    ]
    TablePrinter(SummaryStatsEntry, _column_widths).print_table(
        self._indent_row_names_based_on_depth(
            filtered_summary_table,
            indent_style=lambda indent: "|" + "-" * indent + " "))


def print_model_table(self, column_widths: Dict[str, int] = None):
    _column_widths = dict(name=60,
                          cpu_time_us=12,
                          cuda_time_us=12,
                          pct_cuda_time=12,
                          trace=60)
    if column_widths:
        _column_widths.update(**column_widths)
    filtered_model_table = [
        (depth, row)
        for depth, row in self._flatten_stats_tree(self._model_stats_tree)
        if row.cuda_time_us > 0 or row.cpu_time_us > 0
    ]
    TablePrinter(ModelStatsEntry, _column_widths).print_table(
        self._indent_row_names_based_on_depth(
            filtered_model_table,
            indent_style=lambda indent: "|" + "-" * indent + " "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--json-trace",
                        type=str,
                        required=True,
                        help="json trace file output by "
                        "examples/offline_profile.py")
    parser.add_argument("--phase",
                        type=str,
                        choices=["prefill", "decode"],
                        required=True,
                        help="The phase to print the table for.")
    parser.add_argument("--table",
                        type=str,
                        choices=["summary", "model"],
                        default="summary",
                        help="Which table to print, the summary table or the "
                        "layerwise model table")

    args = parser.parse_args()

    with open(args.json_trace, "r") as f:
        profile_data = json.load(f)

    if args.table == "summary":
        entries_and_depths = flatten_entries(
            SummaryStatsEntry, profile_data[args.phase]["summary_stats"])
        column_widths = dict(name=80,
                             cuda_time_us=12,
                             pct_cuda_time=12,
                             invocations=15)
    elif args.table == "model":
        entries_and_depths = flatten_entries(
            ModelStatsEntry, profile_data[args.phase]["model_stats"])
        column_widths = dict(name=60,
                             cpu_time_us=12,
                             cuda_time_us=12,
                             pct_cuda_time=12,
                             trace=60)

    # ident entry names based on the depth
    entries = []
    for entry, depth in entries_and_depths:
        entry.name = indent_string(
            entry.name,
            indent=depth,
            indent_style=lambda indent: "|" + "-" * indent + " ")
        entries.append(entry)

    TablePrinter(type(entries[0]), column_widths).print_table(entries)
