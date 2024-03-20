import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def trim_string_back(string: str, width: int):
    if len(string) > width:
        offset = len(string) - width + 3
        string = string[:-offset]
        if len(string) > 3:
            string = string + "..."
    return string


def abbreviate_known_names(name: str):
    abbreviations = {
        "MergedColumnParallelLinear": "MCPLinear",
        "QKVParallelLinear": "QKVPLinear",
        "RowParallelLinear": "RPLinear",
        "weight=": "w=",
        "bfloat16": "bf16",
        "float16": "f16",
    }
    for key, value in abbreviations.items():
        name = name.replace(key, value)
    return name


def shorten_plot_legend_strings(l, max_char_len: int):
    for t in l.get_texts():
        t.set_text(
            trim_string_back(abbreviate_known_names(t.get_text()),
                             max_char_len))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_trace", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--depth", type=int, default=-2)
    parser.add_argument("--ignore_sampler", action='store_true')

    args = parser.parse_args()
    depth = args.depth
    ignore_sampler = args.ignore_sampler
    output = args.output

    with open(args.json_trace, "r") as f:
        profile_data = json.load(f)

    prefill_entries = []
    decode_entries = []

    def largest_dist_from_leaf(node, depth=0):
        if len(node["children"]) == 0:
            return depth
        return max([
            largest_dist_from_leaf(child, depth=depth + 1)
            for child in node["children"]
        ])

    def get_entries_at_depth(depth, entries, node, curr_depth=0):
        if ignore_sampler and node["entry"]["name"] == "Sampler":
            return

        if (depth >= 0 and depth == curr_depth) or (
                depth < 0
                and largest_dist_from_leaf(node) == (abs(depth) - 1)):
            entries.append(node["entry"])
        for child in node["children"]:
            get_entries_at_depth(depth,
                                 entries,
                                 child,
                                 curr_depth=curr_depth + 1)

    for root in profile_data["prefill"]["summary_stats"]:
        get_entries_at_depth(depth, prefill_entries, root)
    for root in profile_data["decode"]["summary_stats"]:
        get_entries_at_depth(depth, decode_entries, root)

    prefill_df = pd.DataFrame(prefill_entries)
    prefill_df["phase"] = "prefill"
    decode_df = pd.DataFrame(decode_entries)
    decode_df["phase"] = "decode"
    df = pd.concat([prefill_df, decode_df])
    df["cuda_time_ms"] = df["cuda_time_us"] / 1000

    fig, axes = plt.subplots(2, figsize=(5, 8), sharex=True)

    def plot_metric(metric: str, ax, add_totals=False):
        pivoted_df = df.pivot_table(index="phase",
                                    columns="name",
                                    values=metric,
                                    aggfunc=np.sum)
        pivoted_df.plot.bar(stacked=True, legend=False, ax=ax)
        ax.set_ylabel(metric)

        if add_totals:
            ax.bar_label(ax.containers[-1])

    plot_metric("cuda_time_ms", ax=axes[0], add_totals=True)
    plot_metric("pct_cuda_time", ax=axes[1])

    handles, labels = plt.gca().get_legend_handles_labels()
    l = fig.legend(handles,
                   labels,
                   loc='center left',
                   bbox_to_anchor=(0.93, 0.5))
    shorten_plot_legend_strings(l, 50)

    context = profile_data["context"]
    plt.suptitle(f"{context['model_name']}\n"
                 f"Batch={context['batch_size']}, "
                 f"PromptLen={context['prompt_len']}, "
                 f"NumGpus={context['num_gpus']}"
                 f"{', Sparse' if context['is_sparse'] else ''}")
    plt.savefig(output, bbox_inches='tight')
    print("Created: ", output)
