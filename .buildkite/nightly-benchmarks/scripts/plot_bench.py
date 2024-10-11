import json
import argparse
from pathlib import Path
import glob
import pandas as pd
from typing import List, Any, Dict
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from itertools import product


def get_qps_from_test_name(x: str) -> float:
    # trt_llama8B_tp1_sharegpt_qps_16
    parts = x.split('_')
    assert len(parts) > 2
    parts = parts[-2:]
    assert parts[0] == 'qps'
    return float(parts[1])

def get_qps(df: pd.DataFrame) -> List[int]:
    test_names = df.loc[:,'Test name']
    qps = list(map(lambda test_name: get_qps_from_test_name(test_name), test_names))
    return qps

def get_num_prompts(df: pd.DataFrame) -> List[int]:
    successful_reqs: List[int] = list(df.loc[:,'Successful req.'])
    assert all([x == successful_reqs[0] for x in successful_reqs])
    return successful_reqs

def trim_test_name(df: pd.DataFrame) -> pd.DataFrame:
    def trim(x: str) -> str:
        # trt_llama8B_tp1_sharegpt_qps_16
        parts = x.split('_')
        assert len(parts) > 2
        parts = parts[:-2]
        return '_'.join(parts)

    test_names = list(df.loc[:,'Test name'])
    test_names = list(map(lambda x: str(trim(x)), test_names))
    updated = pd.DataFrame({'Test name': test_names})

    df.update(updated)
    return df

def dataset_name_from_test_name(x: str):
    #trt_llama70B_tp4_sharegpt_qps_32
    parts = x.lower().split('_')
    assert 'sharegpt' in parts or 'sonnet' in parts
    if 'sharegpt' in parts:
        return 'sharegpt'
    if 'sonnet' in parts:
        return 'sonnet'
    raise ValueError(f"Cant decipher dataset name from test name {x}")

def tp_size_from_test_name(x: str):
    #trt_llama70B_tp4_sharegpt_qps_32
    parts = x.lower().split('_')
    parts = list(filter(lambda x: x.startswith("tp"), parts))
    assert len(parts) == 1, f"Cant decipher tp_size from test name {x}"
    return parts[0]


def process_df(df: pd.DataFrame, input_dir: Path):

    test_names = df.loc[:,'Test name']
    all_qps = []
    all_model_id = []
    all_num_prompts = []
    all_dataset_name = []
    all_tp_sizes = []
    for test_name in test_names:
        test_json_file = f'{test_name}.json'
        test_json_file = input_dir / test_json_file
        test_data = None
        with open(str(test_json_file), 'r') as f:
            test_data = json.load(f)

        model_id = test_data['model_id'].replace('/', '_').replace('-', '_')
        qps = float(test_data['request_rate'])
        num_prompts = int(test_data['num_prompts'])
        dataset_name = dataset_name_from_test_name(test_name) 
        tp_size = tp_size_from_test_name(test_name)

        all_qps.append(qps)
        all_model_id.append(model_id)
        all_num_prompts.append(num_prompts)
        all_dataset_name.append(dataset_name)
        all_tp_sizes.append(tp_size)

    df.insert(2, "dataset_name", all_dataset_name, True)
    df.insert(2, "tp_size", all_tp_sizes, True)
    df.insert(2, "qps", all_qps, True)
    df.insert(2, "num_prompts", all_num_prompts, True)
    df.insert(2, "model_id", all_model_id, True)

    return trim_test_name(df)

def plot_model_metric(df: pd.DataFrame, all_qps: List[float], all_servers: List[str],
                      ax: matplotlib.axes, server_plot_colors: List[Any], metric_name: str, metric_unit: str,
                      dataset_name: str):

    assert len(all_servers) == len(server_plot_colors)

    def trim_label(x: str, dataset_name: str):
        # trt_llama8B_tp1_sharegpt_qps_16
        parts = x.split('_')
        # remove parts that reference dataset / 
        parts = list(filter(lambda y: not ('llama' in y.lower() or y.lower().startswith('tp') or dataset_name in y.lower()),parts))
        # use the rest of the parts:w
        return "_".join(parts)

    ax.set_title(metric_name)
    ax.set_ylabel(metric_unit)
    ax.set_xlabel("qps")

    Ys = []
    labels = []
    plot_colors = []
    X = list(map(lambda x: str(x), all_qps))
    for server, plot_color in zip(all_servers, server_plot_colors):
        Y = []
        for qps in all_qps:
            y = df.query(f'`Test name` == "{server}" and qps == {qps}')
            assert len(y) == 1
            y = list(y.loc[:, metric_name])
            assert len(y) == 1
            y = y[0]
            Y.append(y)
        Ys.append(Y)
        labels.append(trim_label(server, dataset_name))
        plot_colors.append(plot_color)

    for y, label, plot_color in zip(Ys, labels, plot_colors):
        ax.plot(X, y, label=label, color=plot_color, linewidth=2, markersize=16, marker='*')

        #ax.scatter(X, y, label=label, c=[plot_color] * len(X), s=32, marker='_')

    ax.set_title(metric_name)

    #ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    #ax.xaxis.set_major_locator(mticker.MultipleLocator(base=10.0))
    ax.tick_params(axis='x', rotation=90) # set tick rotation
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10)) 
    ax.margins(x=0.1)
    ax.set_title(metric_name)
    ax.grid(axis='y')
    #ax.legend()


def tabulate_metrics(df: pd.DataFrame, model_id: str, dataset: str, tp_size: str):

    all_qps = list(set(df.loc[:, 'qps']))
    all_qps = sorted(all_qps)

    all_servers = list(set(df.loc[:, 'Test name']))
    assert len(all_servers) == 9

    #trt_servers = list(filter(lambda x: x.startswith('trt'), all_servers))
    #vllm_servers = list(filter(lambda x: x.startswith('vllm'), all_servers))
    #all_servers = trt_servers + sorted(vllm_servers)
    #assert len(all_servers) == 9

    all_gpus = list(set(df.loc[:, 'GPU']))
    assert len(all_gpus) == 1
    gpu = all_gpus[0]

    all_num_prompts = list(df.loc[:, 'num_prompts'])
    assert all([x == all_num_prompts[0] for x in all_num_prompts])
    num_prompts = all_num_prompts[0]

    table_cols = ['Test name',
                  'Tput (req/s)',
                  'Mean TTFT (ms)',
                  'Median TTFT (ms)',
                  'Mean TPOT (ms)',
                  'Median TPOT (ms)',
                  'Mean ITL (ms)',
                  'Median ITL (ms)']

    header = f'{model_id} {tp_size}x{gpu} {dataset} num-prompts-{num_prompts}'
    print (f"{header}")
    for qps in all_qps:
        print(f"QPS {qps}") 
        qps_df = df.query(fr'qps == {qps}')
        selected_columns = qps_df.loc[:, table_cols] 
        selected_columns = selected_columns.sort_values(by=['Test name'])
        print(selected_columns.to_csv())


def plot_model_df(df: pd.DataFrame, model_id: str, dataset: str, tp_size: str):

    all_qps = list(set(df.loc[:, 'qps']))
    all_qps = sorted(all_qps)

    all_servers = sorted(list(set(df.loc[:, 'Test name'])))
    assert len(all_servers) == 9

    # server_plot_colors
    cmap = plt.get_cmap('tab20')
    server_plot_colors = []
    for i in range(len(all_servers)):
        server_plot_colors.append(cmap(i / (len(all_servers) - 1)))

    fig, axs = plt.subplots(4, 2, figsize=(8, 14))
    plot_model_metric(df, all_qps, all_servers, axs[0, 0], server_plot_colors, 'Tput (req/s)', "req/s", dataset)
    plot_model_metric(df, all_qps, all_servers, axs[1, 0], server_plot_colors, 'Mean TTFT (ms)', "ms", dataset)
    plot_model_metric(df, all_qps, all_servers, axs[1, 1], server_plot_colors, 'Median TTFT (ms)', "ms", dataset)
    plot_model_metric(df, all_qps, all_servers, axs[2, 0], server_plot_colors, 'Mean TPOT (ms)', "ms", dataset)
    plot_model_metric(df, all_qps, all_servers, axs[2, 1], server_plot_colors, 'Median TPOT (ms)', "ms", dataset)
    plot_model_metric(df, all_qps, all_servers, axs[3, 0], server_plot_colors, 'Mean ITL (ms)', "ms", dataset)
    plot_model_metric(df, all_qps, all_servers, axs[3, 1], server_plot_colors, 'Median ITL (ms)', "ms", dataset)

    all_gpus = list(set(df.loc[:, 'GPU']))
    assert len(all_gpus) == 1
    gpu = all_gpus[0]

    all_num_prompts = list(df.loc[:, 'num_prompts'])
    assert all([x == all_num_prompts[0] for x in all_num_prompts])
    num_prompts = all_num_prompts[0]

    plot_title = f'{model_id.replace("/","_")}_{tp_size}x{gpu}_{dataset}_num-prompts-{num_prompts}.png'
    fig.suptitle(plot_title)

    #figure = plt.gcf()
    #plt.legend(loc='upper right', bbox_to_anchor=(0.5, 0.5), ncol=len(all_servers) // 2)
    #fig.legend()

    lines, labels = axs[0,0].get_legend_handles_labels()
    #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels,  bbox_to_anchor=(0.95, 0.95), loc='upper right')

    plt.tight_layout()
    #plt.show()
    plt.savefig(plot_title)
    plt.close(fig)

def plot_df(df: pd.DataFrame):

    def filter_df(df: pd.DataFrame,
                  model_id: str,
                  dataset_name: str,
                  tp_size: str) -> pd.DataFrame:
        filtered_df = df.query(fr'model_id == "{model_id}" and dataset_name == "{dataset_name}" and tp_size == "{tp_size}" ')
        return filtered_df

    models = sorted(list(set(df.loc[:, 'model_id'])))
    datasets = sorted(list(set(df.loc[:, 'dataset_name'])))
    tp_sizes = sorted(list(set(df.loc[:, 'tp_size'])))

    for model_id, dataset, tp_size in product(models, datasets, tp_sizes):
        if model_id == '_models_Meta_Llama_3_70B_Instruct':
            continue
        filtered_df = filter_df(df, model_id, dataset, tp_size)
        if len(filtered_df) == 0:
            continue
        tabulate_metrics(filtered_df, model_id, dataset, tp_size)
        plot_model_df(filtered_df, model_id, dataset, tp_size)


def make_dict(json_file_path : str, gpu_name : str) -> Dict:
    #'date', 'backend', 'model_id', 'tokenizer_id', 'best_of', 'num_prompts', 'request_rate', 'duration', 'completed', 'total_input_tokens', 'total_output_tokens',
    #  'request_throughput', 'output_throughput', 'total_token_throughput', 'input_lens', 'output_lens', 'ttfts', 'itls', 'generated_texts', 'errors', 'mean_ttft_ms', 
    # 'median_ttft_ms', 'std_ttft_ms', 'p99_ttft_ms', 'mean_tpot_ms', 'median_tpot_ms', 'std_tpot_ms', 'p99_tpot_ms', 'mean_itl_ms', 'median_itl_ms', 'std_itl_ms', 'p99_itl_ms'

    data = None
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # make sure the run was sucessful
    assert data['completed'] == data['num_prompts']

    d = {}
    d['Test name'] = Path(json_file_path).stem
    d['Tput (req/s)'] = data['request_throughput']
    d['Mean TTFT (ms)'] = data['mean_ttft_ms']
    d['Median TTFT (ms)'] = data['median_ttft_ms']
    d['Mean TPOT (ms)'] = data['mean_tpot_ms']
    d['Median TPOT (ms)'] = data['median_tpot_ms']
    d['Mean ITL (ms)'] = data['mean_itl_ms']
    d['Median ITL (ms)'] = data['median_itl_ms']
    d['qps'] = float(data['request_rate'])
    d['num_prompts'] = data['num_prompts'] 
    d['model_id'] = data['model_id']
    d['dataset_name'] = dataset_name_from_test_name(d['Test name'])
    d['tp_size'] = tp_size_from_test_name(d['Test name'])
    d['Engine'] = data['backend']
    d['GPU'] = gpu_name
    return d

def make_dataframe(json_file_paths: List[str],
                   gpu_name: str) -> pd.DataFrame:

    aggregate_dict = {}

    def add_to_aggregate_dict(part_dict: Dict):
        for k in part_dict.keys():
            if aggregate_dict.get(k, None) is None:
                aggregate_dict[k] = [part_dict[k]]
            else:
                aggregate_dict[k].append(part_dict[k])

    for json_file_path  in json_file_paths:
        add_to_aggregate_dict(make_dict(json_file_path, gpu_name))

    df = pd.DataFrame.from_dict(aggregate_dict)
    return df

def main(args):
    
    input_dir : Path = Path(args.input_dir)
    assert input_dir.exists()

    if True:
        assert args.gpu_name is not None
        data_json_files = glob.glob(str(input_dir) + '/*.json')
        df : pd.DataFrame = make_dataframe(data_json_files, args.gpu_name)
        df = trim_test_name(df)
    else:
        # Get all jsons that end with with 'nightly_results.json'
        data_json_files = glob.glob(str(input_dir) + '/*nightly_results.json')
        dfs : List[pd.DataFrame] = list(map(lambda x: make_dataframe(x), data_json_files))
        for df in dfs:
            df = process_df(df, input_dir)
        df = pd.concat(dfs)

    plot_df(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    description='plot data from benchmarks. group by model name qps and num-prompts')
    parser.add_argument('-i', '--input-dir',
                        type=str,
                        default=None,
                        help="input directory containing the nightly results json")
    parser.add_argument('--gpu-name',
                        type=str,
                        default="dummy",
                        help="GPU name - Pass this if plotting from the benchmark_serving output jsons")
    args = parser.parse_args()
    main(args)

    

