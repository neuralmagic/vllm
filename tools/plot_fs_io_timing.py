# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Parse FSIO_TIMING lines emitted by vllm/v1/kv_offload/tiering/fs/io.py out of
a (redirected) vLLM log file and render an interactive Plotly HTML dashboard
of store_block/load_block I/O timing.

Usage:
    python tools/plot_fs_io_timing.py /path/to/vllm.log -o fs_io_timing.html
"""

import argparse
import re

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Matches the FSIO_TIMING payload regardless of what (if anything) precedes
# it on the line -- only the first physical line of a flushed batch carries
# the normal log prefix (timestamp/level/module); the rest are raw lines.
_LINE_RE = re.compile(r"FSIO_TIMING\s+(.*)")
_FIELD_RE = re.compile(r"(\w+)=(\S+)")

_FIELD_TYPES = {
    "ts": float,
    "wall_ms": float,
    "cpu_ms": float,
    "nvcsw": int,
    "nivcsw": int,
}

LOAD_OPS = ["load.open", "load.read", "load.close"]
STORE_OPS = [
    "store.ensure_dirs",
    "store.open",
    "store.write",
    "store.close",
    "store.replace",
]

# (title, kinds, group_by_kind) for each of the 6 dashboard panels.
_PLOTS = [
    ("load_block total wall time", ["load.total"], False),
    ("store_block total wall time", ["store.total"], False),
    ("load_block per-operation wall time", LOAD_OPS, True),
    ("store_block per-operation wall time", STORE_OPS, True),
    ("load_block os.readv wall time", ["load.read"], False),
    ("store_block os.write wall time", ["store.write"], False),
]


def parse_log(path: str) -> pd.DataFrame:
    """Parse all FSIO_TIMING lines in *path* into a DataFrame."""
    records = []
    with open(path) as f:
        for line in f:
            match = _LINE_RE.search(line)
            if match is None:
                continue
            fields: dict[str, object] = dict(_FIELD_RE.findall(match.group(1)))
            for key, cast in _FIELD_TYPES.items():
                if key in fields:
                    fields[key] = cast(fields[key])
            records.append(fields)

    if not records:
        raise SystemExit(f"No FSIO_TIMING lines found in {path}")

    df = pd.DataFrame.from_records(records)
    df["t"] = df["ts"] - df["ts"].min()
    return df


def build_dashboard(df: pd.DataFrame) -> go.Figure:
    """Build a 3x2 grid of scatter panels, one per entry in _PLOTS."""
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[title for title, _, _ in _PLOTS],
    )

    for idx, (_title, kinds, group_by_kind) in enumerate(_PLOTS):
        row, col = idx // 2 + 1, idx % 2 + 1
        subset = df[df["kind"].isin(kinds)]

        series = subset.groupby("kind") if group_by_kind else [(kinds[0], subset)]
        for kind, group in series:
            if group.empty:
                continue
            fig.add_trace(
                go.Scattergl(
                    x=group["t"],
                    y=group["wall_ms"],
                    mode="markers",
                    marker=dict(size=4, opacity=0.4),
                    name=kind,
                    legendgroup=kind,
                    showlegend=group_by_kind,
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="time (s)", row=row, col=col)
        fig.update_yaxes(title_text="wall time (ms)", row=row, col=col)

    fig.update_layout(height=1400, width=1400, title_text="FSIO Timing")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logfile", help="Path to a vLLM log file (or redirected stdout)."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="fs_io_timing.html",
        help="Path to write the interactive HTML dashboard to.",
    )
    args = parser.parse_args()

    df = parse_log(args.logfile)
    fig = build_dashboard(df)
    fig.write_html(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
