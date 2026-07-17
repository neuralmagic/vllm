# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Parse FSIO_TIMING lines emitted by vllm/v1/kv_offload/tiering/fs/io.py out of
a (redirected) vLLM log file and plot store_block/load_block I/O timing.

Usage:
    python tools/plot_fs_io_timing.py /path/to/vllm.log
"""

import argparse
import re

import matplotlib.pyplot as plt
import pandas as pd

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


def plot_wall_time(
    df: pd.DataFrame,
    kinds: list[str],
    title: str,
    group_by_kind: bool = False,
) -> None:
    """Scatter wall_ms vs elapsed time, one figure, optionally one series per kind."""
    subset = df[df["kind"].isin(kinds)]
    fig, ax = plt.subplots(figsize=(10, 5))
    if subset.empty:
        ax.set_title(f"{title} (no data)")
    elif group_by_kind:
        for kind, group in subset.groupby("kind"):
            ax.scatter(group["t"], group["wall_ms"], s=4, alpha=0.4, label=kind)
        ax.legend(markerscale=3)
        ax.set_title(title)
    else:
        ax.scatter(subset["t"], subset["wall_ms"], s=4, alpha=0.4)
        ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("wall time (ms)")
    fig.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logfile", help="Path to a vLLM log file (or redirected stdout)."
    )
    args = parser.parse_args()

    df = parse_log(args.logfile)

    plot_wall_time(df, ["load.total"], "load_block total wall time")
    plot_wall_time(df, ["store.total"], "store_block total wall time")
    plot_wall_time(
        df, LOAD_OPS, "load_block per-operation wall time", group_by_kind=True
    )
    plot_wall_time(
        df, STORE_OPS, "store_block per-operation wall time", group_by_kind=True
    )
    plot_wall_time(df, ["load.read"], "load_block os.readv wall time")
    plot_wall_time(df, ["store.write"], "store_block os.write wall time")

    plt.show()


if __name__ == "__main__":
    main()
