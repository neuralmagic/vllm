#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline analysis for the FS KV-offload tier timing log produced by
``vllm.v1.kv_offload.tiering.fs.timing_debug.TimingRecorder`` when
``VLLM_KV_OFFLOAD_FS_TIMING_LOG`` is set to a file path.

Standalone script: only depends on the Python standard library, so it can
be run outside a vLLM environment against a timing log collected elsewhere.

Usage:
    python tools/analyze_fs_tier_timing.py /path/to/timing.log
    python tools/analyze_fs_tier_timing.py /path/to/timing.log --kind load --top 20
    python tools/analyze_fs_tier_timing.py /path/to/timing.log --job-id 12345
"""

import argparse
import statistics
from dataclasses import dataclass


@dataclass
class EnqueueRecord:
    t: float
    job_id: str
    kind: str
    n_tasks: int
    n_batches: int
    queue_depth_before: int


@dataclass
class DequeueRecord:
    t: float
    job_id: str
    thread: str
    batch_no: int
    batch_size: int
    queue_wait: float
    queue_depth_after: int


@dataclass
class FinishRecord:
    t: float
    job_id: str
    thread: str
    batch_no: int
    duration: float
    success: bool


@dataclass
class JobRecord:
    t: float
    job_id: str
    kind: str
    n_batches: int
    queue_wait_first: float
    pickup_spread: float
    span: float
    sum_batch_time: float
    parallel_efficiency: float


def parse(path: str):
    enqueues: list[EnqueueRecord] = []
    dequeues: list[DequeueRecord] = []
    finishes: list[FinishRecord] = []
    jobs: list[JobRecord] = []
    n_bad = 0
    with open(path) as f:
        for line in f:
            fields = line.strip().split(",")
            if not fields or not fields[0]:
                continue
            tag = fields[0]
            try:
                if tag == "E":
                    _, t, job_id, kind, n_tasks, n_batches, qdb = fields
                    enqueues.append(
                        EnqueueRecord(
                            float(t), job_id, kind, int(n_tasks), int(n_batches), int(qdb)
                        )
                    )
                elif tag == "D":
                    _, t, job_id, thread, batch_no, batch_size, qw, qda = fields
                    dequeues.append(
                        DequeueRecord(
                            float(t),
                            job_id,
                            thread,
                            int(batch_no),
                            int(batch_size),
                            float(qw),
                            int(qda),
                        )
                    )
                elif tag == "F":
                    _, t, job_id, thread, batch_no, duration, success = fields
                    finishes.append(
                        FinishRecord(
                            float(t),
                            job_id,
                            thread,
                            int(batch_no),
                            float(duration),
                            bool(int(success)),
                        )
                    )
                elif tag == "J":
                    _, t, job_id, kind, n_batches, qwf, spread, span, sbt, eff = fields
                    jobs.append(
                        JobRecord(
                            float(t),
                            job_id,
                            kind,
                            int(n_batches),
                            float(qwf),
                            float(spread),
                            float(span),
                            float(sbt),
                            float(eff),
                        )
                    )
                else:
                    n_bad += 1
            except ValueError:
                # Truncated/partial line, e.g. the process was killed
                # mid-write. Skip rather than aborting the whole analysis.
                n_bad += 1
    if n_bad:
        print(f"warning: skipped {n_bad} malformed/unrecognized line(s)")
    return enqueues, dequeues, finishes, jobs


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.5),
        "p90": _percentile(values, 0.9),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _print_table(title: str, columns: dict[str, dict[str, float]]) -> None:
    print(f"\n=== {title} ===")
    columns = {k: v for k, v in columns.items() if v}
    if not columns:
        print("  (no data)")
        return
    print(f"  {'metric':<18}{'n':>8}{'mean':>12}{'p50':>12}{'p90':>12}{'p99':>12}{'max':>12}")
    for name, s in columns.items():
        print(
            f"  {name:<18}{s['n']:>8}{s['mean']:>12.4f}{s['p50']:>12.4f}"
            f"{s['p90']:>12.4f}{s['p99']:>12.4f}{s['max']:>12.4f}"
        )


def print_job_timeline(job_id: str, enqueues, dequeues, finishes, jobs) -> None:
    rows: list[tuple[float, str, object]] = []
    rows += [(r.t, "ENQUEUE", r) for r in enqueues if r.job_id == job_id]
    rows += [(r.t, "DEQUEUE", r) for r in dequeues if r.job_id == job_id]
    rows += [(r.t, "FINISH", r) for r in finishes if r.job_id == job_id]
    rows += [(r.t, "JOB_DONE", r) for r in jobs if r.job_id == job_id]
    rows.sort(key=lambda x: x[0])
    if not rows:
        print(f"No records found for job_id={job_id!r}")
        return
    t0 = rows[0][0]
    print(f"=== Timeline for job {job_id} (t relative to first event) ===")
    for t, label, r in rows:
        print(f"  t+{t - t0:9.4f}  {label:<9} {r}")


def summarize_all(enqueues, dequeues, finishes, jobs, top: int) -> None:
    print(
        f"Parsed {len(enqueues)} enqueue, {len(dequeues)} dequeue, "
        f"{len(finishes)} finish, {len(jobs)} job-summary records."
    )

    _print_table(
        "Per-job metrics (seconds unless noted)",
        {
            "queue_wait_first": _summarize([j.queue_wait_first for j in jobs]),
            "pickup_spread": _summarize([j.pickup_spread for j in jobs]),
            "span": _summarize([j.span for j in jobs]),
            "sum_batch_time": _summarize([j.sum_batch_time for j in jobs]),
            "parallel_efficiency": _summarize([j.parallel_efficiency for j in jobs]),
            "n_batches": _summarize([float(j.n_batches) for j in jobs]),
        },
    )

    _print_table(
        "Queue backlog observed at job-enqueue time",
        {"queue_depth_before": _summarize([float(e.queue_depth_before) for e in enqueues])},
    )

    worst = sorted(jobs, key=lambda j: j.pickup_spread, reverse=True)[:top]
    print(f"\n=== Top {len(worst)} jobs by pickup_spread (evidence of staggering) ===")
    print(
        f"  {'job_id':<12}{'kind':<7}{'n_batches':>10}{'queue_wait_first':>18}"
        f"{'pickup_spread':>15}{'span':>10}{'parallel_eff':>14}"
    )
    for j in worst:
        print(
            f"  {j.job_id:<12}{j.kind:<7}{j.n_batches:>10}{j.queue_wait_first:>18.4f}"
            f"{j.pickup_spread:>15.4f}{j.span:>10.4f}{j.parallel_efficiency:>14.4f}"
        )

    multi_batch = [j for j in jobs if j.n_batches > 1 and j.span > 0]
    if multi_batch:
        frac = statistics.fmean(j.pickup_spread / j.span for j in multi_batch)
        print(
            f"\nAvg(pickup_spread / span) across {len(multi_batch)} multi-batch "
            f"jobs: {frac:.3f}\n"
            "  (near 0 => a job's batches were picked up by threads together, "
            "i.e. true parallel dispatch;\n"
            "   near 1 => a job's batches were staggered across its entire "
            "lifetime, i.e. no better than serial)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log_path", help="Path to the timing log written by TimingRecorder")
    parser.add_argument("--kind", choices=["load", "store"], default=None)
    parser.add_argument(
        "--top", type=int, default=10, help="Number of worst jobs (by pickup_spread) to list"
    )
    parser.add_argument(
        "--job-id", default=None, help="Print the full E/D/F/J timeline for one job id"
    )
    args = parser.parse_args()

    enqueues, dequeues, finishes, jobs = parse(args.log_path)

    if args.job_id is not None:
        print_job_timeline(args.job_id, enqueues, dequeues, finishes, jobs)
        return

    if args.kind is not None:
        enqueues = [e for e in enqueues if e.kind == args.kind]
        jobs = [j for j in jobs if j.kind == args.kind]

    summarize_all(enqueues, dequeues, finishes, jobs, args.top)


if __name__ == "__main__":
    main()
