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
                            float(t),
                            job_id,
                            kind,
                            int(n_tasks),
                            int(n_batches),
                            int(qdb),
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
    print(
        f"  {'metric':<18}{'n':>8}{'mean':>12}{'p50':>12}{'p90':>12}{'p99':>12}{'max':>12}"
    )
    for name, s in columns.items():
        print(
            f"  {name:<18}{s['n']:>8}{s['mean']:>12.4f}{s['p50']:>12.4f}"
            f"{s['p90']:>12.4f}{s['p99']:>12.4f}{s['max']:>12.4f}"
        )


def _max_batch_size_per_job(dequeues: list[DequeueRecord]) -> dict[str, int]:
    max_batch: dict[str, int] = {}
    for d in dequeues:
        if d.batch_size > max_batch.get(d.job_id, 0):
            max_batch[d.job_id] = d.batch_size
    return max_batch


def _distinct_thread_count_per_job(dequeues: list[DequeueRecord]) -> dict[str, int]:
    threads_by_job: dict[str, set[str]] = {}
    for d in dequeues:
        threads_by_job.setdefault(d.job_id, set()).add(d.thread)
    return {job_id: len(threads) for job_id, threads in threads_by_job.items()}


def _batch_intervals_per_job(
    dequeues: list[DequeueRecord], finishes: list[FinishRecord]
) -> dict[str, list[tuple[float, float]]]:
    """job_id -> [(dequeue_t, finish_t), ...], one interval per batch,
    matched on (job_id, batch_no)."""
    starts: dict[tuple[str, int], float] = {(d.job_id, d.batch_no): d.t for d in dequeues}
    intervals: dict[str, list[tuple[float, float]]] = {}
    for f in finishes:
        start = starts.get((f.job_id, f.batch_no))
        if start is None:
            continue
        intervals.setdefault(f.job_id, []).append((start, f.t))
    return intervals


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    """Total wall-clock time covered by the union of (start, end) intervals."""
    if not intervals:
        return 0.0
    ivs = sorted(intervals)
    total = 0.0
    cur_start, cur_end = ivs[0]
    for s, e in ivs[1:]:
        if s > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    total += cur_end - cur_start
    return total


def _peak_concurrency(intervals: list[tuple[float, float]]) -> int:
    """Max number of batches simultaneously in flight at any instant."""
    events = [(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals]
    events.sort(key=lambda x: (x[0], x[1]))  # ends (-1) before starts (+1) at ties
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def plot_load_job_bars(
    enqueues: list[EnqueueRecord],
    dequeues: list[DequeueRecord],
    jobs: list[JobRecord],
    output_path: str,
) -> None:
    """Save 3 stacked bar charts (queue_wait_first, span, max batch size),
    one bar per load job, ordered chronologically by enqueue time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    job_by_id = {j.job_id: j for j in jobs if j.kind == "load"}
    max_batch_by_id = _max_batch_size_per_job(dequeues)

    load_enq = sorted(
        (e for e in enqueues if e.job_id in job_by_id), key=lambda e: e.t
    )
    if not load_enq:
        print("No load jobs found; nothing to plot.")
        return

    job_ids = [e.job_id for e in load_enq]
    queue_wait = [job_by_id[jid].queue_wait_first for jid in job_ids]
    spans = [job_by_id[jid].span for jid in job_ids]
    max_batch = [max_batch_by_id.get(jid, 0) for jid in job_ids]

    x = range(len(job_ids))
    fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)

    axes[0].bar(x, queue_wait, width=1.0, color="tab:red")
    axes[0].set_ylabel("queue_wait_first (s)")
    axes[0].set_title("Time in queue before first batch pickup, per load job")

    axes[1].bar(x, spans, width=1.0, color="tab:blue")
    axes[1].set_ylabel("span (s)")
    axes[1].set_title("Span (first batch pickup -> job completion), per load job")

    axes[2].bar(x, max_batch, width=1.0, color="tab:green")
    axes[2].set_ylabel("max batch size (blocks)")
    axes[2].set_title("Maximum batch size, per load job")
    axes[2].set_xlabel("Load job index (chronological, by enqueue order)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Wrote {len(job_ids)}-job plot to {output_path}")


def plot_thread_count_histogram(
    dequeues: list[DequeueRecord],
    jobs: list[JobRecord],
    output_path: str,
    max_threads: int = 32,
) -> None:
    """Save a histogram: number of load jobs bucketed by how many distinct
    threads processed their batches (1 thread, 2 threads, ..., max_threads)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    load_job_ids = {j.job_id for j in jobs if j.kind == "load"}
    n_threads_by_id = _distinct_thread_count_per_job(
        [d for d in dequeues if d.job_id in load_job_ids]
    )
    if not n_threads_by_id:
        print("No load jobs found; nothing to plot.")
        return

    buckets = list(range(1, max_threads + 1))
    counts = [0] * len(buckets)
    overflow = 0
    for n in n_threads_by_id.values():
        if 1 <= n <= max_threads:
            counts[n - 1] += 1
        else:
            overflow += 1
    if overflow:
        print(f"warning: {overflow} job(s) used >{max_threads} distinct threads")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(buckets, counts, width=0.9, color="tab:purple")
    ax.set_xlabel("Number of distinct threads that processed the job's batches")
    ax.set_ylabel("Number of load jobs")
    ax.set_title("Distribution of distinct-thread count across load jobs")
    ax.set_xticks(buckets)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Wrote {len(n_threads_by_id)}-job histogram to {output_path}")


def plot_thread_count_chronological(
    enqueues: list[EnqueueRecord],
    dequeues: list[DequeueRecord],
    jobs: list[JobRecord],
    output_path: str,
    n_deciles: int = 10,
) -> None:
    """Save a 2-panel plot showing how the distinct-thread-count histogram
    buckets are distributed over the course of the run, for load jobs.

    Top panel: one point per load job (chronological order by enqueue time)
    vs. how many distinct threads processed it -- shows whether low-thread
    jobs cluster at particular points in the run.
    Bottom panel: the run split into `n_deciles` equal-sized chronological
    chunks, showing the mean thread count and the fraction of jobs that hit
    the full 16-thread bucket in each chunk.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    job_by_id = {j.job_id: j for j in jobs if j.kind == "load"}
    n_threads_by_id = _distinct_thread_count_per_job(
        [d for d in dequeues if d.job_id in job_by_id]
    )
    load_enq = sorted(
        (e for e in enqueues if e.job_id in job_by_id), key=lambda e: e.t
    )
    if not load_enq:
        print("No load jobs found; nothing to plot.")
        return

    job_ids = [e.job_id for e in load_enq]
    n_threads = [n_threads_by_id.get(jid, 0) for jid in job_ids]
    x = range(len(job_ids))

    # Chunk into n_deciles equal-sized windows (by job count, i.e. chronological
    # order), and summarize each.
    n = len(job_ids)
    chunk_size = max(1, -(-n // n_deciles))  # ceil division
    chunk_x, chunk_mean, chunk_full16_frac = [], [], []
    for start in range(0, n, chunk_size):
        chunk = n_threads[start : start + chunk_size]
        if not chunk:
            continue
        chunk_x.append(start + len(chunk) / 2)
        chunk_mean.append(statistics.fmean(chunk))
        chunk_full16_frac.append(sum(1 for c in chunk if c == 16) / len(chunk))

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    axes[0].scatter(x, n_threads, s=4, alpha=0.4, color="tab:purple")
    axes[0].set_ylabel("distinct threads")
    axes[0].set_title("Distinct threads used per load job, chronological order")
    axes[0].set_xlabel("Load job index (chronological, by enqueue order)")
    axes[0].set_ylim(0, 17)

    bar_width = chunk_size * 0.8
    ax2 = axes[1]
    ax2.bar(chunk_x, chunk_mean, width=bar_width, color="tab:purple", alpha=0.7, label="mean distinct threads")
    ax2.set_ylabel("mean distinct threads")
    ax2.set_ylim(0, 17)
    ax2.set_xlabel(f"Load job index, chunked into {len(chunk_x)} windows of ~{chunk_size} jobs")
    ax2.set_title("Per-window mean thread count and fraction reaching all 16 threads")

    ax3 = ax2.twinx()
    ax3.plot(chunk_x, [100 * f for f in chunk_full16_frac], color="tab:orange", marker="o", label="% jobs using all 16 threads")
    ax3.set_ylabel("% jobs using all 16 threads")
    ax3.set_ylim(0, 105)

    fig.legend(loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=150)
    print(f"Wrote {len(job_ids)}-job chronological bucket plot to {output_path}")


def print_gil_contention_report(
    enqueues: list[EnqueueRecord],
    dequeues: list[DequeueRecord],
    finishes: list[FinishRecord],
    jobs: list[JobRecord],
) -> None:
    """Estimate how much more often worker threads have to re-acquire the
    shared pool lock (and therefore fight over the GIL) post-batching vs.
    the pre-batching design, where each job was exactly one queue item
    serviced by exactly one C call on one thread.

    Pre-batching, each job contributes exactly one "dequeue" + one "finish"
    event (2 lock-guarded dispatch events). Post-batching, each job
    contributes 2 * n_batches such events. The ratio is therefore just the
    average number of batches per job.
    """
    if not jobs:
        print("No jobs found.")
        return

    t0 = min(r.t for r in enqueues) if enqueues else min(d.t for d in dequeues)
    t1 = max(r.t for r in finishes) if finishes else max(d.t for d in dequeues)
    span = t1 - t0

    n_jobs = len(jobs)
    n_batches_total = sum(j.n_batches for j in jobs)
    post_events = len(dequeues) + len(finishes)
    pre_equivalent_events = 2 * n_jobs

    print("\n=== Dispatch-event (lock/GIL re-acquisition) frequency ===")
    print(f"  jobs: {n_jobs}, wall-clock span: {span:.1f}s")
    print(
        f"  post-batching dispatch events (D+F): {post_events}  "
        f"({post_events / span:.1f}/s across {32} threads, "
        f"~{post_events / span / 32:.2f}/s per thread)"
    )
    print(
        f"  pre-batching equivalent (1 batch/job): {pre_equivalent_events}  "
        f"({pre_equivalent_events / span:.1f}/s across 32 threads, "
        f"~{pre_equivalent_events / span / 32:.2f}/s per thread)"
    )
    ratio = n_batches_total / n_jobs
    print(
        f"\n  => batching multiplies the rate at which threads re-acquire "
        f"the shared pool lock/condition (and therefore contend for the "
        f"GIL) by ~{ratio:.1f}x, for the *same* underlying I/O work.\n"
        "     Each re-acquisition is short, but with 32 threads all doing "
        "it far more often, aggregate GIL handoff/scheduling overhead is a "
        "plausible contributor to the regression on top of the pure "
        "queueing-delay effect (see --overlap / summary output)."
    )
    print(
        "  NOTE: this log was itself captured with TimingRecorder enabled, "
        "which adds one more lock + file write per dispatch event on top of "
        "the pool's own lock -- treat absolute rates as an upper bound, "
        "but the ~batches-per-job multiplier is unaffected by that."
    )


def print_worst_span_timeline(
    dequeues: list[DequeueRecord],
    jobs: list[JobRecord],
    top: int,
    kind: str | None,
) -> None:
    """For the `top` jobs with the largest span, print when (normalized to
    0-100, where 0 = first batch picked up and 100 = job completion) each
    thread grabbed one of that job's batches."""
    candidates = jobs if kind is None else [j for j in jobs if j.kind == kind]
    worst = sorted(candidates, key=lambda j: j.span, reverse=True)[:top]
    if not worst:
        print("No jobs found.")
        return

    deq_by_job: dict[str, list[DequeueRecord]] = {}
    for d in dequeues:
        deq_by_job.setdefault(d.job_id, []).append(d)

    print(f"\n=== Batch pickup times, normalized to job lifetime (0-100), "
          f"for the {len(worst)} worst-span jobs ===")
    for j in worst:
        rows = sorted(deq_by_job.get(j.job_id, []), key=lambda d: d.t)
        if not rows:
            continue
        t_first = rows[0].t
        print(
            f"\n  job_id={j.job_id}  kind={j.kind}  span={j.span:.4f}s  "
            f"n_batches={j.n_batches}  queue_wait_first={j.queue_wait_first:.4f}s"
        )
        for d in rows:
            norm = 100.0 * (d.t - t_first) / j.span if j.span > 0 else 0.0
            print(
                f"    t={norm:6.2f}/100  batch_no={d.batch_no:<3} "
                f"thread={d.thread}"
            )


def _global_batch_intervals(
    dequeues: list[DequeueRecord], finishes: list[FinishRecord]
) -> list[tuple[float, float, int]]:
    """All batch (start, end, batch_size) intervals, across every job/thread,
    matched on (job_id, batch_no) (unique within a job, and each batch_no is
    only ever dispatched once)."""
    starts = {(d.job_id, d.batch_no): (d.t, d.batch_size) for d in dequeues}
    out = []
    for f in finishes:
        s = starts.get((f.job_id, f.batch_no))
        if s is None:
            continue
        t_start, batch_size = s
        out.append((t_start, f.t, batch_size))
    return out


def print_concurrency_vs_duration_report(
    dequeues: list[DequeueRecord], finishes: list[FinishRecord]
) -> None:
    """Tests the GIL/resource-contention hypothesis directly: if threads are
    fighting over the GIL (or disk bandwidth) as more batches run at once,
    per-block execution time should rise with system-wide concurrency. If
    per-block time is flat across concurrency levels, the regression is
    better explained by queueing/admission effects than by contention during
    execution itself.

    For each batch, computes how many *other* batches (any job, load or
    store) were already in flight at the instant it started, then buckets
    batches by that concurrency level and reports mean duration-per-block.
    """
    import heapq

    intervals = _global_batch_intervals(dequeues, finishes)
    if not intervals:
        print("No matched batch intervals found.")
        return

    intervals.sort(key=lambda iv: iv[0])
    heap: list[float] = []  # end times of currently-active intervals
    rows = []  # (concurrency_at_start, duration, batch_size)
    for t_start, t_end, batch_size in intervals:
        while heap and heap[0] <= t_start:
            heapq.heappop(heap)
        concurrency = len(heap)  # excludes this interval itself
        rows.append((concurrency, t_end - t_start, batch_size))
        heapq.heappush(heap, t_end)

    buckets = [(0, 4), (5, 8), (9, 12), (13, 16), (17, 20), (21, 24), (25, 28), (29, 64)]
    print("\n=== Per-block duration vs. system-wide concurrency at batch start ===")
    print(
        f"  {'concurrency':<14}{'n_batches':>10}{'mean_dur(ms)':>14}"
        f"{'mean_batch_sz':>14}{'mean_dur/block(us)':>20}"
    )
    for lo, hi in buckets:
        sel = [(dur, bs) for c, dur, bs in rows if lo <= c <= hi]
        if not sel:
            continue
        mean_dur = statistics.fmean(d for d, _ in sel)
        mean_bs = statistics.fmean(bs for _, bs in sel)
        mean_dur_per_block = statistics.fmean(
            (d / bs) for d, bs in sel if bs > 0
        )
        label = f"{lo}-{hi}"
        print(
            f"  {label:<14}{len(sel):>10}{mean_dur * 1000:>14.3f}"
            f"{mean_bs:>14.1f}{mean_dur_per_block * 1e6:>20.2f}"
        )

    concs = [c for c, _, _ in rows]
    per_block = [d / bs for _, d, bs in rows if bs > 0]
    concs_for_corr = [c for c, _, bs in rows if bs > 0]
    if len(set(concs_for_corr)) > 1 and len(set(per_block)) > 1:
        r = statistics.correlation(concs_for_corr, per_block)
        print(f"\nPearson correlation(concurrency, duration_per_block): {r:.3f}")
        print(
            "  (near 0 => execution time is insensitive to concurrency, i.e. "
            "no meaningful GIL/resource contention during execution;\n"
            "   strongly positive => execution genuinely slows down as more "
            "batches run at once, consistent with GIL/IO contention)"
        )
    print(f"\noverall concurrency range observed: {min(concs)}-{max(concs)}")


def print_worst_span_batch_summary(
    dequeues: list[DequeueRecord],
    jobs: list[JobRecord],
    top: int,
    kind: str | None,
) -> None:
    """For the `top` jobs with the largest span, print each job's average
    batch size and its distinct-thread count, plus aggregate stats across
    that set."""
    candidates = jobs if kind is None else [j for j in jobs if j.kind == kind]
    worst = sorted(candidates, key=lambda j: j.span, reverse=True)[:top]
    if not worst:
        print("No jobs found.")
        return

    deq_by_job: dict[str, list[DequeueRecord]] = {}
    for d in dequeues:
        deq_by_job.setdefault(d.job_id, []).append(d)

    print(
        f"\n=== Avg batch size + distinct threads, for the {len(worst)} "
        "worst-span jobs ==="
    )
    print(
        f"  {'job_id':<12}{'kind':<7}{'span':>10}{'n_batches':>10}"
        f"{'avg_batch_size':>16}{'distinct_threads':>18}"
    )
    avg_batch_sizes, thread_counts = [], []
    for j in worst:
        rows = deq_by_job.get(j.job_id, [])
        if not rows:
            continue
        avg_bs = statistics.fmean(d.batch_size for d in rows)
        n_threads = len({d.thread for d in rows})
        avg_batch_sizes.append(avg_bs)
        thread_counts.append(n_threads)
        print(
            f"  {j.job_id:<12}{j.kind:<7}{j.span:>10.4f}{j.n_batches:>10}"
            f"{avg_bs:>16.2f}{n_threads:>18}"
        )

    print(
        f"\nAcross these {len(avg_batch_sizes)} jobs: "
        f"mean avg_batch_size={statistics.fmean(avg_batch_sizes):.2f}, "
        f"mean distinct_threads={statistics.fmean(thread_counts):.2f}"
    )


def print_overlap_report(
    dequeues: list[DequeueRecord],
    finishes: list[FinishRecord],
    jobs: list[JobRecord],
    n_threads: int,
) -> None:
    """For load jobs whose batches touched exactly `n_threads` distinct
    threads, report span plus how much those threads' batch-processing
    intervals actually overlapped in wall-clock time.

    avg_concurrency = sum(batch durations) / union(batch intervals):
        1.0 means the batches never overlapped at all (fully serial);
        n_threads means they were all in flight simultaneously the whole
        time (perfect overlap).
    peak_concurrency = max number of that job's batches in flight at any
        single instant (<= n_threads).
    """
    load_job_ids = {j.job_id for j in jobs if j.kind == "load"}
    thread_count = _distinct_thread_count_per_job(
        [d for d in dequeues if d.job_id in load_job_ids]
    )
    target_ids = {jid for jid, n in thread_count.items() if n == n_threads}
    job_by_id = {j.job_id: j for j in jobs if j.job_id in target_ids}
    if not job_by_id:
        print(f"No load jobs found with exactly {n_threads} distinct threads.")
        return

    intervals_by_job = _batch_intervals_per_job(
        [d for d in dequeues if d.job_id in target_ids],
        [f for f in finishes if f.job_id in target_ids],
    )

    spans, avg_conc, peak_conc = [], [], []
    for jid in target_ids:
        j = job_by_id[jid]
        ivs = intervals_by_job.get(jid, [])
        if not ivs:
            continue
        union = _union_duration(ivs)
        sum_dur = sum(e - s for s, e in ivs)
        spans.append(j.span)
        avg_conc.append(sum_dur / union if union > 0 else 0.0)
        peak_conc.append(_peak_concurrency(ivs))

    print(f"\n=== Jobs with exactly {n_threads} distinct threads (n={len(spans)}) ===")
    _print_table(
        f"Span and thread overlap ({n_threads}-thread bucket)",
        {
            "span": _summarize(spans),
            "avg_concurrency": _summarize(avg_conc),
            "peak_concurrency": _summarize([float(p) for p in peak_conc]),
        },
    )
    print(
        f"\navg_concurrency: mean {statistics.fmean(avg_conc):.2f} out of a "
        f"possible {n_threads} (1.0 = fully serial, {n_threads}.0 = fully "
        "overlapped).\n"
        f"peak_concurrency: mean {statistics.fmean(peak_conc):.2f}, "
        f"i.e. on average only ~{statistics.fmean(peak_conc):.1f} of the "
        f"{n_threads} threads were ever simultaneously active on the same "
        "job, even though all 16 eventually touched it."
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
        {
            "queue_depth_before": _summarize(
                [float(e.queue_depth_before) for e in enqueues]
            )
        },
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
    parser.add_argument(
        "log_path", help="Path to the timing log written by TimingRecorder"
    )
    parser.add_argument("--kind", choices=["load", "store"], default=None)
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of worst jobs (by pickup_spread) to list",
    )
    parser.add_argument(
        "--job-id", default=None, help="Print the full E/D/F/J timeline for one job id"
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="fs_tier_load_job_metrics.png",
        default=None,
        metavar="PNG_PATH",
        help=(
            "Save 3 stacked bar charts (queue_wait_first, span, max batch "
            "size), one bar per load job in chronological order, to "
            "PNG_PATH (default: fs_tier_load_job_metrics.png)."
        ),
    )
    parser.add_argument(
        "--thread-hist",
        nargs="?",
        const="fs_tier_thread_count_hist.png",
        default=None,
        metavar="PNG_PATH",
        help=(
            "Save a histogram of the number of load jobs by how many "
            "distinct threads processed their batches, to PNG_PATH "
            "(default: fs_tier_thread_count_hist.png)."
        ),
    )
    parser.add_argument(
        "--overlap",
        nargs="?",
        type=int,
        const=16,
        default=None,
        metavar="N_THREADS",
        help=(
            "Report span + thread-overlap stats (avg/peak concurrency) for "
            "load jobs whose batches touched exactly N_THREADS distinct "
            "threads (default: 16)."
        ),
    )
    parser.add_argument(
        "--gil-report",
        action="store_true",
        help="Report dispatch-event (lock/GIL re-acquisition) frequency, "
        "batched vs. the pre-batching one-item-per-job design.",
    )
    parser.add_argument(
        "--worst-span-timeline",
        nargs="?",
        type=int,
        const=10,
        default=None,
        metavar="N",
        help="Print normalized (0-100) batch pickup times for the N "
        "worst-span jobs (default: 10). Combine with --kind.",
    )
    parser.add_argument(
        "--worst-span-batch-summary",
        nargs="?",
        type=int,
        const=10,
        default=None,
        metavar="N",
        help="Print avg batch size + distinct-thread count for the N "
        "worst-span jobs (default: 10). Combine with --kind.",
    )
    parser.add_argument(
        "--concurrency-vs-duration",
        action="store_true",
        help="Report mean per-block execution duration bucketed by "
        "system-wide concurrency at batch start, to test whether execution "
        "actually slows down under contention (GIL/IO) vs. queueing alone.",
    )
    parser.add_argument(
        "--bucket-timeline",
        nargs="?",
        const="fs_tier_thread_count_chronological.png",
        default=None,
        metavar="PNG_PATH",
        help="Save a plot of how the distinct-thread-count histogram "
        "buckets are distributed chronologically over the run, to PNG_PATH "
        "(default: fs_tier_thread_count_chronological.png).",
    )
    args = parser.parse_args()

    enqueues, dequeues, finishes, jobs = parse(args.log_path)

    if args.plot is not None:
        plot_load_job_bars(enqueues, dequeues, jobs, args.plot)
        return

    if args.thread_hist is not None:
        plot_thread_count_histogram(dequeues, jobs, args.thread_hist)
        return

    if args.overlap is not None:
        print_overlap_report(dequeues, finishes, jobs, args.overlap)
        return

    if args.gil_report:
        print_gil_contention_report(enqueues, dequeues, finishes, jobs)
        return

    if args.worst_span_timeline is not None:
        print_worst_span_timeline(dequeues, jobs, args.worst_span_timeline, args.kind)
        return

    if args.concurrency_vs_duration:
        print_concurrency_vs_duration_report(dequeues, finishes)
        return

    if args.worst_span_batch_summary is not None:
        print_worst_span_batch_summary(
            dequeues, jobs, args.worst_span_batch_summary, args.kind
        )
        return

    if args.bucket_timeline is not None:
        plot_thread_count_chronological(enqueues, dequeues, jobs, args.bucket_timeline)
        return

    if args.job_id is not None:
        print_job_timeline(args.job_id, enqueues, dequeues, finishes, jobs)
        return

    if args.kind is not None:
        enqueues = [e for e in enqueues if e.kind == args.kind]
        jobs = [j for j in jobs if j.kind == args.kind]

    summarize_all(enqueues, dequeues, finishes, jobs, args.top)


if __name__ == "__main__":
    main()
