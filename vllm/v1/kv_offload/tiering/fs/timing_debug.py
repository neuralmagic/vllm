# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Diagnostic-only timing instrumentation for ``DualQueueThreadPool``.

Writes raw, timestamped CSV-ish records to a file for later offline
analysis, so that batching behaviour (in particular, whether a job's
batches are actually picked up by threads at (roughly) the same time, or
trickle in staggered as threads free up) can be reconstructed after the
fact without perturbing the normal logger output.

Enabled by setting ``VLLM_KV_OFFLOAD_FS_TIMING_LOG`` to a file path; unset
(the default) makes every call a no-op.

Record types (first field), one per line:
    E,<t>,<job_id>,<kind>,<n_tasks>,<n_batches>,<queue_depth_before>
        A job was enqueued and split into ``n_batches`` batches.
        ``queue_depth_before`` is the length of the target queue just
        before this job's batches were appended (i.e. backlog ahead of
        it), so unexpectedly high pickup latency can be attributed to
        contention rather than "batching not working".
    D,<t>,<job_id>,<thread>,<batch_no>,<batch_size>,<queue_wait>,<queue_depth_after>
        A worker thread popped one batch off the queue.
        ``queue_wait = t - t_enqueue`` for that job.
    F,<t>,<job_id>,<thread>,<batch_no>,<duration>,<success>
        A batch finished executing (success is "1"/"0").
    J,<t>,<job_id>,<kind>,<n_batches>,<queue_wait_first>,<pickup_spread>,
      <span>,<sum_batch_time>,<parallel_efficiency>
        The job (all batches) finished. ``pickup_spread`` is the time
        between the first and last batch of the job being picked up by a
        thread -- near 0 means the batches started together (true
        parallel dispatch); a value comparable to a batch's own duration
        means the batches were staggered onto threads one at a time.
        ``parallel_efficiency = sum_batch_time / (span * n_batches)``:
        close to 1.0 means the batches genuinely overlapped across
        ``n_batches`` threads; close to ``1 / n_batches`` means they ran
        essentially back-to-back, i.e. no better than a single thread.
"""

import threading


class TimingRecorder:
    """Thread-safe, opt-in writer of diagnostic timing records.

    A no-op (branch-cheap) when ``path`` is ``None`` or empty, so it is
    safe to construct unconditionally.
    """

    def __init__(self, path: str | None) -> None:
        self._enabled = bool(path)
        self._lock = threading.Lock()
        self._file = open(path, "a", buffering=1) if self._enabled else None  # noqa: SIM115

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(self, *fields: object) -> None:
        if not self._enabled:
            return
        line = ",".join(str(f) for f in fields)
        with self._lock:
            assert self._file is not None
            self._file.write(line + "\n")

    def close(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            assert self._file is not None
            self._file.flush()
            self._file.close()
            self._enabled = False
