# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AsyncLookupManager: per-tier async lookup manager for secondary tier
existence checks.

Each secondary tier that wants non-blocking lookups composes its own
AsyncLookupManager instance internally.  The manager maintains lookup
state and uses a background worker to execute batch_lookup() calls.

Locking design
--------------
There is no explicit lock.  Thread safety is achieved by ownership:

* _lookup_state and _lookup_batch are owned exclusively by the scheduler
  thread.  lookup(), flush(), and cleanup() read and write them directly.

* _lookup_queue is written by the scheduler (flush → put, one item per step)
  and read by the background worker (get).

* _pending_results is written by the background worker (put) and read by
  the scheduler (get_nowait inside drain_results).  queue.SimpleQueue is
  thread-safe by design.

lookup() accumulates new keys in _lookup_batch without touching the queue.
flush() is called once per step from the tier's on_schedule_end(), posting
the entire batch as a single queue item so the background worker sees one
batch per step.
drain_results() is called before any lookup() calls in the same step, so
lookup() is a pure OrderedDict operation.
"""

import queue
import signal
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue as MPQueue
from types import FrameType
from typing import Any, Literal

from vllm.logger import init_logger
from vllm.utils.system_utils import get_mp_context
from vllm.v1.kv_offload.base import OffloadKey, ReqContext

logger = init_logger(__name__)


@dataclass(slots=True)
class LookupState:
    result: bool | None = None  # True (found), False (not found), None
    request_ids: set[str] = field(default_factory=set)  # requests asking for the lookup


BatchLookupFn = Callable[[list[OffloadKey], ReqContext], Iterable[bool]]
ProcessBatchLookupFn = Callable[[list[OffloadKey], ReqContext, Any], Iterable[bool]]
LookupBatch = list[tuple[OffloadKey, ReqContext]]
LookupResultBatch = list[tuple[OffloadKey, bool]]
LookupQueue = queue.SimpleQueue[LookupBatch | None] | MPQueue[LookupBatch | None]
PendingResultsQueue = queue.SimpleQueue[LookupResultBatch] | MPQueue[LookupResultBatch]


def _run_lookup_loop(
    tier_type: str,
    lookup_queue: LookupQueue,
    pending_results: PendingResultsQueue,
    batch_lookup_fn: BatchLookupFn,
) -> None:
    while True:
        pending = lookup_queue.get()
        if pending is None:
            break

        # Group by req_id.
        batches: dict[str, tuple[ReqContext, list[OffloadKey]]] = {}
        for key, req_context in pending:
            req_id = req_context.req_id
            if req_id not in batches:
                batches[req_id] = (req_context, [])
            batches[req_id][1].append(key)

        if not batches:
            continue

        results: LookupResultBatch = []
        for req_context, keys in batches.values():
            try:
                hits = batch_lookup_fn(keys, req_context)
            except Exception as exc:
                logger.warning(
                    "batch_lookup failed on tier %s for %d keys: %s",
                    tier_type,
                    len(keys),
                    exc,
                )
                hits = (False for _ in keys)

            for key, hit in zip(keys, hits):
                results.append((key, hit))

        # Post the entire batch as one item — no lock needed.
        if results:
            pending_results.put(results)


class _LookupWorker(ABC):
    """Internal worker interface used by AsyncLookupManager."""

    @abstractmethod
    def put_batch(self, batch: list[tuple[OffloadKey, ReqContext]]) -> None: ...

    @abstractmethod
    def get_result_nowait(self) -> list[tuple[OffloadKey, bool]]: ...

    @abstractmethod
    def shutdown(self) -> None: ...


class _ThreadLookupWorker(_LookupWorker):
    """Lookup worker backed by a daemon thread."""

    def __init__(self, tier_type: str, batch_lookup_fn: BatchLookupFn) -> None:
        self._lookup_queue: queue.SimpleQueue[LookupBatch | None] = queue.SimpleQueue()
        self._pending_results: queue.SimpleQueue[LookupResultBatch] = (
            queue.SimpleQueue()
        )
        self._thread = threading.Thread(
            target=_run_lookup_loop,
            kwargs={
                "tier_type": tier_type,
                "lookup_queue": self._lookup_queue,
                "pending_results": self._pending_results,
                "batch_lookup_fn": batch_lookup_fn,
            },
            name=f"vllm_offloading_lookup_{tier_type}",
            daemon=True,
        )
        self._thread.start()

    def put_batch(self, batch: LookupBatch) -> None:
        self._lookup_queue.put(batch)

    def get_result_nowait(self) -> LookupResultBatch:
        return self._pending_results.get_nowait()

    def shutdown(self) -> None:
        self._lookup_queue.put(None)
        self._thread.join()


class _ProcessLookupWorker(_LookupWorker):
    """Lookup worker backed by a subprocess."""

    def __init__(
        self,
        tier_type: str,
        process_batch_lookup_fn: ProcessBatchLookupFn,
        process_context: Any,
    ) -> None:
        context = get_mp_context()
        self._lookup_queue = context.Queue()
        self._pending_results = context.Queue()
        ready_reader, ready_writer = context.Pipe(duplex=False)
        death_reader, death_writer = context.Pipe(duplex=False)
        self._death_writer: Connection | None = death_writer
        self._proc = context.Process(
            target=_ProcessLookupWorker._proc_main,
            kwargs={
                "tier_type": tier_type,
                "lookup_queue": self._lookup_queue,
                "pending_results": self._pending_results,
                "process_batch_lookup_fn": process_batch_lookup_fn,
                "process_context": process_context,
                "ready_writer": ready_writer,
                "death_pipe": death_reader,
            },
            name=f"vllm_offloading_lookup_{tier_type}",
            daemon=True,
        )
        self._proc.start()
        ready_writer.close()
        death_reader.close()
        try:
            status = ready_reader.recv()
        finally:
            ready_reader.close()
        if status.get("status") != "READY":
            raise RuntimeError(
                f"Lookup subprocess failed to initialize for tier {tier_type}"
            )

    @staticmethod
    def _proc_main(
        tier_type: str,
        lookup_queue: MPQueue[LookupBatch | None],
        pending_results: MPQueue[LookupResultBatch],
        process_batch_lookup_fn: ProcessBatchLookupFn,
        process_context: Any,
        ready_writer: Connection,
        death_pipe: Connection | None = None,
    ) -> None:
        def _signal_handler(signum: int, _frame: FrameType | None) -> None:
            raise SystemExit(f"lookup subprocess got signal {signum}")

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        def _death_pipe_monitor(death_conn: Connection) -> None:
            try:
                death_conn.recv()
            except EOFError:
                with suppress(Exception):
                    lookup_queue.put(None)
            finally:
                death_conn.close()

        if death_pipe is not None:
            threading.Thread(
                target=_death_pipe_monitor,
                args=(death_pipe,),
                daemon=True,
                name=f"vllm_offloading_lookup_death_{tier_type}",
            ).start()

        ready_writer.send({"status": "READY"})
        ready_writer.close()

        def _batch_lookup_with_context(
            keys: list[OffloadKey], req_context: ReqContext
        ) -> Iterable[bool]:
            return process_batch_lookup_fn(keys, req_context, process_context)

        _run_lookup_loop(
            tier_type, lookup_queue, pending_results, _batch_lookup_with_context
        )

    def put_batch(self, batch: LookupBatch) -> None:
        self._lookup_queue.put(batch)

    def get_result_nowait(self) -> LookupResultBatch:
        return self._pending_results.get_nowait()

    def shutdown(self) -> None:
        if self._death_writer is not None:
            self._death_writer.close()
            self._death_writer = None
        self._lookup_queue.put(None)
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2)
        if self._proc.is_alive():
            self._proc.kill()


class AsyncLookupManager(ABC):
    """
    Per-tier async lookup manager for secondary tier existence checks.

    Each secondary tier that wants non-blocking lookups composes its own
    AsyncLookupManager instance internally. The manager maintains lookup
    state (cache, queue) and uses a background worker (thread or subprocess)
    to execute the actual
    batch_lookup() calls.

    Subclasses implement only batch_lookup() — all queue management,
    state tracking, and result delivery is provided by this base class.

    The owning tier delegates its lookup(), on_schedule_end(), and
    on_request_finished() to this manager:
      - lookup() → drain_results() + lookup state check
      - on_schedule_end() → flush()
      - on_request_finished() → cleanup()
    """

    def __init__(
        self,
        tier_type: str,
        worker_mode: Literal["thread", "process"] = "thread",
        process_context: Any = None,
    ) -> None:
        self._tier_type = tier_type

        # key → LookupState; scheduler-owned, no lock needed.
        self._lookup_state: dict[OffloadKey, LookupState] = {}
        # req_id → keys looked up by that request (reverse index for cleanup).
        self._req_keys: dict[str, set[OffloadKey]] = {}

        # Accumulates (key, req_context) pairs during lookup() calls.
        # Flushed as one queue item per step by flush().
        self._lookup_batch: list[tuple[OffloadKey, ReqContext]] = []

        if worker_mode == "thread":
            self._worker: _LookupWorker = _ThreadLookupWorker(
                tier_type, self.batch_lookup
            )
        elif worker_mode == "process":
            self._worker = _ProcessLookupWorker(
                tier_type,
                self.__class__.process_batch_lookup,
                process_context,
            )
        else:
            raise ValueError(
                f"Invalid lookup worker_mode={worker_mode!r}, "
                "expected 'thread' or 'process'"
            )

        self._need_to_drain: bool = False

    @abstractmethod
    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        """
        Check whether a batch of blocks exist in this tier.

        Called from the lookup worker (thread/process) — must be synchronous and
        must not touch the primary tier or scheduler state.

        Returns a list parallel to keys: True if present, False if not.
        """
        ...

    @staticmethod
    def process_batch_lookup(
        keys: list[OffloadKey],
        req_context: ReqContext,
        process_context: Any,
    ) -> Iterable[bool]:
        """Process-safe lookup implementation for process worker mode."""
        raise NotImplementedError(
            "process_batch_lookup() is not implemented for this tier."
        )

    # ------------------------------------------------------------------
    # Scheduler-thread API
    # ------------------------------------------------------------------

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Non-blocking lookup called from the scheduler thread.

        Returns:
            True  — block is present in this tier.
            False — block is not present in this tier.
            None  — result not yet available; retry next step.
        """
        if self._need_to_drain:
            self.drain_results()
            self._need_to_drain = False
        req_id = req_context.req_id
        state = self._lookup_state.get(key)
        if state is None:
            state = LookupState()
            self._lookup_state[key] = state
            self._lookup_batch.append((key, req_context))
        state.request_ids.add(req_id)
        self._req_keys.setdefault(req_id, set()).add(key)
        return state.result

    def flush(self) -> None:
        """Post this step's accumulated keys to the lookup worker.

        Called once per step from on_schedule_end() after all lookup() calls
        are done. The worker receives the full batch and processes it during
        the model-execution window, maximising time available before the next
        step's drain_results().  Safe to call with an empty batch (no-op).
        """
        self._need_to_drain = True
        if self._lookup_batch:
            self._worker.put_batch(self._lookup_batch)
            self._lookup_batch = []

    def drain_results(self) -> None:
        """Apply pending worker results to _lookup_state.

        Called from lookup() before checking state.
        """
        while True:
            try:
                batch = self._worker.get_result_nowait()
            except queue.Empty:
                break
            for key, result in batch:
                state = self._lookup_state.get(key)
                if state is not None:
                    state.result = result

    def cleanup(self, req_id: str) -> None:
        """Remove entries no longer needed by any active request.

        Called from the tier's on_request_finished(). Uses the reverse
        index to visit only keys associated with this request.
        """
        for key in self._req_keys.pop(req_id, ()):
            state = self._lookup_state[key]
            state.request_ids.discard(req_id)
            if not state.request_ids:
                del self._lookup_state[key]

    def shutdown(self) -> None:
        """Stop the lookup worker (thread or subprocess)."""
        self._worker.shutdown()
