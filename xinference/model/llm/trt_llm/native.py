# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Release live context transmissions in the pinned TRT-LLM TP1 executor."""

import threading
import time
from concurrent.futures import Future, TimeoutError


class NativeSourceReleaser:
    def __init__(self, executor):
        self.executor = executor
        self._lock = threading.Lock()
        self._pending: dict[int, Future] = {}
        self._cancelled: dict[int, object] = {}
        original = executor._terminate_disagg_ctx_finished_requests
        check_status = executor._check_disagg_ctx_cache_transfer_status
        self._restore = (original, check_status)

        def check(atLeastNum=0):
            # TRT 1.2.1 retains a cancelled sender future until a peer contacts
            # it. Blocking on that future prevents unrelated queued requests
            # from progressing. Poll instead when capacity is temporarily full.
            if (self._pending or self._cancelled) and atLeastNum:
                time.sleep(0.001)
                atLeastNum = 0
            return check_status(atLeastNum)

        def poll():
            # Both cancellation and KV unpinning run on TRT's execution thread.
            # Reuse its timeout cleanup even when the configured timeout is None.
            with self._lock:
                sources = {}
                for ticket in self._pending:
                    entry = executor.ctx_in_transmission_requests.get(ticket)
                    if entry is not None:
                        entry[0].py_kv_transfer_timed_out = True
                        if not entry[0].is_disagg_context_complete_state:
                            sources[ticket] = entry[0]
                try:
                    original()
                except Exception as error:
                    for future in self._pending.values():
                        future.set_exception(error)
                    self._pending.clear()
                    raise
                for ticket, future in list(self._pending.items()):
                    if ticket not in executor.ctx_in_transmission_requests:
                        self._pending.pop(ticket)
                        if ticket in sources:
                            # The native future holds a raw LlmRequest pointer.
                            # Keep that metadata alive until engine shutdown;
                            # cancellation has already released its GPU blocks.
                            self._cancelled[ticket] = sources[ticket]
                        future.set_result(None)

        executor._terminate_disagg_ctx_finished_requests = poll
        executor._check_disagg_ctx_cache_transfer_status = check

    def engine_stopped(self) -> None:
        # Only after native shutdown may its retained raw pointers be dropped.
        executor = self.executor
        (
            executor._terminate_disagg_ctx_finished_requests,
            executor._check_disagg_ctx_cache_transfer_status,
        ) = self._restore
        self._cancelled.clear()
        self.executor = self._restore = None

    def find_source(self, client_id: int):
        # The final response may have retired BaseWorker's mapping before the
        # Python result has consumed its context metadata.
        with self._lock:
            for ticket, (request, _, _) in list(
                self.executor.ctx_in_transmission_requests.items()
            ):
                if request.py_client_id == client_id:
                    return ticket
        return None

    def release(self, ticket: int) -> None:
        with self._lock:
            if ticket not in self.executor.ctx_in_transmission_requests:
                return  # The native sender already completed and unpinned it.
            future = self._pending.setdefault(ticket, Future())
        # No wall-clock cancellation: keep slots owned until native cancellation
        # succeeds and TRT has removed the transmission and released its blocks.
        while not future.done():
            # With no active generation TRT sleeps in its request queue, even
            # with live source transmissions. Queue the runtime ID directly to
            # wake that execution thread; BaseWorker's client mapping is gone.
            self.executor.cancel_request(ticket)
            try:
                future.result(timeout=0.1)
            except TimeoutError:
                pass
        future.result()
