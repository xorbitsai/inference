# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Release live context transmissions on TRT-LLM's TP1 execution thread."""

import threading
import time
from concurrent.futures import Future, TimeoutError


class NativeSourceReleaser:
    def __init__(self, executor):
        self.executor = executor
        self._lock = threading.Lock()
        self._pending: dict[int, Future] = {}
        self._cancelled: dict[int, object] = {}
        if not hasattr(executor, "ctx_in_transmission_requests"):
            self._initialize_async_transfers(executor)
            return
        self._transmissions = lambda: {
            ticket: entry[0]
            for ticket, entry in executor.ctx_in_transmission_requests.items()
        }
        self._status_owner = None
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

    def _initialize_async_transfers(self, executor) -> None:
        manager = executor.async_transfer_manager
        self._transmissions = manager.requests_in_transfer
        if hasattr(executor, "_check_disagg_ctx_cache_transfer_status"):
            owner, name = executor, "_check_disagg_ctx_cache_transfer_status"
        else:
            # Later 1.3 releases moved transfer orchestration out of PyExecutor.
            owner, name = executor.disagg, "reap_context_sends"
        original = getattr(owner, name)
        self._status_owner = owner, name
        self._restore = original

        def poll(at_least=0):
            if (self._pending or self._cancelled) and at_least:
                # Cancelled native sender pointers must remain alive until
                # shutdown. Match the legacy nonblocking poll's backoff.
                time.sleep(0.001)
                at_least = 0
            with self._lock:
                sources = {}
                for ticket in self._pending:
                    request = self._transmissions().get(ticket)
                    if request is not None:
                        # The native status path retries cancellation, then
                        # removes the transfer only after native KV cleanup.
                        request.py_kv_transfer_timed_out = True
                        if not request.is_disagg_context_complete_state:
                            sources[ticket] = request
                try:
                    original(0 if self._pending or self._cancelled else at_least)
                except Exception as error:
                    for future in self._pending.values():
                        future.set_exception(error)
                    self._pending.clear()
                    raise
                for ticket, future in list(self._pending.items()):
                    if ticket not in self._transmissions():
                        self._pending.pop(ticket)
                        if ticket in sources:
                            self._cancelled[ticket] = sources[ticket]
                        future.set_result(None)

        setattr(owner, name, poll)

    def engine_stopped(self) -> None:
        # Only after native shutdown may its retained raw pointers be dropped.
        executor = self.executor
        if self._status_owner is None:
            (
                executor._terminate_disagg_ctx_finished_requests,
                executor._check_disagg_ctx_cache_transfer_status,
            ) = self._restore
        else:
            owner, name = self._status_owner
            setattr(owner, name, self._restore)
        self._cancelled.clear()
        self._transmissions = None
        self._status_owner = None
        self.executor = self._restore = None

    def find_source(self, client_id: int):
        # The final response may have retired BaseWorker's mapping before the
        # Python result has consumed its context metadata.
        with self._lock:
            for ticket, request in list(self._transmissions().items()):
                if request.py_client_id == client_id:
                    return ticket
        return None

    def release(self, ticket: int) -> None:
        with self._lock:
            if ticket not in self._transmissions():
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
