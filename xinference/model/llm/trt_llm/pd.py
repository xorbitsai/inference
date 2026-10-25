# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""TRT P/D: native NIXL or request-owned Xavier GPU pages."""

import asyncio
import base64
import json
import logging
import threading
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, replace
from pathlib import Path

from ..xavier.pd_contract import (
    build_pd_contract,
    prompt_digest,
    validate_handoff_scope,
)
from .xavier import EXECUTION_OPTIONS

logger = logging.getLogger(__name__)

# The connector receives a C++ request, which does not retain Python client-ID
# attributes. Its cache_salt_id survives that boundary. Register an internal,
# unique salt before enqueueing, also isolating native prefix reuse per request.
_lock = threading.Lock()
_requests: dict = {}


@dataclass
class XavierRequest:
    role: str
    tokens: tuple
    room: int
    imported_tokens: int = 0
    engine_request_id: int = 0
    registered: bool = False
    load_future: Future | None = None
    cancelled: bool = False
    heterogeneous: bool = False

    @property
    def num_pages(self):
        tokens = (
            len(self.tokens)
            if self.heterogeneous and self.role == "prefill"
            else max(len(self.tokens) - 1, 0)
        )
        return (tokens + 63) // 64


def submit_xavier(engine, tokens, params, state):
    from tensorrt_llm.inputs import get_cache_salt_id

    salt = uuid.uuid4().hex
    salt_id = get_cache_salt_id(salt)
    with _lock:
        _requests[salt_id] = state
    try:
        request = engine.generate_async(tokens, params, streaming=True, cache_salt=salt)
        request._xinference_xavier_salt_id = salt_id
    except BaseException:
        with _lock:
            _requests.pop(salt_id, None)
        raise
    return request


def request_state(request):
    with _lock:
        return _requests.get(getattr(request, "cache_salt_id", None))


def forget_request(request):
    with _lock:
        _requests.pop(request._xinference_xavier_salt_id, None)


def configure_native(model_path, config, replica):
    if replica.get("role") not in ("prefill", "decode"):
        raise ValueError("TRT-LLM native P/D requires a prefill or decode role")
    if config.get("kv_connector_config") is not None:
        raise ValueError("TRT-LLM native P/D cannot use a KV connector")
    if config.get("dtype", "auto") not in ("auto", "bfloat16"):
        raise ValueError("TRT-LLM P/D requires BF16 weights and KV")
    if set(config) - (EXECUTION_OPTIONS | {"cache_transceiver_config"}) or config.get(
        "trust_remote_code"
    ):
        raise ValueError(
            "TRT-LLM native P/D does not allow model or execution overrides"
        )
    checkpoint = json.loads((Path(model_path) / "config.json").read_text())
    if checkpoint.get("dtype", checkpoint.get("torch_dtype")) != "bfloat16":
        raise ValueError("TRT-LLM P/D requires a BF16 checkpoint")
    kv = dict(config.get("kv_cache_config") or {})
    if (
        set(kv)
        - {
            "max_tokens",
            "free_gpu_memory_fraction",
            "tokens_per_block",
            "enable_block_reuse",
        }
        or kv.get("tokens_per_block", 64) != 64
    ):
        raise ValueError(
            "TRT-LLM P/D requires an unquantized GPU KV pool with 64-token pages"
        )
    kv["tokens_per_block"] = 64
    if (
        config.get("enable_chunked_prefill")
        or config.get("disable_overlap_scheduler", True) is not True
    ):
        raise ValueError("TRT-LLM P/D requires unchunked, non-overlapped scheduling")
    transceiver = dict(config.get("cache_transceiver_config") or {})
    if transceiver.get("backend", "NIXL") != "NIXL":
        raise ValueError("TRT-LLM native P/D currently requires NIXL")
    transceiver.setdefault("kv_transfer_timeout_ms", 30000)
    if transceiver.get("kv_transfer_sender_future_timeout_ms", 1000) is None:
        # An unbounded C++ sender wait blocks the executor thread that must
        # process source cancellation when no decode receiver arrives.
        transceiver["kv_transfer_sender_future_timeout_ms"] = 1000
    transceiver["backend"] = "NIXL"
    config.update(
        cache_transceiver_config=transceiver,
        disable_overlap_scheduler=True,
        enable_chunked_prefill=False,
        kv_cache_config=kv,
    )
    return replace(
        build_pd_contract(model_path, config.get("max_seq_len")),
        logical_dtype="bfloat16",
    )


class TRTLLMPD:
    def __init__(self, role, backend, contract=None):
        self.role, self.backend, self.contract = role, backend, contract
        self.client = None
        self._native = None
        self.active: dict = {}
        if backend == "xavier":
            from .connector import XavierClient

            self.client = XavierClient()

    def bind_engine(self, engine):
        if self.backend == "nixl" and self.role == "prefill":
            from .native import NativeSourceReleaser

            self._native = NativeSourceReleaser(engine._executor.engine)

    def engine_stopped(self):
        if self._native is not None:
            self._native.engine_stopped()
            self._native = None

    def _release_native(self, state):
        request = state.get("request")
        if request is None:
            return
        context = getattr(request, "disaggregated_params", None)
        ticket = state.get("ticket")
        if ticket is None:
            ticket = getattr(context, "ctx_request_id", None)
        if ticket is None and self._native is not None:
            ticket = self._native.find_source(request.request_id)
        if self.role == "prefill" and ticket is not None:
            if self._native is None:
                raise RuntimeError("TRT native source release is not initialized")
            self._native.release(ticket)
        else:
            request.abort()

    async def prepare(self, request_id, tokens, transfer):
        prefill = self.role == "prefill"
        flag = "do_remote_decode" if prefill else "do_remote_prefill"
        if not isinstance(transfer, dict) or not transfer.get(flag):
            raise ValueError(f"TRT-LLM {self.role} replicas require P/D metadata")
        if self.client is not None:
            return await self._prepare_gpu(request_id, tokens, transfer)
        if prefill:
            if request_id in self.active:
                raise ValueError("A TRT-LLM P/D request with this ID is already active")
            state = dict(request=None, ticket=None, released=False)
            self.active[request_id] = state
            return state
        handoff = transfer.get("trt_llm")
        if (
            not isinstance(handoff, dict)
            or handoff.get("backend") != self.backend
            or handoff.get("prompt_digest") != prompt_digest(tokens)
        ):
            raise ValueError("Incompatible TRT-LLM P/D prompt or transport")
        if self.client is None:
            from tensorrt_llm.disaggregated_params import DisaggregatedParams

            if handoff.get("contract") != self.contract.fingerprint:
                raise ValueError("Incompatible TRT-LLM P/D checkpoint or KV contract")
            context = dict(handoff["context"])
            context["opaque_state"] = base64.b64decode(
                context["opaque_state"], validate=True
            )
            return dict(
                params=DisaggregatedParams(request_type="generation_only", **context)
            )

    async def _prepare_gpu(self, request_id, tokens, transfer):
        from ..xavier.backends.torch.gpu_transfer import finish_before_cancel

        if request_id in self.active:
            raise ValueError("A TRT-LLM P/D request with this ID is already active")
        heterogeneous = self.client.config.get("heterogeneous", False)
        if heterogeneous:
            handoff = transfer.get("sglang_xavier")
            mode = "host" if self.client.config.get("host_handoff") else "gpu"
            if not isinstance(handoff, dict) or handoff.get("mode") != mode:
                raise ValueError("Missing cross-engine Xavier handoff room")
            validate_handoff_scope(handoff)
            room = handoff["room"]
        elif self.role == "prefill":
            room = (uuid.uuid4().int & (2**63 - 1)) or 1
        else:
            handoff = transfer.get("trt_llm")
            if (
                not isinstance(handoff, dict)
                or handoff.get("backend") != "xavier"
                or handoff.get("prompt_digest") != prompt_digest(tokens)
                or handoff.get("namespace") != self.client.namespace
            ):
                raise ValueError("Incompatible TRT-LLM Xavier GPU handoff")
            room = handoff["room"]
        state = dict(
            request=None,
            released=False,
            prepared=False,
            context=XavierRequest(
                self.role, tuple(tokens), room, heterogeneous=heterogeneous
            ),
            handoff=handoff if heterogeneous else None,
        )
        self.active[request_id] = state

        async def reserve():
            await asyncio.to_thread(
                self.client.rpc,
                "prepare",
                room,
                self.client.namespace,
                prompt_digest(tokens),
                self.role,
                self.client.config["rank"],
                prompt_tokens=len(tokens),
            )
            state["prepared"] = True

        reservation = state["reservation"] = asyncio.create_task(reserve())
        try:
            try:
                await asyncio.shield(reservation)
            except asyncio.CancelledError:
                await finish_before_cancel(reservation)
                raise
            if state["released"]:
                raise asyncio.CancelledError("TRT-LLM P/D request was released")
            return state
        except BaseException:
            await self.release(request_id)
            raise

    def submit(self, engine, tokens, params, state):
        if self.client is not None:
            request = submit_xavier(engine, tokens, params, state["context"])
        else:
            from tensorrt_llm.disaggregated_params import DisaggregatedParams

            native = state.get("params") or DisaggregatedParams(
                request_type="context_only"
            )
            request = engine.generate_async(
                tokens, params, streaming=True, disaggregated_params=native
            )
        state["request"] = request
        return request

    def check_import(self, state):
        context = state.get("context")
        if context is not None and context.role == "decode":
            if context.imported_tokens != max(len(context.tokens) - 1, 0):
                raise RuntimeError("TRT-LLM Xavier decode did not import the P/D KV")

    async def publish(self, request_id, tokens, output):
        state = self.active.get(request_id)
        if state is None:
            raise asyncio.CancelledError("TRT-LLM prefill was released")
        handoff = dict(backend=self.backend, prompt_digest=prompt_digest(tokens))
        if self.client is None:
            context = output.disaggregated_params
            if context is None:
                raise RuntimeError(
                    "TRT-LLM prefill did not return native context metadata"
                )
            state["ticket"] = context.ctx_request_id
            handoff.update(
                contract=self.contract.fingerprint,
                context=dict(
                    first_gen_tokens=context.first_gen_tokens,
                    ctx_request_id=context.ctx_request_id,
                    opaque_state=base64.b64encode(context.opaque_state).decode("ascii"),
                    draft_tokens=context.draft_tokens,
                ),
            )
        else:
            if self.client.config.get("heterogeneous"):
                from .gpu import service

                if not output.outputs or not output.outputs[0].token_ids:
                    raise RuntimeError("Incomplete TRT-LLM cross-engine prefill")
                await asyncio.to_thread(
                    service().rpc,
                    "finalize",
                    state["context"].room,
                    output.outputs[0].token_ids[0],
                )
                return dict(do_remote_prefill=True, sglang_xavier=state["handoff"])
            source = await asyncio.to_thread(
                self.client.rpc, "source", state["context"].room
            )
            if source is None:
                raise RuntimeError("TRT-LLM prefill did not publish its GPU KV")
            handoff.update(
                namespace=self.client.namespace,
                room=state["context"].room,
            )
        return dict(do_remote_prefill=True, trt_llm=handoff)

    async def release(self, request_id):
        state = self.active.pop(request_id, None)
        if state is not None:
            state["released"] = True
            if self.client is None:
                from ..xavier.backends.torch.gpu_transfer import finish_before_cancel

                await finish_before_cancel(
                    asyncio.create_task(asyncio.to_thread(self._release_native, state))
                )
            else:
                from ..xavier.backends.torch.gpu_transfer import finish_before_cancel

                await finish_before_cancel(
                    asyncio.create_task(self._release_gpu(state))
                )

    async def _release_gpu(self, state):
        state["context"].cancelled = True
        reservation = state.get("reservation")
        if reservation is not None:
            await asyncio.gather(reservation, return_exceptions=True)
        try:
            if state.get("request") is not None:
                state["request"].abort()
        finally:
            if state["prepared"]:
                room = state["context"].room
                if self.role == "prefill":
                    from .gpu import service

                    await asyncio.to_thread(service().rpc, "abort", room)
                await asyncio.to_thread(
                    self.client.rpc,
                    "release",
                    room,
                    None if self.role == "prefill" else self.role,
                )

    def close(self):
        states, self.active = self.active, {}
        try:
            for state in states.values():
                state["released"] = True
                try:
                    try:
                        if self.client is None:
                            self._release_native(state)
                        else:
                            state["context"].cancelled = True
                            if state.get("request") is not None:
                                state["request"].abort()
                    finally:
                        if self.client is not None and state.get("prepared"):
                            room = state["context"].room
                            if self.role == "prefill":
                                from .gpu import service

                                service().rpc("abort", room)
                            self.client.rpc(
                                "release",
                                room,
                                None if self.role == "prefill" else self.role,
                            )
                except Exception:
                    logger.warning(
                        "Could not release TRT-LLM P/D source", exc_info=True
                    )
        finally:
            if self.client is not None:
                self.client.close()
