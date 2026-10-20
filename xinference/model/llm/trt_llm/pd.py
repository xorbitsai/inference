# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Same-engine TRT P/D: native NIXL or request-owned Xavier CPU pages."""

import asyncio
import base64
import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..xavier.pd_contract import build_pd_contract, prompt_digest
from .xavier import EXECUTION_OPTIONS

logger = logging.getLogger(__name__)

# TRT's embedded executor schedules on a background thread. Register its client
# ID before the scheduler can inspect the request; never key by prompt text.
_lock = threading.Lock()
_requests: dict = {}


@dataclass
class XavierRequest:
    role: str
    tokens: tuple
    keys: list
    pages: list = field(default_factory=list)
    imported_tokens: int = 0


def submit_xavier(engine, tokens, params, state):
    with _lock:
        request = engine.generate_async(
            tokens, params, streaming=True, cache_salt=uuid.uuid4().hex
        )
        _requests[request.request_id] = state
    return request


def request_state(request):
    with _lock:
        return _requests.get(getattr(request, "py_client_id", None))


def forget_request(request):
    with _lock:
        _requests.pop(request.request_id, None)


def configure_native(model_path, config, replica):
    if replica.get("role") not in ("prefill", "decode"):
        raise ValueError("TRT-LLM native P/D requires a prefill or decode role")
    if config.get("kv_connector_config") is not None:
        raise ValueError("TRT-LLM native P/D cannot use a KV connector")
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
        self.active: dict = {}
        if backend == "xavier":
            from .connector import XavierClient

            self.client = XavierClient()

    def _keys(self, tokens, nonce):
        count = (max(len(tokens) - 1, 0) + 63) // 64
        prefix = prompt_digest(tokens) + ":" + nonce
        return [
            hashlib.sha256(f"{prefix}:{i}".encode()).hexdigest() for i in range(count)
        ]

    async def prepare(self, request_id, tokens, transfer):
        prefill = self.role == "prefill"
        flag = "do_remote_decode" if prefill else "do_remote_prefill"
        if not isinstance(transfer, dict) or not transfer.get(flag):
            raise ValueError(f"TRT-LLM {self.role} replicas require P/D metadata")
        if prefill:
            if request_id in self.active:
                raise ValueError("A TRT-LLM P/D request with this ID is already active")
            state = dict(request=None, ticket=None, released=False)
            self.active[request_id] = state
            if self.client is not None:
                nonce = uuid.uuid4().hex
                keys = self._keys(tokens, nonce)
                reservation = asyncio.create_task(
                    asyncio.to_thread(
                        self.client.rpc, "prepare_handoff", self.client.namespace, keys
                    )
                )
                try:
                    try:
                        state["ticket"] = await asyncio.shield(reservation)
                    except asyncio.CancelledError:
                        # The RPC may have committed even though its caller was
                        # cancelled. Recover the ticket before releasing it.
                        state["ticket"] = await reservation
                        raise
                    if state["released"]:
                        await asyncio.to_thread(
                            self.client.rpc, "release_handoff", state["ticket"]
                        )
                        raise asyncio.CancelledError(
                            "TRT-LLM P/D reservation was cancelled"
                        )
                    state["context"] = XavierRequest("prefill", tuple(tokens), keys)
                    state["nonce"] = nonce
                except BaseException:
                    await self.release(request_id)
                    raise
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
        if handoff.get("namespace") != self.client.namespace:
            raise ValueError("Incompatible TRT-LLM Xavier namespace")
        keys = self._keys(tokens, handoff["nonce"])
        ready = await asyncio.to_thread(
            self.client.rpc,
            "validate_handoff",
            self.client.namespace,
            keys,
            handoff["ticket"],
        )
        if not ready:
            raise ValueError("Missing, expired or incomplete TRT-LLM Xavier handoff")
        pages = await asyncio.to_thread(self.client.read_prefix, keys)
        if len(pages) != len(keys):
            raise RuntimeError("TRT-LLM Xavier P/D pages disappeared before import")
        return dict(context=XavierRequest("decode", tuple(tokens), keys, pages))

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
            ready = await asyncio.to_thread(
                self.client.rpc,
                "validate_handoff",
                self.client.namespace,
                state["context"].keys,
                state["ticket"],
            )
            if not ready:
                raise RuntimeError(
                    "TRT-LLM prefill did not publish its complete Xavier KV"
                )
            handoff.update(
                namespace=self.client.namespace,
                nonce=state["nonce"],
                ticket=state["ticket"],
            )
        return dict(do_remote_prefill=True, trt_llm=handoff)

    async def release(self, request_id):
        state = self.active.pop(request_id, None)
        if state is not None:
            state["released"] = True
            request = state.get("request")
            try:
                if request is not None:
                    request.abort()
            finally:
                if self.client is not None and state.get("ticket"):
                    await asyncio.to_thread(
                        self.client.rpc, "release_handoff", state["ticket"]
                    )

    def close(self):
        states, self.active = self.active, {}
        try:
            for state in states.values():
                state["released"] = True
                try:
                    try:
                        if state.get("request") is not None:
                            state["request"].abort()
                    finally:
                        if self.client is not None and state.get("ticket"):
                            self.client.rpc("release_handoff", state["ticket"])
                except Exception:
                    logger.warning(
                        "Could not release TRT-LLM P/D source", exc_info=True
                    )
        finally:
            if self.client is not None:
                self.client.close()
