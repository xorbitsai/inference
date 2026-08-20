# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Hot-reloadable runtime state for the independent Token Router process."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

import httpx

from .admission import CapacityGate
from .classifier import RoutingPolicy
from .config import RouterConfig
from .metrics import RouterMetrics
from .tokenization import TokenizationService


class RouterDisabled(RuntimeError):
    """Raised when a disabled Router refuses a new request."""


@dataclass
class RuntimeSnapshot:
    config: RouterConfig
    tokenization: TokenizationService
    policy: RoutingPolicy
    gates: Dict[str, CapacityGate]
    client: httpx.AsyncClient
    active_requests: int = 0
    draining: bool = False
    closed: bool = False

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        await self.client.aclose()
        await self.tokenization.aclose()


class RouterRuntime:
    """Own the active immutable config snapshot and drain replaced snapshots."""

    def __init__(
        self,
        config: RouterConfig,
        metrics: Optional[RouterMetrics] = None,
        on_swap: Optional[Callable[[RuntimeSnapshot], None]] = None,
    ) -> None:
        self.metrics = metrics or RouterMetrics()
        self._current = self._build_snapshot(config)
        self._lock = asyncio.Lock()
        self._started = False
        self._on_swap = on_swap
        self._drain_tasks: set[asyncio.Task[None]] = set()

    def _build_snapshot(self, config: RouterConfig) -> RuntimeSnapshot:
        tokenization = TokenizationService(
            config.tokenizer_path,
            self.metrics,
            reserve_tokens=config.context_reserve_tokens,
            default_output_tokens=config.default_output_tokens,
            max_workers=config.tokenization.max_workers,
            max_active=config.tokenization.max_active,
            max_queue=config.tokenization.max_queue,
            queue_timeout_seconds=config.tokenization.queue_timeout_seconds,
            retry_after_seconds=config.tokenization.retry_after_seconds,
            tokenizer_asset_files=config.tokenizer_asset_files,
        )
        policy = RoutingPolicy(
            backends=config.backends,
            rules=config.rules,
            default_action=config.default_action,
        )
        gates = {
            backend.id: CapacityGate(
                backend.id,
                max_active=backend.max_active,
                max_queue=backend.max_queue,
                queue_timeout_seconds=backend.queue_timeout_seconds,
                retry_after_seconds=backend.retry_after_seconds,
            )
            for backend in config.backends
        }
        timeout = httpx.Timeout(
            connect=config.connect_timeout_seconds,
            read=config.request_timeout_seconds,
            write=config.request_timeout_seconds,
            pool=config.connect_timeout_seconds,
        )
        return RuntimeSnapshot(
            config=config,
            tokenization=tokenization,
            policy=policy,
            gates=gates,
            client=httpx.AsyncClient(timeout=timeout, http2=False),
        )

    def set_on_swap(self, callback: Callable[[RuntimeSnapshot], None]) -> None:
        self._on_swap = callback
        callback(self._current)

    @property
    def current(self) -> RuntimeSnapshot:
        return self._current

    async def start(self) -> None:
        if self._started:
            return
        await self._current.tokenization.start()
        self._started = True
        self._notify_swap(self._current)

    async def acquire(self) -> RuntimeSnapshot:
        async with self._lock:
            snapshot = self._current
            if not snapshot.config.enabled:
                raise RouterDisabled("Token Router is disabled")
            snapshot.active_requests += 1
            return snapshot

    async def release(self, snapshot: RuntimeSnapshot) -> None:
        close = False
        async with self._lock:
            snapshot.active_requests = max(0, snapshot.active_requests - 1)
            close = snapshot.draining and snapshot.active_requests == 0
        if close:
            await snapshot.close()

    async def apply(self, config: RouterConfig) -> None:
        replacement = self._build_snapshot(config)
        try:
            await replacement.tokenization.start()
        except Exception:
            await replacement.close()
            raise

        old: RuntimeSnapshot
        async with self._lock:
            old = self._current
            old.draining = True
            self._current = replacement
        self._notify_swap(replacement)
        if old.active_requests == 0:
            await old.close()
        else:
            task = asyncio.create_task(self._drain(old))
            self._drain_tasks.add(task)
            task.add_done_callback(self._drain_tasks.discard)

    async def _drain(self, snapshot: RuntimeSnapshot) -> None:
        while snapshot.active_requests:
            await asyncio.sleep(0.1)
        await snapshot.close()

    def _notify_swap(self, snapshot: RuntimeSnapshot) -> None:
        if self._on_swap is not None:
            self._on_swap(snapshot)

    async def summary(self) -> Dict[str, Any]:
        snapshot = self._current
        pools = {
            name: asdict(await gate.snapshot()) for name, gate in snapshot.gates.items()
        }
        tokenization = asdict(await snapshot.tokenization.snapshot())
        tokenization["worker_pids"] = snapshot.tokenization.worker_pids
        return {
            "enabled": snapshot.config.enabled,
            "revision": snapshot.config.revision,
            "active_requests": snapshot.active_requests,
            "pools": pools,
            "tokenization": tokenization,
            "tokenizer_asset": {
                "asset_id": snapshot.config.tokenizer_asset_id,
                "revision": snapshot.tokenization.asset_revision,
                "fingerprint": snapshot.tokenization.asset_fingerprint,
            },
        }

    async def aclose(self) -> None:
        async with self._lock:
            current = self._current
            current.draining = True
        if current.active_requests == 0:
            await current.close()
        else:
            await self._drain(current)
        if self._drain_tasks:
            await asyncio.gather(*self._drain_tasks, return_exceptions=True)
