from __future__ import annotations

import asyncio
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

from .admission import AdmissionRejected, CapacityGate, GateSnapshot
from .metrics import RouterMetrics
from .tokenization_worker import (
    estimate_in_worker,
    initialize_tokenization_worker,
    ping_worker,
)
from .tokenizer import TokenBudget
from .tokenizer_asset import DEFAULT_TOKENIZER_ASSET_FILES

logger = logging.getLogger("deepseek_v4_token_router.tokenization")


class TokenizationWorkerUnavailable(RuntimeError):
    """Raised when the isolated tokenization process pool is unavailable."""


class TokenizationService:
    """Run prompt rendering/tokenization in a bounded spawn process pool."""

    def __init__(
        self,
        tokenizer_path: Path,
        metrics: RouterMetrics,
        *,
        reserve_tokens: int,
        default_output_tokens: int,
        max_workers: int,
        max_active: int,
        max_queue: int,
        queue_timeout_seconds: float,
        retry_after_seconds: int,
        tokenizer_asset_files: tuple[str, ...] = DEFAULT_TOKENIZER_ASSET_FILES,
    ) -> None:
        self._tokenizer_path = tokenizer_path
        self._reserve_tokens = reserve_tokens
        self._default_output_tokens = default_output_tokens
        self._max_workers = max_workers
        self._tokenizer_asset_files = tokenizer_asset_files
        self._asset_fingerprint = ""
        self._asset_revision = ""
        self._metrics = metrics
        self._gate = CapacityGate(
            "tokenization",
            max_active=max_active,
            max_queue=max_queue,
            queue_timeout_seconds=queue_timeout_seconds,
            retry_after_seconds=retry_after_seconds,
        )
        self._executor = self._create_executor()
        self._replace_lock = asyncio.Lock()
        self._closed = False
        self._worker_pids: tuple[int, ...] = ()

    def _create_executor(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_tokenization_worker,
            initargs=(
                str(self._tokenizer_path),
                self._reserve_tokens,
                self._default_output_tokens,
                self._tokenizer_asset_files,
            ),
        )

    async def start(self) -> None:
        """Prestart all workers and verify backend credentials were removed."""
        if self._closed:
            raise TokenizationWorkerUnavailable("Tokenization service is closed")
        executor = self._executor
        loop = asyncio.get_running_loop()
        results: list[tuple[int, bool, bool, str, str]] = []

        # ProcessPoolExecutor starts processes lazily. Multiple bounded waves
        # ensure every configured spawn worker completes its initializer even
        # when loading the tokenizer takes different amounts of time.
        for _ in range(3):
            probes = [
                loop.run_in_executor(executor, ping_worker, 0.2)
                for _ in range(self._max_workers * 2)
            ]
            try:
                results.extend(await asyncio.gather(*probes))
            except Exception as exc:
                raise TokenizationWorkerUnavailable(
                    f"Unable to start tokenization workers: {exc}"
                ) from exc
            if any(
                api_key_present or internal_token_present
                for _, api_key_present, internal_token_present, _, _ in results
            ):
                raise TokenizationWorkerUnavailable(
                    "Tokenization worker retained a Router credential"
                )
            fingerprints = {fingerprint for _, _, _, fingerprint, _ in results}
            revisions = {revision for _, _, _, _, revision in results}
            if len(fingerprints) != 1 or "" in fingerprints:
                raise TokenizationWorkerUnavailable(
                    "Tokenization workers loaded different Tokenizer assets"
                )
            if len(revisions) != 1:
                raise TokenizationWorkerUnavailable(
                    "Tokenization workers loaded different Tokenizer asset revisions"
                )
            worker_pids = {pid for pid, _, _, _, _ in results}
            if len(worker_pids) >= self._max_workers:
                self._asset_fingerprint = fingerprints.pop()
                self._asset_revision = revisions.pop()
                self._worker_pids = tuple(sorted(worker_pids))
                logger.info("Tokenization workers started: pids=%s", self._worker_pids)
                return

        raise TokenizationWorkerUnavailable(
            "Tokenization process pool did not start all configured workers"
        )

    async def estimate(
        self, payload: dict[str, Any], *, input_bytes: int
    ) -> TokenBudget:
        try:
            await self._gate.acquire()
        except AdmissionRejected as exc:
            await self._metrics.increment_tokenization_rejected(exc.reason)
            await self._publish_gate_snapshot()
            raise

        await self._publish_gate_snapshot()
        started = time.monotonic()
        outcome = "completed"
        future: asyncio.Future[TokenBudget] | None = None
        executor = self._executor
        try:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(
                executor,
                estimate_in_worker,
                payload,
            )
            # Shield the process future so request cancellation does not make
            # the slot available while CPU tokenization is still running.
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            outcome = "cancelled"
            if future is not None:
                try:
                    await asyncio.shield(future)
                except Exception:
                    pass
            raise
        except BrokenProcessPool as exc:
            outcome = "worker_unavailable"
            await self._replace_broken_executor(executor)
            raise TokenizationWorkerUnavailable(
                "Tokenization worker process failed"
            ) from exc
        except Exception:
            outcome = "failed"
            raise
        finally:
            await self._metrics.observe_tokenization(
                duration_seconds=time.monotonic() - started,
                input_bytes=input_bytes,
                outcome=outcome,
            )
            await self._gate.release()
            await self._publish_gate_snapshot()

    @property
    def asset_fingerprint(self) -> str:
        return self._asset_fingerprint

    @property
    def asset_revision(self) -> str:
        return self._asset_revision

    async def snapshot(self) -> GateSnapshot:
        return await self._gate.snapshot()

    @property
    def worker_pids(self) -> tuple[int, ...]:
        return self._worker_pids

    async def _publish_gate_snapshot(self) -> None:
        snapshot = await self._gate.snapshot()
        await self._metrics.set_tokenization_capacity(
            active=snapshot.active,
            waiting=snapshot.waiting,
        )

    async def _replace_broken_executor(
        self, failed_executor: ProcessPoolExecutor
    ) -> None:
        async with self._replace_lock:
            if self._closed or self._executor is not failed_executor:
                return
            replacement = self._create_executor()
            self._executor = replacement
            self._worker_pids = ()
            failed_executor.shutdown(wait=False, cancel_futures=True)
            try:
                await self.start()
            except Exception:
                # Keep the replacement assigned so subsequent submissions
                # receive BrokenProcessPool and can trigger another guarded
                # recovery attempt instead of using the failed executor.
                logger.exception("Failed to restart tokenization process pool")

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        executor = self._executor
        self._worker_pids = ()
        await asyncio.to_thread(
            executor.shutdown,
            wait=True,
            cancel_futures=True,
        )
