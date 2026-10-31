# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Storage selection for Supervisor-owned Worker handoff/relaunch records."""

from pathlib import Path
from typing import Any, Literal

from ..extensions import (
    SupervisorJournalStoreContext,
    _require_sync_result,
    resolve_supervisor_journal_store,
)
from .worker_state import WorkerJournalError, read_worker_journal, write_worker_journal


class LocalSupervisorJournal:
    def __init__(self, path: Path):
        self.path = path

    def load(self, default: dict) -> dict:
        return read_worker_journal(self.path, default)

    def save(self, payload: dict) -> None:
        write_worker_journal(self.path, payload)


class _ProviderJournal:
    def __init__(self, provider: Any):
        self.provider = provider

    def load(self, default: dict) -> dict:
        try:
            payload = self.provider.load(default)
        except WorkerJournalError:
            raise
        except Exception:
            raise WorkerJournalError("Supervisor journal load failed") from None
        return _require_sync_result(payload, "Supervisor journal load")

    def save(self, payload: dict) -> None:
        _require_sync_result(self.provider.save(payload), "Supervisor journal save")


def supervisor_journal(
    owner: Any, kind: Literal["worker_ownership", "worker_relaunch"], path: Path
) -> Any:
    control = getattr(owner, "_controller_control", None)
    store = resolve_supervisor_journal_store(
        SupervisorJournalStoreContext(
            kind,
            str(path),
            getattr(owner, "control_address", None)
            or getattr(owner, "address", "")
            or "",
            control.session if control else None,
        )
    )
    return LocalSupervisorJournal(path) if store is None else _ProviderJournal(store)
