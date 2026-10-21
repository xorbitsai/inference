# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Atomic, private JSON journals for Worker control state."""

import contextlib
import json
import os
import tempfile
from pathlib import Path


class WorkerJournalError(RuntimeError):
    pass


def read_worker_journal(path: Path, default: dict) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as error:
        raise WorkerJournalError(
            f"Cannot read Worker journal {path}: {error}"
        ) from error


def write_worker_journal(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".worker-state-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
