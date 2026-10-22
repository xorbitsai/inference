# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Per-account favorites, with a SQLite default and an optional storage provider."""

from __future__ import annotations

import inspect
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, List, Optional, Protocol


class ModelFavoritesStore(Protocol):
    """Synchronous operations. Owners are authenticated user IDs, or None without auth.

    Implementations must isolate owners, persist changes across processes, and
    make add/remove idempotent without replacing the owner's whole collection.
    Route handlers run these operations in the API's thread pool.
    """

    def list(self, owner: Optional[int]) -> List[str]: ...

    def add(self, owner: Optional[int], model_name: str) -> None: ...

    def remove(self, owner: Optional[int], model_name: str) -> None: ...


class SQLiteModelFavoritesStore:
    def __init__(self, db_path: str):
        if not db_path or db_path == ":memory:":
            raise ValueError("Model favorites require a persistent SQLite file")
        self._path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path, timeout=10)) as conn, conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS model_favorites ("
                "owner INTEGER NOT NULL, model_name TEXT NOT NULL, "
                "PRIMARY KEY (owner, model_name))"
            )

    def list(self, owner: Optional[int]) -> List[str]:
        with closing(sqlite3.connect(self._path, timeout=10)) as conn:
            return [
                row[0]
                for row in conn.execute(
                    "SELECT model_name FROM model_favorites WHERE owner=? "
                    "ORDER BY model_name",
                    (0 if owner is None else owner,),
                )
            ]

    def add(self, owner: Optional[int], model_name: str) -> None:
        with closing(sqlite3.connect(self._path, timeout=10)) as conn, conn:
            conn.execute(
                "INSERT INTO model_favorites (owner, model_name) VALUES (?, ?) "
                "ON CONFLICT (owner, model_name) DO NOTHING",
                (0 if owner is None else owner, model_name),
            )

    def remove(self, owner: Optional[int], model_name: str) -> None:
        with closing(sqlite3.connect(self._path, timeout=10)) as conn, conn:
            conn.execute(
                "DELETE FROM model_favorites WHERE owner=? AND model_name=?",
                (0 if owner is None else owner, model_name),
            )


def create_model_favorites_store(db_path: str, auth_store: Any) -> ModelFavoritesStore:
    from ..extensions import ModelFavoritesStoreContext, resolve_model_favorites_store

    store = resolve_model_favorites_store(
        ModelFavoritesStoreContext(db_path, auth_store)
    )
    if store is None:
        return SQLiteModelFavoritesStore(db_path)
    for name in ("list", "add", "remove"):
        method = getattr(store, name, None)
        if (
            not callable(method)
            or getattr(method, "__func__", method) is getattr(ModelFavoritesStore, name)
            or inspect.iscoroutinefunction(method)
            or inspect.iscoroutinefunction(getattr(method, "__call__", None))
            or inspect.isasyncgenfunction(method)
            or inspect.isasyncgenfunction(getattr(method, "__call__", None))
        ):
            raise TypeError(f"Model favorites store must implement synchronous {name}")
    return store
