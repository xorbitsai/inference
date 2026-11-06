# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Observe current serving limits without changing compatibility responses."""

import asyncio
from typing import Any, Dict

from ..model.llm.context_length import normalize_runtime_positive_int
from .worker_registry import call_worker_route


async def get_model_runtime_info(supervisor: Any, model_uid: str) -> Dict[str, Any]:
    replica_uids = supervisor._iter_active_replica_model_uids(model_uid)

    async def probe(replica_uid: str) -> Dict[str, Any]:
        async def read():
            worker = supervisor._replica_model_uid_to_worker.get(replica_uid)
            if isinstance(worker, (list, tuple)):
                worker = worker[0] if worker else None
            if worker is None:
                return None
            model = await call_worker_route(
                supervisor, replica_uid, worker, "get_model", model_uid=replica_uid
            )
            return await model.get_runtime_context_length()

        try:
            length = normalize_runtime_positive_int(
                await asyncio.wait_for(read(), timeout=3)
            )
        except Exception:
            # Loading/reloading, older workers, and unsupported engines have
            # no confirmed value. Never substitute the model family's limit.
            length = None
        return {"replica_uid": replica_uid, "runtime_context_length": length}

    replicas = await asyncio.gather(*(probe(uid) for uid in replica_uids))
    lengths = [replica["runtime_context_length"] for replica in replicas]
    topology_unchanged = replica_uids == supervisor._iter_active_replica_model_uids(
        model_uid
    )
    return {
        "model_uid": model_uid,
        "runtime_context_length": (
            min(lengths)
            if lengths
            and all(length is not None for length in lengths)
            and topology_unchanged
            else None
        ),
        "replicas": replicas,
    }
