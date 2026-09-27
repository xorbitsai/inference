"""Read application logs from the API node or a registered worker."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import xoscar as xo
from fastapi import Depends, HTTPException, Query, Security

from ...core.runtime_logs import read_runtime_log
from ..dependencies import get_api
from ..responses import JSONResponse

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI

logger = logging.getLogger(__name__)


async def _worker_addresses(api: "RESTfulAPI") -> list[str]:
    supervisor = await api._get_supervisor_ref()
    if supervisor is None:
        raise RuntimeError("Supervisor is unavailable")
    status = await supervisor.get_status()
    return sorted(status.get("workers", {}))


async def _worker_addresses_or_503(api: "RESTfulAPI") -> list[str]:
    try:
        return await _worker_addresses(api)
    except Exception as exc:
        logger.warning("Could not retrieve workers for runtime logs", exc_info=True)
        raise HTTPException(
            status_code=503, detail="Supervisor is unavailable"
        ) from exc


async def list_runtime_log_sources(
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        workers = await _worker_addresses(api)
    except Exception:
        logger.warning("Could not list workers for runtime logs", exc_info=True)
        workers = []

    if api._supervisor_address in workers:
        sources = [{"id": "local", "label": "Local"}]
    else:
        sources = [{"id": "supervisor", "label": "Supervisor"}]
    sources.extend(
        {"id": address, "label": f"Worker {address}"}
        for address in workers
        if address != api._supervisor_address
    )
    return JSONResponse(content={"sources": sources})


async def read_runtime_logs(
    source: str = Query("supervisor", max_length=256),
    cursor: str = Query("", max_length=100),
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    if source in ("local", "supervisor"):
        if source == "local":
            workers = await _worker_addresses_or_503(api)
            if api._supervisor_address not in workers:
                raise HTTPException(status_code=404, detail="Log source not found")
        try:
            result = await asyncio.to_thread(read_runtime_log, cursor)
        except OSError as exc:
            logger.warning("Could not read runtime logs from %s", source, exc_info=True)
            raise HTTPException(
                status_code=503, detail="Runtime logs are unavailable"
            ) from exc
    else:
        workers = await _worker_addresses_or_503(api)
        if source not in workers:
            raise HTTPException(status_code=404, detail="Log source not found")
        try:
            from ...core.worker import WorkerActor

            worker = await xo.actor_ref(address=source, uid=WorkerActor.default_uid())
            result = await asyncio.wait_for(
                worker.read_runtime_logs(cursor), timeout=10
            )
        except Exception:
            logger.warning(
                "Could not read runtime logs from worker %s", source, exc_info=True
            )
            raise HTTPException(status_code=503, detail="Worker logs are unavailable")
    return JSONResponse(content=result)


def register_routes(api: "RESTfulAPI") -> None:
    dependencies = (
        [Security(api._auth_service, scopes=["logs:list"])]
        if api.is_authenticated()
        else None
    )
    api._router.add_api_route(
        "/v1/cluster/runtime-logs/sources",
        list_runtime_log_sources,
        methods=["GET"],
        dependencies=dependencies,
    )
    api._router.add_api_route(
        "/v1/cluster/runtime-logs",
        read_runtime_logs,
        methods=["GET"],
        dependencies=dependencies,
    )
