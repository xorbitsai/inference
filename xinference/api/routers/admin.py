"""Admin / cluster / infrastructure route registration and handlers."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import TYPE_CHECKING, Any

import aiohttp
from fastapi import Depends, HTTPException, Query, Request, Security

from ..._version import get_versions
from ..dependencies import get_api
from ..oauth2.types import LoginUserForm
from ..responses import JSONResponse

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI

logger = logging.getLogger(__name__)


# --- Handlers (top-level, inject dependencies via Depends) ---


async def get_status(api: "RESTfulAPI" = Depends(get_api)) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        data = await supervisor_ref.get_status()
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def get_address(api: "RESTfulAPI" = Depends(get_api)) -> JSONResponse:
    return JSONResponse(content=api._supervisor_address)


async def login_for_access_token(
    request: Request, api: "RESTfulAPI" = Depends(get_api)
) -> JSONResponse:
    form_data = LoginUserForm.parse_obj(await request.json())
    result = api._auth_service.generate_token_for_user(
        form_data.username, form_data.password
    )
    return JSONResponse(content=result)


async def is_cluster_authenticated(
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    return JSONResponse(content={"auth": api.is_authenticated()})


async def get_cluster_device_info(
    api: "RESTfulAPI" = Depends(get_api),
    detailed: bool = Query(False),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        data = await supervisor_ref.get_cluster_device_info(detailed=detailed)
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def get_cluster_version() -> JSONResponse:
    try:
        data = get_versions()
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def get_devices_count(
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        data = await supervisor_ref.get_devices_count()
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def get_workers_info(
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        res = await supervisor_ref.get_workers_info()
        return JSONResponse(content=res)
    except ValueError as re:
        logger.error(re, exc_info=True)
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def get_supervisor_info(
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        res = await supervisor_ref.get_supervisor_info()
        return JSONResponse(content=res)
    except ValueError as re:
        logger.error(re, exc_info=True)
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def abort_cluster(
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        res = await supervisor_ref.abort_cluster()
        os.kill(os.getpid(), signal.SIGINT)
        return JSONResponse(content={"result": res})
    except ValueError as re:
        logger.error(re, exc_info=True)
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def list_cached_models(
    api: "RESTfulAPI" = Depends(get_api),
    model_name: str = Query(None),
    worker_ip: str = Query(None),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        data = await supervisor_ref.list_cached_models(model_name, worker_ip)
        resp = {"list": data}
        return JSONResponse(content=resp)
    except ValueError as re:
        logger.error(re, exc_info=True)
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def list_model_files(
    api: "RESTfulAPI" = Depends(get_api),
    model_version: str = Query(None),
    worker_ip: str = Query(None),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        data = await supervisor_ref.list_deletable_models(model_version, worker_ip)
        response = {
            "model_version": model_version,
            "worker_ip": worker_ip,
            "paths": data,
        }
        return JSONResponse(content=response)
    except ValueError as re:
        logger.error(re, exc_info=True)
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def confirm_and_remove_model(
    api: "RESTfulAPI" = Depends(get_api),
    model_version: str = Query(None),
    worker_ip: str = Query(None),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        res = await supervisor_ref.confirm_and_remove_model(
            model_version=model_version, worker_ip=worker_ip
        )
        return JSONResponse(content={"result": res})
    except ValueError as re:
        logger.error(re, exc_info=True)
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def list_virtual_envs(
    api: "RESTfulAPI" = Depends(get_api),
    model_name: str = Query(None),
    model_engine: str = Query(None),
    worker_ip: str = Query(None),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        data = await supervisor_ref.list_virtual_envs(
            model_name, model_engine, worker_ip
        )
        resp = {"list": data}
        return JSONResponse(content=resp)
    except ValueError as re:
        logger.error(re, exc_info=True)
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def remove_virtual_env(
    api: "RESTfulAPI" = Depends(get_api),
    model_name: str = Query(None),
    model_engine: str = Query(None),
    python_version: str = Query(None),
    worker_ip: str = Query(None),
) -> JSONResponse:
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name parameter is required")
    try:
        supervisor_ref = await api._get_supervisor_ref()
        res = await supervisor_ref.remove_virtual_env(
            model_name=model_name,
            model_engine=model_engine,
            python_version=python_version,
            worker_ip=worker_ip,
        )
        return JSONResponse(content={"result": res})
    except ValueError as re:
        logger.error(re, exc_info=True)
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def get_progress(
    request_id: str,
    api: "RESTfulAPI" = Depends(get_api),
) -> JSONResponse:
    try:
        supervisor_ref = await api._get_supervisor_ref()
        result = {"progress": await supervisor_ref.get_progress(request_id)}
        return JSONResponse(content=result)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def get_ui_config() -> JSONResponse:
    grafana_datasource = os.environ.get("XINFERENCE_GRAFANA_DATASOURCE", "")
    return JSONResponse(
        content={
            "grafana_url": os.environ.get("XINFERENCE_GRAFANA_URL", ""),
            "grafana_datasource": grafana_datasource,
            "grafana_alert_datasource": os.environ.get(
                "XINFERENCE_GRAFANA_ALERT_DATASOURCE", ""
            )
            or grafana_datasource,
            "grafana_dashboard_uid": os.environ.get(
                "XINFERENCE_GRAFANA_DASHBOARD_UID", "xinference-overview"
            ),
            "cluster_name": os.environ.get("XINFERENCE_CLUSTER_NAME", ""),
            "es_enabled": bool(os.environ.get("XINFERENCE_ES_URL", "")),
        }
    )


async def search_logs(
    q: str = "",
    level: str = "",
    module: str = "",
    node: str = "",
    log_type: str = "",
    time_from: str = "now-1h",
    time_to: str = "now",
    size: int = 200,
    page_from: int = 0,
) -> JSONResponse:
    es_url = os.environ.get("XINFERENCE_ES_URL", "")
    if not es_url:
        raise HTTPException(status_code=503, detail="Elasticsearch is not configured")

    es_index = os.environ.get("XINFERENCE_ES_INDEX", "xinference-logs-*")
    es_auth = os.environ.get("XINFERENCE_ES_AUTH", "")

    size = max(1, min(size, 500))
    page_from = max(0, min(page_from, 10000 - size))

    must = []
    filter_clauses: list[dict[str, Any]] = [
        {"range": {"@timestamp": {"gte": time_from, "lte": time_to}}}
    ]

    if q:
        must.append(
            {
                "simple_query_string": {
                    "query": q,
                    "fields": ["message"],
                    "default_operator": "AND",
                }
            }
        )

    for field, value in [
        ("level", level),
        ("module", module),
        ("node", node),
        ("log_type", log_type),
    ]:
        if value:
            terms = [v.strip() for v in value.split(",") if v.strip()]
            if terms:
                filter_clauses.append({"terms": {field: terms}})

    body: dict[str, Any] = {
        "query": {"bool": {"must": must, "filter": filter_clauses}},
        "sort": [{"@timestamp": "desc"}],
        "from": page_from,
        "size": size,
        "_source": {"excludes": ["@version"]},
    }

    headers = {"Content-Type": "application/json"}
    auth = None
    if es_auth:
        if es_auth.startswith("ApiKey "):
            headers["Authorization"] = es_auth
        else:
            parts = es_auth.split(":", 1)
            if len(parts) == 2:
                auth = aiohttp.BasicAuth(parts[0], parts[1])

    url = f"{es_url.rstrip('/')}/{es_index}/_search"

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout, auth=auth) as session:
            async with session.post(url, json=body, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(
                        "ES query failed: status=%d body=%s", resp.status, text[:500]
                    )
                    raise HTTPException(
                        status_code=502, detail="Elasticsearch query failed"
                    )
                data = await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error("ES connection error or timeout: %s", e)
        raise HTTPException(
            status_code=502,
            detail="Failed to connect to Elasticsearch or query timed out",
        )

    hits = [hit["_source"] for hit in data.get("hits", {}).get("hits", [])]
    total_value = data.get("hits", {}).get("total", {})
    total = (
        total_value.get("value", 0) if isinstance(total_value, dict) else total_value
    )

    return JSONResponse(content={"hits": hits, "total": total})


# --- Route registration ---


def register_routes(api: "RESTfulAPI") -> None:
    router = api._router
    auth = api._auth_service
    is_auth = api.is_authenticated()

    router.add_api_route("/status", get_status, methods=["GET"])
    router.add_api_route("/v1/address", get_address, methods=["GET"])
    router.add_api_route("/token", login_for_access_token, methods=["POST"])
    router.add_api_route("/v1/cluster/auth", is_cluster_authenticated, methods=["GET"])
    router.add_api_route("/v1/cluster/ui_config", get_ui_config, methods=["GET"])

    router.add_api_route(
        "/v1/cluster/info",
        get_cluster_device_info,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cluster/version",
        get_cluster_version,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cluster/devices",
        get_devices_count,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:list"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/workers",
        get_workers_info,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/supervisor",
        get_supervisor_info,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/clusters",
        abort_cluster,
        methods=["DELETE"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/cache/models",
        list_cached_models,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["cache:list"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cache/models/files",
        list_model_files,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["cache:list"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cache/models",
        confirm_and_remove_model,
        methods=["DELETE"],
        dependencies=([Security(auth, scopes=["cache:delete"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/virtualenvs",
        list_virtual_envs,
        methods=["GET"],
        dependencies=(
            [Security(auth, scopes=["virtualenv:list"])] if is_auth else None
        ),
    )
    router.add_api_route(
        "/v1/virtualenvs",
        remove_virtual_env,
        methods=["DELETE"],
        dependencies=(
            [Security(auth, scopes=["virtualenv:delete"])] if is_auth else None
        ),
    )

    router.add_api_route(
        "/v1/requests/{request_id}/progress",
        get_progress,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["models:read"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/cluster/logs",
        search_logs,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
