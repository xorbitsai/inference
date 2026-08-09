"""Admin / cluster / infrastructure route registration and handlers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

import aiohttp
from fastapi import Body, Depends, HTTPException, Query, Request, Security
from pydantic import BaseModel

from ... import __version__
from ..dependencies import get_api
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
        # keep the response keys the versioneer-era API exposed; the build
        # backend records the full 40-character SHA in _commit.py, with the
        # setuptools-scm short node as a fallback for artifacts built without
        # git metadata
        try:
            from ..._commit import full_revisionid
        except ImportError:
            try:
                from ..._version import commit_id
            except ImportError:
                commit_id = None
            full_revisionid = commit_id.lstrip("g") if commit_id else None
        data = {
            "version": __version__,
            "full-revisionid": full_revisionid,
        }
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


async def get_ui_config(request: Request) -> JSONResponse:
    store = request.app.state.monitor_config_store
    mon = store.get_all()
    dashboards = store.get_dashboards()

    return JSONResponse(
        content={
            "grafana_url": mon["grafana_url"],
            "grafana_datasource": mon["grafana_datasource"],
            "grafana_alert_datasource": mon["grafana_alert_datasource"],
            "grafana_dashboard_uid": dashboards.get("overview", "xinference-overview"),
            "grafana_dashboards": dashboards,
            "grafana_dashboards_configured": store.get_configured_dashboard_keys(),
            "cluster_name": mon["cluster_name"],
            "es_enabled": bool(os.environ.get("XINFERENCE_ES_URL", "")),
            "auth_advanced": os.environ.get("XINFERENCE_AUTH_ADVANCED", "true").lower()
            not in ("0", "false", "no"),
            "oidc_enabled": os.environ.get("XINFERENCE_OIDC_ENABLED", "").lower()
            in ("1", "true", "yes"),
        }
    )


class MonitorConfigUpdate(BaseModel):
    grafana_url: Optional[str] = None
    grafana_datasource: Optional[str] = None
    grafana_alert_datasource: Optional[str] = None
    cluster_name: Optional[str] = None
    grafana_dashboards: Optional[Dict[str, str]] = None


class CheckGrafanaRequest(BaseModel):
    grafana_url: str


async def get_monitor_config(request: Request) -> JSONResponse:
    store = request.app.state.monitor_config_store
    all_cfg = store.get_all()
    sources = store.get_sources()
    dashboards = store.get_dashboards()

    return JSONResponse(
        content={
            "grafana_url": all_cfg["grafana_url"],
            "grafana_datasource": all_cfg["grafana_datasource"],
            "grafana_alert_datasource": all_cfg["grafana_alert_datasource"],
            "cluster_name": all_cfg["cluster_name"],
            "grafana_dashboards": dashboards,
            "grafana_dashboards_configured": store.get_configured_dashboard_keys(),
            "sources": sources,
        }
    )


async def update_monitor_config(
    request: Request,
    body: MonitorConfigUpdate = Body(...),
) -> JSONResponse:
    store = request.app.state.monitor_config_store

    username = ""
    advanced_auth = getattr(request.app.state, "advanced_auth", None)
    if advanced_auth:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                payload = advanced_auth.verify_access_token(auth_header[7:])
                username = payload.get("sub", "")
            except Exception:
                pass

    data = body.model_dump(exclude_none=True)
    updates = {}
    for field in (
        "grafana_url",
        "grafana_datasource",
        "grafana_alert_datasource",
        "cluster_name",
    ):
        if field in data:
            updates[field] = data[field]
    if "grafana_dashboards" in data:
        for dashboard_key, uid in data["grafana_dashboards"].items():
            updates[f"dashboard_{dashboard_key}"] = uid

    store.update(updates, username=username)
    return JSONResponse(content={"status": "ok"})


async def check_grafana(
    request: Request,
    body: CheckGrafanaRequest = Body(...),
) -> JSONResponse:
    grafana_url = body.grafana_url.rstrip("/")
    if not grafana_url:
        return JSONResponse(
            content={"ok": False, "error": "Grafana URL is empty"},
            status_code=400,
        )

    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{grafana_url}/api/health")
            resp.raise_for_status()
            return JSONResponse(content={"ok": True, "body": resp.json()})
    except Exception as e:
        return JSONResponse(
            content={"ok": False, "error": str(e)},
            status_code=200,
        )


async def reset_monitor_config(request: Request) -> JSONResponse:
    store = request.app.state.monitor_config_store
    store.reset()
    return JSONResponse(content={"status": "ok"})


_FIELD_NAME_RE = re.compile(r"^[a-zA-Z0-9_.@]+$")
_TEXT_FIELDS = {"message"}
_ES_PIT_KEEP_ALIVE = "1m"
_ES_SEARCH_AFTER_BATCH_SIZE = 5000
_ES_MAX_SEARCH_AFTER_REQUESTS = 10


def _get_es_total(data: dict[str, Any]) -> int:
    total_value = data.get("hits", {}).get("total", 0)
    return (
        int(total_value.get("value", 0))
        if isinstance(total_value, dict)
        else int(total_value)
    )


async def _search_es_page(
    session: aiohttp.ClientSession,
    *,
    es_url: str,
    es_index: str,
    headers: dict[str, str],
    query: dict[str, Any],
    page_from: int,
    size: int,
    source: Optional[dict[str, Any]] = None,
    error_context: str = "ES",
) -> tuple[list[dict[str, Any]], int]:
    """Fetch an arbitrary page with PIT-backed ``search_after`` queries.

    Elasticsearch limits ``from + size`` pagination to 10,000 hits by
    default.  A PIT supplies the stable ``_shard_doc`` tiebreaker required to
    walk past that boundary safely.  For pages closer to the end of the result
    set, walking in ascending order bounds the work by the nearer edge.
    """
    base_url = es_url.rstrip("/")
    pit_id = ""

    async def _read_response(
        resp: aiohttp.ClientResponse, operation: str
    ) -> dict[str, Any]:
        if resp.status != 200:
            text = await resp.text()
            logger.error(
                "%s %s failed: status=%d body=%s",
                error_context,
                operation,
                resp.status,
                text[:500],
            )
            raise HTTPException(status_code=502, detail="Elasticsearch query failed")
        return await resp.json()

    async def _search(
        body: dict[str, Any], *, include_source: bool = True
    ) -> dict[str, Any]:
        nonlocal pit_id
        request_body = {
            **body,
            "pit": {"id": pit_id, "keep_alive": _ES_PIT_KEEP_ALIVE},
        }
        if include_source and source is not None:
            request_body["_source"] = source
        async with session.post(
            f"{base_url}/_search", json=request_body, headers=headers
        ) as resp:
            data = await _read_response(resp, "search")
        pit_id = data.get("pit_id") or pit_id
        return data

    try:
        async with session.post(
            f"{base_url}/{es_index}/_pit",
            params={"keep_alive": _ES_PIT_KEEP_ALIVE},
            headers=headers,
        ) as resp:
            pit_data = await _read_response(resp, "PIT open")
        pit_id = pit_data.get("id", "")
        if not pit_id:
            logger.error("%s PIT open returned no id", error_context)
            raise HTTPException(status_code=502, detail="Elasticsearch query failed")

        count_data = await _search(
            {"query": query, "size": 0, "track_total_hits": True, "_source": False},
            include_source=False,
        )
        total = _get_es_total(count_data)
        if page_from >= total:
            return [], total

        result_size = min(size, total - page_from)
        reverse_offset = total - (page_from + result_size)
        ascending = reverse_offset < page_from
        remaining = reverse_offset if ascending else page_from
        max_traversal_hits = _ES_SEARCH_AFTER_BATCH_SIZE * _ES_MAX_SEARCH_AFTER_REQUESTS
        if remaining > max_traversal_hits:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Requested page is too far from either end of the result set; "
                    "narrow the filters or time range"
                ),
            )
        order = "asc" if ascending else "desc"
        sort = [
            {
                "@timestamp": {
                    "order": order,
                    "format": "strict_date_optional_time_nanos",
                    "numeric_type": "date_nanos",
                }
            },
            {"_shard_doc": order},
        ]
        search_after: Optional[list[Any]] = None

        while remaining > 0:
            batch_size = min(remaining, _ES_SEARCH_AFTER_BATCH_SIZE)
            batch_body: dict[str, Any] = {
                "query": query,
                "size": batch_size,
                "sort": sort,
                "track_total_hits": False,
                "_source": False,
            }
            if search_after is not None:
                batch_body["search_after"] = search_after
            batch_data = await _search(batch_body, include_source=False)
            batch_hits = batch_data.get("hits", {}).get("hits", [])
            if not batch_hits:
                return [], total
            search_after = batch_hits[-1].get("sort")
            if not search_after:
                logger.error("%s search hit returned no sort values", error_context)
                raise HTTPException(
                    status_code=502, detail="Elasticsearch query failed"
                )
            remaining -= len(batch_hits)
            if len(batch_hits) < batch_size:
                return [], total

        page_body: dict[str, Any] = {
            "query": query,
            "size": result_size,
            "sort": sort,
            "track_total_hits": False,
        }
        if search_after is not None:
            page_body["search_after"] = search_after
        page_data = await _search(page_body)
        page_hits = page_data.get("hits", {}).get("hits", [])
        if ascending:
            page_hits.reverse()
        return [hit["_source"] for hit in page_hits], total
    finally:
        if pit_id:
            try:
                async with session.delete(
                    f"{base_url}/_pit", json={"id": pit_id}, headers=headers
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning(
                            "%s PIT close failed: status=%d body=%s",
                            error_context,
                            resp.status,
                            text[:500],
                        )
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning("%s PIT close failed: %s", error_context, e)


async def search_logs(
    q: str = "",
    level: str = "",
    module: str = "",
    node: str = "",
    log_type: str = "",
    filters: list[str] = Query(
        [], description="Field filters, e.g. filters=+node:val1&filters=-level:val2"
    ),
    time_from: str = "now-1h",
    time_to: str = "now",
    size: int = 200,
    page_from: int = 0,
    node_field: str = "node",
) -> JSONResponse:
    es_url = os.environ.get("XINFERENCE_ES_URL", "")
    if not es_url:
        raise HTTPException(status_code=503, detail="Elasticsearch is not configured")

    es_index = os.environ.get("XINFERENCE_ES_INDEX", "xinference-logs-*")
    es_auth = os.environ.get("XINFERENCE_ES_AUTH", "")

    size = max(1, min(size, 500))
    page_from = max(0, page_from)
    time_from, time_to = _freeze_es_time_bounds(time_from, time_to)
    if node_field not in ("node", "node.keyword"):
        node_field = "node"

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
        (node_field, node),
        ("log_type", log_type),
    ]:
        if value:
            terms = [v.strip() for v in value.split(",") if v.strip()]
            if terms:
                filter_clauses.append({"terms": {field: terms}})

    must_not: list[dict[str, Any]] = []
    plus_filters: dict[str, list[str]] = {}
    for token in filters:
        token = token.strip()
        if len(token) < 3 or token[0] not in ("+", "-"):
            continue
        sep = token.find(":", 1)
        if sep < 0:
            continue
        op = token[0]
        field_name = token[1:sep]
        field_value = token[sep + 1 :]
        if field_name == "node":
            field_name = node_field
        if not _FIELD_NAME_RE.match(field_name) or not field_value:
            continue
        if op == "+":
            plus_filters.setdefault(field_name, []).append(field_value)
        else:
            if field_name in _TEXT_FIELDS:
                must_not.append({"match_phrase": {field_name: field_value}})
            else:
                must_not.append({"term": {field_name: field_value}})

    for field_name, values in plus_filters.items():
        if field_name in _TEXT_FIELDS:
            filter_clauses.append(
                {
                    "bool": {
                        "should": [{"match_phrase": {field_name: v}} for v in values]
                    }
                }
            )
        else:
            filter_clauses.append({"terms": {field_name: values}})

    query: dict[str, Any] = {
        "bool": {"must": must, "filter": filter_clauses, "must_not": must_not}
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

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout, auth=auth) as session:
            hits, total = await _search_es_page(
                session,
                es_url=es_url,
                es_index=es_index,
                headers=headers,
                query=query,
                page_from=page_from,
                size=size,
                source={"excludes": ["@version"]},
                error_context="ES log query",
            )
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error("ES connection error or timeout: %s", e)
        raise HTTPException(
            status_code=502,
            detail="Failed to connect to Elasticsearch or query timed out",
        )

    return JSONResponse(content={"hits": hits, "total": total})


async def list_log_nodes() -> JSONResponse:
    es_url = os.environ.get("XINFERENCE_ES_URL", "")
    if not es_url:
        raise HTTPException(status_code=503, detail="Elasticsearch is not configured")

    es_index = os.environ.get("XINFERENCE_ES_INDEX", "xinference-logs-*")
    es_auth = os.environ.get("XINFERENCE_ES_AUTH", "")

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

    async def _aggregate(field: str) -> Optional[list[dict[str, Any]]]:
        body = {"size": 0, "aggs": {"nodes": {"terms": {"field": field, "size": 200}}}}
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout, auth=auth) as session:
            async with session.post(url, json=body, headers=headers) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
        return data.get("aggregations", {}).get("nodes", {}).get("buckets", [])

    try:
        # "node" is usually a keyword field; fall back to "node.keyword" if the
        # mapping is text (terms aggregation requires a keyword/fielddata field).
        buckets = await _aggregate("node")
        node_field = "node"
        if buckets is None:
            buckets = await _aggregate("node.keyword")
            node_field = "node.keyword"
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error("ES connection error or timeout: %s", e)
        raise HTTPException(
            status_code=502,
            detail="Failed to connect to Elasticsearch or query timed out",
        )

    if buckets is None:
        logger.error("ES node aggregation failed for both 'node' and 'node.keyword'")
        raise HTTPException(status_code=502, detail="Elasticsearch query failed")

    nodes = [b["key"] for b in buckets if b.get("key")]
    return JSONResponse(content={"nodes": nodes, "node_field": node_field})


async def search_logs_context(
    timestamp: str = "",
    size: int = 5,
    node: str = "",
    node_field: str = "node",
) -> JSONResponse:
    if not timestamp:
        raise HTTPException(status_code=400, detail="timestamp is required")

    es_url = os.environ.get("XINFERENCE_ES_URL", "")
    if not es_url:
        raise HTTPException(status_code=503, detail="Elasticsearch is not configured")

    es_index = os.environ.get("XINFERENCE_ES_INDEX", "xinference-logs-*")
    es_auth = os.environ.get("XINFERENCE_ES_AUTH", "")

    size = max(1, min(size, 50))
    if node_field not in ("node", "node.keyword"):
        node_field = "node"

    node_filter: list[dict[str, Any]] = []
    if node:
        node_filter = [{"term": {node_field: node}}]

    older_body: dict[str, Any] = {
        "query": {
            "bool": {
                "filter": [
                    {"range": {"@timestamp": {"lt": timestamp}}},
                    *node_filter,
                ]
            }
        },
        "sort": [{"@timestamp": "desc"}],
        "size": size + 1,
        "_source": {"excludes": ["@version"]},
    }

    newer_body: dict[str, Any] = {
        "query": {
            "bool": {
                "filter": [
                    {"range": {"@timestamp": {"gt": timestamp}}},
                    *node_filter,
                ]
            }
        },
        "sort": [{"@timestamp": "asc"}],
        "size": size + 1,
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

            async def fetch(body: dict, name: str) -> dict:
                async with session.post(url, json=body, headers=headers) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(
                            "ES context %s query failed: status=%d body=%s",
                            name,
                            resp.status,
                            text[:500],
                        )
                        raise HTTPException(
                            status_code=502,
                            detail="Elasticsearch query failed",
                        )
                    return await resp.json()

            older_data, newer_data = await asyncio.gather(
                fetch(older_body, "older"),
                fetch(newer_body, "newer"),
            )
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error("ES connection error or timeout: %s", e)
        raise HTTPException(
            status_code=502,
            detail="Failed to connect to Elasticsearch or query timed out",
        )

    older_hits_raw = [
        hit["_source"] for hit in older_data.get("hits", {}).get("hits", [])
    ]
    newer_hits_raw = [
        hit["_source"] for hit in newer_data.get("hits", {}).get("hits", [])
    ]

    has_more_older = len(older_hits_raw) > size
    has_more_newer = len(newer_hits_raw) > size

    older_hits = older_hits_raw[:size]
    newer_hits = newer_hits_raw[:size]

    return JSONResponse(
        content={
            "older": older_hits,
            "newer": newer_hits,
            "anchor_timestamp": timestamp,
            "has_more_older": has_more_older,
            "has_more_newer": has_more_newer,
        }
    )


# --- Route registration ---


def _parse_relative_time(
    expr: str, *, now: Optional[datetime] = None
) -> Optional[datetime]:
    """Parse ES-style relative time, epoch milliseconds, or ISO timestamp."""
    reference_time = now or datetime.now(timezone.utc)

    if expr == "now":
        return reference_time
    m = re.match(r"now-(\d+)([mhdw])", expr)
    if m:
        val, unit = int(m.group(1)), m.group(2)
        delta = {
            "m": timedelta(minutes=val),
            "h": timedelta(hours=val),
            "d": timedelta(days=val),
            "w": timedelta(weeks=val),
        }.get(unit)
        if delta is None:
            return None
        return reference_time - delta
    # Epoch milliseconds (numeric string like "1716854400000")
    if expr.isdigit():
        return datetime.fromtimestamp(int(expr) / 1000, tz=timezone.utc)
    # ISO 8601 timestamp (e.g. "2026-05-28T03:00:00.000Z")
    try:
        return datetime.fromisoformat(expr.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        pass
    return None


def _freeze_es_time_bounds(
    time_from: str,
    time_to: str,
    *,
    now: Optional[datetime] = None,
) -> tuple[str, str]:
    """Resolve supported relative date math against one shared instant."""
    reference_time = now or datetime.now(timezone.utc)

    def _freeze(value: str) -> str:
        if value != "now" and re.fullmatch(r"now-\d+[mhdw]", value) is None:
            return value
        parsed = _parse_relative_time(value, now=reference_time)
        return parsed.isoformat().replace("+00:00", "Z") if parsed else value

    return _freeze(time_from), _freeze(time_to)


def _escape_es_wildcard(value: str) -> str:
    """Escape characters that are special to an Elasticsearch wildcard query."""
    return value.replace("\\", "\\\\").replace("*", "\\*").replace("?", "\\?")


def _es_substring_clause(field_name: str, value: str) -> dict[str, Any]:
    """Build a substring query compatible with common ES string mappings."""
    pattern = f"*{_escape_es_wildcard(value)}*"
    return {
        "bool": {
            "should": [
                {"wildcard": {field: {"value": pattern, "case_insensitive": True}}}
                for field in (field_name, f"{field_name}.keyword")
            ],
            "minimum_should_match": 1,
        }
    }


_AUDIT_TEXT_FILTER_FIELDS = (
    "user",
    "api_key_name",
    "model_id",
    "model_name",
    "client_ip",
)
_AUDIT_FILTER_OPTION_LIMIT = 500


def _add_bounded_audit_filter_option(
    options: list[tuple[str, str]], seen: set[str], value: Any
) -> None:
    if value is None:
        return
    text = str(value)
    if not text or text in seen:
        return

    item = (text.casefold(), text)
    position = bisect_left(options, item)
    if len(options) >= _AUDIT_FILTER_OPTION_LIMIT and position >= len(options):
        return

    options.insert(position, item)
    seen.add(text)
    if len(options) > _AUDIT_FILTER_OPTION_LIMIT:
        _, removed = options.pop()
        seen.remove(removed)


def _audit_filter_aggregation_body(
    time_from: str,
    time_to: str,
    fields: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    fields = fields or {
        field_name: field_name for field_name in _AUDIT_TEXT_FILTER_FIELDS
    }
    return {
        "size": 0,
        "query": {"range": {"@timestamp": {"gte": time_from, "lte": time_to}}},
        "aggs": {
            response_field: {
                "terms": {
                    "field": es_field,
                    "size": _AUDIT_FILTER_OPTION_LIMIT,
                }
            }
            for response_field, es_field in fields.items()
        },
    }


def _aggregatable_field_indices(
    field_caps: dict[str, Any], field_name: str, all_indices: set[str]
) -> set[str]:
    result: set[str] = set()
    capabilities = field_caps.get("fields", {}).get(field_name, {})
    for capability in capabilities.values():
        capability_indices = set(capability.get("indices") or all_indices)
        non_aggregatable = set(capability.get("non_aggregatable_indices") or [])
        if capability.get("aggregatable") or non_aggregatable:
            result.update(capability_indices - non_aggregatable)
    return result


def _audit_filter_field_groups(
    field_caps: dict[str, Any],
) -> list[tuple[list[str], dict[str, str]]]:
    all_indices = {str(index_name) for index_name in field_caps.get("indices", [])}
    index_fields: dict[str, dict[str, str]] = {
        index_name: {} for index_name in all_indices
    }

    for field_name in _AUDIT_TEXT_FILTER_FIELDS:
        direct_indices = _aggregatable_field_indices(
            field_caps, field_name, all_indices
        )
        keyword_field = f"{field_name}.keyword"
        keyword_indices = _aggregatable_field_indices(
            field_caps, keyword_field, all_indices
        )
        for index_name in all_indices:
            if index_name in direct_indices:
                index_fields[index_name][field_name] = field_name
            elif index_name in keyword_indices:
                index_fields[index_name][field_name] = keyword_field

    groups: dict[tuple[tuple[str, str], ...], list[str]] = {}
    for index_name, fields in index_fields.items():
        if not fields:
            continue
        signature = tuple(sorted(fields.items()))
        groups.setdefault(signature, []).append(index_name)

    return [
        (sorted(indices), dict(signature))
        for signature, indices in sorted(groups.items())
    ]


def _audit_entry_in_time_range(
    entry: dict[str, Any], t_from: Optional[datetime], t_to: Optional[datetime]
) -> bool:
    if not t_from and not t_to:
        return True
    try:
        timestamp = datetime.fromisoformat(
            str(entry.get("@timestamp", "")).replace("Z", "+00:00")
        )
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return not ((t_from and timestamp < t_from) or (t_to and timestamp > t_to))
    except (ValueError, TypeError):
        return False


def _match_substring(stored: Any, needle: str) -> bool:
    """Case-insensitive substring match used by audit free-text filters.

    An empty ``needle`` means the filter is not active and matches everything.
    """
    if not needle:
        return True
    if not isinstance(stored, str):
        return False
    return needle.lower() in stored.lower()


def _match_enum(stored: Any, allowed: set) -> bool:
    """Case-insensitive exact match against a set of allowed values.

    An empty ``allowed`` set means the filter is not active.
    """
    if not allowed:
        return True
    if not isinstance(stored, str):
        return False
    return stored.lower() in allowed


async def _search_audit_from_file(
    *,
    time_from: str,
    time_to: str,
    user: str,
    api_key_name: str,
    model_id: str,
    model_name: str,
    model_type: str,
    category: str,
    auth_type: str,
    status: str,
    client_ip: str,
    page_from: int,
    size: int,
) -> JSONResponse:
    """Fallback: search audit events from local audit.log file."""
    from ...constants import XINFERENCE_LOG_DIR

    audit_path = os.path.join(XINFERENCE_LOG_DIR, "audit.log")
    if not os.path.exists(audit_path):
        return JSONResponse(content={"hits": [], "total": 0})

    t_from = _parse_relative_time(time_from)
    t_to = _parse_relative_time(time_to)

    status_set = (
        {v.strip().lower() for v in status.split(",") if v.strip()} if status else set()
    )
    category_set = (
        {v.strip().lower() for v in category.split(",") if v.strip()}
        if category
        else set()
    )
    model_type_set = (
        {v.strip().lower() for v in model_type.split(",") if v.strip()}
        if model_type
        else set()
    )
    auth_type_set = (
        {v.strip().lower() for v in auth_type.split(",") if v.strip()}
        if auth_type
        else set()
    )

    results: list[dict] = []
    try:
        with open(audit_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                # A line may be valid JSON yet not an object (e.g. `null`,
                # `[1,2]`); `entry.get(...)` below would raise AttributeError.
                if not isinstance(entry, dict):
                    continue

                ts_str = entry.get("@timestamp", "")
                if t_from or t_to:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        continue
                    if t_from and ts < t_from:
                        continue
                    if t_to and ts > t_to:
                        continue

                if not all(
                    _match_substring(entry.get(field), needle)
                    for field, needle in (
                        ("user", user),
                        ("api_key_name", api_key_name),
                        ("model_id", model_id),
                        ("model_name", model_name),
                        ("client_ip", client_ip),
                    )
                    if needle
                ):
                    continue
                if not all(
                    _match_enum(entry.get(field), allowed)
                    for field, allowed in (
                        ("status", status_set),
                        ("category", category_set),
                        ("model_type", model_type_set),
                        ("auth_type", auth_type_set),
                    )
                    if allowed
                ):
                    continue

                results.append(entry)
    except OSError:
        return JSONResponse(content={"hits": [], "total": 0})

    results.sort(key=lambda x: x.get("@timestamp", ""), reverse=True)
    total = len(results)
    hits = results[page_from : page_from + size]
    return JSONResponse(content={"hits": hits, "total": total})


async def _list_audit_filter_options_from_file(
    *, time_from: str, time_to: str
) -> JSONResponse:
    from ...constants import XINFERENCE_LOG_DIR

    audit_path = os.path.join(XINFERENCE_LOG_DIR, "audit.log")
    t_from = _parse_relative_time(time_from)
    t_to = _parse_relative_time(time_to)

    def _read_options() -> dict[str, list[str]]:
        options: dict[str, list[tuple[str, str]]] = {
            field_name: [] for field_name in _AUDIT_TEXT_FILTER_FIELDS
        }
        seen: dict[str, set[str]] = {
            field_name: set() for field_name in _AUDIT_TEXT_FILTER_FIELDS
        }
        try:
            with open(audit_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if not isinstance(entry, dict):
                        continue
                    if not _audit_entry_in_time_range(entry, t_from, t_to):
                        continue
                    for field_name in _AUDIT_TEXT_FILTER_FIELDS:
                        _add_bounded_audit_filter_option(
                            options[field_name], seen[field_name], entry.get(field_name)
                        )
        except OSError:
            return {key: [] for key in options}

        return {key: [value for _, value in values] for key, values in options.items()}

    content = await asyncio.to_thread(_read_options)
    return JSONResponse(content=content)


async def list_audit_filter_options(
    time_from: str = "now-1h", time_to: str = "now"
) -> JSONResponse:
    es_url = os.environ.get("XINFERENCE_ES_URL", "")
    if not es_url:
        return await _list_audit_filter_options_from_file(
            time_from=time_from, time_to=time_to
        )

    from ...constants import XINFERENCE_AUDIT_ES_INDEX

    headers = {"Content-Type": "application/json"}
    auth = None
    es_auth = os.environ.get("XINFERENCE_ES_AUTH", "")
    if es_auth:
        if es_auth.startswith("ApiKey "):
            headers["Authorization"] = es_auth
        else:
            parts = es_auth.split(":", 1)
            if len(parts) == 2:
                auth = aiohttp.BasicAuth(parts[0], parts[1])

    es_base_url = es_url.rstrip("/")
    field_names = ",".join(
        field_name
        for response_field in _AUDIT_TEXT_FILTER_FIELDS
        for field_name in (response_field, f"{response_field}.keyword")
    )
    field_caps_url = (
        f"{es_base_url}/{XINFERENCE_AUDIT_ES_INDEX}/_field_caps"
        f"?fields={field_names}&include_unmapped=true"
    )
    option_values: dict[str, list[tuple[str, str]]] = {
        field_name: [] for field_name in _AUDIT_TEXT_FILTER_FIELDS
    }
    seen_values: dict[str, set[str]] = {
        field_name: set() for field_name in _AUDIT_TEXT_FILTER_FIELDS
    }
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout, auth=auth) as session:

            async def fetch(
                url: str, request_body: Optional[dict[str, Any]] = None
            ) -> tuple[int, str, dict[str, Any]]:
                request_kwargs: dict[str, Any] = {"headers": headers}
                if request_body is not None:
                    request_kwargs["json"] = request_body
                async with session.post(url, **request_kwargs) as resp:
                    if resp.status != 200:
                        return resp.status, await resp.text(), {}
                    return resp.status, "", await resp.json()

            status, response_text, field_caps = await fetch(field_caps_url)
            if status != 200:
                logger.error(
                    "ES audit field capabilities query failed: status=%d body=%s",
                    status,
                    response_text[:500],
                )
                raise HTTPException(
                    status_code=502, detail="Elasticsearch query failed"
                )

            for indices, fields in _audit_filter_field_groups(field_caps):
                search_url = f"{es_base_url}/{','.join(indices)}/_search"
                body = _audit_filter_aggregation_body(time_from, time_to, fields=fields)
                status, response_text, data = await fetch(search_url, body)
                if status != 200:
                    logger.error(
                        "ES audit filter aggregation failed: status=%d body=%s",
                        status,
                        response_text[:500],
                    )
                    raise HTTPException(
                        status_code=502, detail="Elasticsearch query failed"
                    )

                aggregations = data.get("aggregations", {})
                for field_name in _AUDIT_TEXT_FILTER_FIELDS:
                    for bucket in aggregations.get(field_name, {}).get("buckets", []):
                        _add_bounded_audit_filter_option(
                            option_values[field_name],
                            seen_values[field_name],
                            bucket.get("key"),
                        )
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
        logger.error("ES connection error, timeout, or invalid response: %s", e)
        raise HTTPException(status_code=502, detail="Audit service unavailable")

    return JSONResponse(
        content={
            field_name: [value for _, value in option_values[field_name]]
            for field_name in option_values
        }
    )


async def search_audit_logs(
    time_from: str = "now-1h",
    time_to: str = "now",
    user: str = "",
    api_key_name: str = "",
    model_id: str = "",
    model_name: str = "",
    model_type: str = "",
    category: str = "",
    auth_type: str = "",
    status: str = "",
    client_ip: str = "",
    page_from: int = 0,
    size: int = 50,
) -> JSONResponse:
    es_url = os.environ.get("XINFERENCE_ES_URL", "")
    if not es_url:
        return await _search_audit_from_file(
            time_from=time_from,
            time_to=time_to,
            user=user,
            api_key_name=api_key_name,
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            category=category,
            auth_type=auth_type,
            status=status,
            client_ip=client_ip,
            page_from=page_from,
            size=size,
        )

    from ...constants import XINFERENCE_AUDIT_ES_INDEX

    es_index = XINFERENCE_AUDIT_ES_INDEX
    es_auth = os.environ.get("XINFERENCE_ES_AUTH", "")

    size = max(1, min(size, 500))
    page_from = max(0, page_from)
    time_from, time_to = _freeze_es_time_bounds(time_from, time_to)

    must: list[dict[str, Any]] = []
    filter_clauses: list[dict[str, Any]] = [
        {"range": {"@timestamp": {"gte": time_from, "lte": time_to}}}
    ]

    for field_name, value in [
        ("user", user),
        ("api_key_name", api_key_name),
        ("model_id", model_id),
        ("model_name", model_name),
        ("client_ip", client_ip),
    ]:
        if value:
            # Substring, case-insensitive, to match the file-mode semantics.
            # Dynamic mapping commonly creates `text` + `.keyword`, while an
            # index template may map the field directly as `keyword` or
            # `wildcard`. Query both names so either mapping works. An unmapped
            # alternative simply contributes no match.
            #
            # NOTE: a leading wildcard cannot use the index and forces a scan of
            # all terms in the segment. That is acceptable for typical audit
            # volumes, but on a large index deployments should install an index
            # template mapping these fields directly to the `wildcard` type
            # (ES >= 7.9), which is covered by the first alternative.
            filter_clauses.append(_es_substring_clause(field_name, value))

    for field_name, value in [
        ("model_type", model_type),
        ("category", category),
        ("auth_type", auth_type),
        ("status", status),
    ]:
        if value:
            terms = [v.strip().lower() for v in value.split(",") if v.strip()]
            if len(terms) == 1:
                filter_clauses.append({"term": {field_name: terms[0]}})
            else:
                filter_clauses.append({"terms": {field_name: terms}})

    query: dict[str, Any] = {"bool": {"must": must, "filter": filter_clauses}}

    headers = {"Content-Type": "application/json"}
    auth = None
    if es_auth:
        if es_auth.startswith("ApiKey "):
            headers["Authorization"] = es_auth
        else:
            parts = es_auth.split(":", 1)
            if len(parts) == 2:
                auth = aiohttp.BasicAuth(parts[0], parts[1])

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout, auth=auth) as session:
            hits, total = await _search_es_page(
                session,
                es_url=es_url,
                es_index=es_index,
                headers=headers,
                query=query,
                page_from=page_from,
                size=size,
                error_context="ES audit query",
            )
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error("ES connection error or timeout: %s", e)
        raise HTTPException(
            status_code=502,
            detail="Audit service unavailable",
        )

    return JSONResponse(content={"hits": hits, "total": total})


# --- Route registration (original) ---


def register_routes(api: "RESTfulAPI") -> None:
    router = api._router
    auth = api._auth_service
    is_auth = api.is_authenticated()

    router.add_api_route("/status", get_status, methods=["GET"])
    router.add_api_route("/v1/address", get_address, methods=["GET"])
    router.add_api_route("/v1/cluster/auth", is_cluster_authenticated, methods=["GET"])
    router.add_api_route("/v1/cluster/ui_config", get_ui_config, methods=["GET"])

    router.add_api_route(
        "/v1/cluster/monitor_config",
        get_monitor_config,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cluster/monitor_config",
        update_monitor_config,
        methods=["PUT"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cluster/monitor_config/check-grafana",
        check_grafana,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
    router.add_api_route(
        "/v1/cluster/monitor_config/reset",
        reset_monitor_config,
        methods=["POST"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )

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
        dependencies=([Security(auth, scopes=["logs:list"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/cluster/logs/context",
        search_logs_context,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["logs:list"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/cluster/logs/nodes",
        list_log_nodes,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["logs:list"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/audit/filter-options",
        list_audit_filter_options,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/audit/search",
        search_audit_logs,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
