"""Admin / cluster / infrastructure route registration and handlers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

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
    result = api._auth_service.generate_token_for_user(  # type: ignore[union-attr]
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
            "auth_advanced": os.environ.get("XINFERENCE_AUTH_ADVANCED", "").lower()
            in ("1", "true", "yes"),
            "oidc_enabled": os.environ.get("XINFERENCE_OIDC_ENABLED", "").lower()
            in ("1", "true", "yes"),
        }
    )


_FIELD_NAME_RE = re.compile(r"^[a-zA-Z0-9_.@]+$")
_TEXT_FIELDS = {"message"}


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
    page_from = max(0, min(page_from, 10000 - size))
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

    body: dict[str, Any] = {
        "query": {
            "bool": {"must": must, "filter": filter_clauses, "must_not": must_not}
        },
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


def _parse_relative_time(expr: str) -> Optional[datetime]:
    """Parse ES-style relative time, epoch milliseconds, or ISO timestamp."""
    import re

    if expr == "now":
        return datetime.now(timezone.utc)
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
        return datetime.now(timezone.utc) - delta
    # Epoch milliseconds (numeric string like "1716854400000")
    if expr.isdigit():
        return datetime.fromtimestamp(int(expr) / 1000, tz=timezone.utc)
    # ISO 8601 timestamp (e.g. "2026-05-28T03:00:00.000Z")
    try:
        return datetime.fromisoformat(expr.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        pass
    return None


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

                if user and entry.get("user") != user:
                    continue
                if api_key_name and entry.get("api_key_name") != api_key_name:
                    continue
                if model_id and entry.get("model_id") != model_id:
                    continue
                if model_name and entry.get("model_name") != model_name:
                    continue
                if client_ip and entry.get("client_ip") != client_ip:
                    continue
                if status_set and entry.get("status", "").lower() not in status_set:
                    continue
                if (
                    category_set
                    and entry.get("category", "").lower() not in category_set
                ):
                    continue
                if (
                    model_type_set
                    and entry.get("model_type", "").lower() not in model_type_set
                ):
                    continue
                if (
                    auth_type_set
                    and entry.get("auth_type", "").lower() not in auth_type_set
                ):
                    continue

                results.append(entry)
    except OSError:
        return JSONResponse(content={"hits": [], "total": 0})

    results.sort(key=lambda x: x.get("@timestamp", ""), reverse=True)
    total = len(results)
    hits = results[page_from : page_from + size]
    return JSONResponse(content={"hits": hits, "total": total})


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
    page_from = max(0, min(page_from, 10000 - size))

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
            filter_clauses.append({"term": {field_name: value}})

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

    body: dict[str, Any] = {
        "query": {"bool": {"must": must, "filter": filter_clauses}},
        "sort": [{"@timestamp": "desc"}],
        "from": page_from,
        "size": size,
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
                        "ES audit query failed: status=%d body=%s",
                        resp.status,
                        text[:500],
                    )
                    raise HTTPException(
                        status_code=502, detail="Elasticsearch query failed"
                    )
                data = await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error("ES connection error or timeout: %s", e)
        raise HTTPException(
            status_code=502,
            detail="Audit service unavailable",
        )

    hits = [hit["_source"] for hit in data.get("hits", {}).get("hits", [])]
    total_value = data.get("hits", {}).get("total", {})
    total = (
        total_value.get("value", 0) if isinstance(total_value, dict) else total_value
    )

    return JSONResponse(content={"hits": hits, "total": total})


# --- Route registration (original) ---


def register_routes(api: "RESTfulAPI") -> None:
    router = api._router
    auth = api._auth_service
    is_auth = api.is_authenticated()
    is_advanced = api._advanced_auth_service is not None

    router.add_api_route("/status", get_status, methods=["GET"])
    router.add_api_route("/v1/address", get_address, methods=["GET"])
    if not is_advanced:
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

    router.add_api_route(
        "/v1/cluster/logs/context",
        search_logs_context,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/cluster/logs/nodes",
        list_log_nodes,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )

    router.add_api_route(
        "/v1/audit/search",
        search_audit_logs,
        methods=["GET"],
        dependencies=([Security(auth, scopes=["admin"])] if is_auth else None),
    )
