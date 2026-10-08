from __future__ import annotations

import hmac
import os
from typing import Iterable

import httpx

from .constants import TOKEN_ROUTER_BACKEND_AUTHORIZATION_HEADER

FORWARDED_REQUEST_HEADERS = {
    "accept",
    "content-type",
    "user-agent",
    "x-request-id",
    "authorization",
}
FORWARDED_RESPONSE_HEADERS = {
    "cache-control",
    "content-type",
    "x-request-id",
}


def request_headers(
    incoming: Iterable[tuple[bytes, bytes]], *, backend_api_key: str, request_id: str
) -> dict[str, str]:
    headers: dict[str, str] = {}
    backend_authorization = ""
    internal_tokens = tuple(
        token
        for name in (
            "XINFERENCE_TOKEN_ROUTER_DATA_PLANE_TOKEN",
            "XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN",
        )
        if (token := os.getenv(name))
    )
    for key_bytes, value_bytes in incoming:
        key = key_bytes.decode("latin-1").lower()
        value = value_bytes.decode("latin-1")
        if key == TOKEN_ROUTER_BACKEND_AUTHORIZATION_HEADER:
            backend_authorization = value
        elif key in FORWARDED_REQUEST_HEADERS:
            if key == "authorization" and any(
                hmac.compare_digest(value, f"Bearer {token}")
                for token in internal_tokens
            ):
                continue
            headers[key] = value
    if backend_api_key:
        headers["authorization"] = f"Bearer {backend_api_key}"
    elif backend_authorization:
        headers["authorization"] = backend_authorization
    headers["content-type"] = "application/json"
    headers["x-request-id"] = request_id
    return headers


def response_headers(response: httpx.Response) -> dict[str, str]:
    headers = {
        key: value
        for key, value in response.headers.items()
        if key.lower() in FORWARDED_RESPONSE_HEADERS
    }
    headers.pop("content-length", None)
    return headers
