# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Data-plane response schemas and constants."""

from __future__ import annotations

from typing import Any, Dict

from .._compat import BaseModel


class RouterError(BaseModel):
    message: str
    type: str
    code: int


class RouterErrorResponse(BaseModel):
    error: RouterError


class RouterHealth(BaseModel):
    status: str
    logical_model: str
    backend_url: str
    pools: Dict[str, Any]
    tokenization: Dict[str, Any]
