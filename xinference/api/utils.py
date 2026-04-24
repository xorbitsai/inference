# Copyright 2022-2026 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
API utilities for reducing code duplication in RESTful API.
"""

import logging
import os
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Negative cache for ``get_model`` – prevents retry floods from blocking
# the Supervisor Actor message queue.
#
# When ``get_model`` raises ``ValueError`` ("Model not found …"), the uid is
# stored with an expiry timestamp.  Subsequent requests for the **same uid**
# within the TTL window are rejected at the REST layer (HTTP 404) **without**
# sending an RPC to the Supervisor Actor, thereby avoiding Actor-lock
# contention caused by high-frequency retries with non-existent uids.
# ---------------------------------------------------------------------------

_MODEL_NOT_FOUND_CACHE: OrderedDict[str, float] = OrderedDict()

# TTL in seconds – configurable via environment variable.
_MODEL_NOT_FOUND_TTL: float = float(
    os.environ.get("XINFERENCE_MODEL_NOT_FOUND_CACHE_TTL", "10")
)

# Maximum number of cached entries (LRU eviction when exceeded).
_MODEL_NOT_FOUND_MAX_SIZE: int = int(
    os.environ.get("XINFERENCE_MODEL_NOT_FOUND_CACHE_MAX_SIZE", "1000")
)


def invalidate_model_not_found_cache(model_uid: str) -> None:
    """Remove *model_uid* from the negative cache.

    Call this after a model is successfully launched so that ``get_model``
    requests for the newly available uid are **not** blocked by a stale
    negative-cache entry.
    """
    _MODEL_NOT_FOUND_CACHE.pop(model_uid, None)


def _check_negative_cache(model_uid: str) -> Optional[str]:
    """Return the cached error detail if *model_uid* is in the negative cache
    and has not expired; otherwise return ``None``.

    Expired entries are lazily evicted on access.
    """
    expiry = _MODEL_NOT_FOUND_CACHE.get(model_uid)
    if expiry is None:
        return None
    if time.monotonic() > expiry:
        _MODEL_NOT_FOUND_CACHE.pop(model_uid, None)
        return None
    return f"Model not found in the model list, uid: {model_uid} (cached, retry after {_MODEL_NOT_FOUND_TTL}s)"


def _put_negative_cache(model_uid: str) -> None:
    """Insert *model_uid* into the negative cache with the configured TTL."""
    _MODEL_NOT_FOUND_CACHE[model_uid] = time.monotonic() + _MODEL_NOT_FOUND_TTL
    # LRU eviction: drop oldest entries when over capacity.
    while len(_MODEL_NOT_FOUND_CACHE) > _MODEL_NOT_FOUND_MAX_SIZE:
        _MODEL_NOT_FOUND_CACHE.popitem(last=False)


async def require_model(
    get_supervisor_ref: Callable,
    model_uid: str,
    report_error_event: Optional[Callable] = None,
) -> Any:
    """
    Get a model with standardized error handling.

    Replaces the repeated pattern:
    ```python
    try:
        model = await (await self._get_supervisor_ref()).get_model(model_uid)
    except ValueError as ve:
        logger.error(str(ve), exc_info=True)
        await self._report_error_event(model_uid, str(ve))
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(e, exc_info=True)
        await self._report_error_event(model_uid, str(e))
        raise HTTPException(status_code=500, detail=str(e))
    ```

    Usage:
    ```python
    model = await require_model(
        self._get_supervisor_ref, model_uid, self._report_error_event
    )
    ```

    Args:
        get_supervisor_ref: Async function to get supervisor reference
        model_uid: Model unique identifier
        report_error_event: Optional async function to report error events

    Returns:
        The model instance

    Raises:
        HTTPException: With appropriate status code on error
    """
    try:
        supervisor = await get_supervisor_ref()

        # ---- negative-cache fast path ----
        cached_detail = _check_negative_cache(model_uid)
        if cached_detail is not None:
            logger.debug("get_model blocked by negative cache for uid: %s", model_uid)
            raise HTTPException(status_code=404, detail=cached_detail)

        return await supervisor.get_model(model_uid)
    except HTTPException:
        raise
    except ValueError as ve:
        logger.error(str(ve), exc_info=True)
        if report_error_event:
            await report_error_event(model_uid, str(ve))
        # Write to negative cache so that rapid retries for the same
        # non-existent uid are handled at the REST layer without reaching
        # the Supervisor Actor.
        _put_negative_cache(model_uid)
        # HTTP 404 – "resource not found" is the correct semantic for a
        # model uid that does not exist.
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(e, exc_info=True)
        if report_error_event:
            await report_error_event(model_uid, str(e))
        raise HTTPException(status_code=500, detail=str(e))
