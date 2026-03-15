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
from typing import Any, Callable, Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)


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
        return await supervisor.get_model(model_uid)
    except ValueError as ve:
        logger.error(str(ve), exc_info=True)
        if report_error_event:
            await report_error_event(model_uid, str(ve))
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(e, exc_info=True)
        if report_error_event:
            await report_error_event(model_uid, str(e))
        raise HTTPException(status_code=500, detail=str(e))
