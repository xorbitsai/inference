from __future__ import annotations

from collections.abc import Iterator

import pytest
from sse_starlette.sse import AppStatus


@pytest.fixture(autouse=True)
def reset_sse_starlette_exit_event() -> Iterator[None]:
    if not hasattr(AppStatus, "should_exit_event"):
        yield
        return

    setattr(AppStatus, "should_exit_event", None)
    try:
        yield
    finally:
        setattr(AppStatus, "should_exit_event", None)
