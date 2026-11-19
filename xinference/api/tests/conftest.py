from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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


@pytest.fixture
def isolated_system_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ...core import observability_settings_store, system_settings_store
    from .. import restful_api

    monkeypatch.setattr(
        restful_api, "XINFERENCE_SYSTEM_SETTINGS_PATH", str(tmp_path / "settings.json")
    )
    monkeypatch.setattr(system_settings_store, "_store", None)
    monkeypatch.setattr(observability_settings_store, "_store", None)
