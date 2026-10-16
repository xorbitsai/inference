# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from .. import profiling


def test_disabled_profile_does_not_touch_clock_or_cuda(monkeypatch):
    monkeypatch.setattr(profiling, "_ENABLED", False)
    clock = Mock(side_effect=AssertionError("unexpected timer"))
    monkeypatch.setattr(profiling.time, "perf_counter", clock)
    with profiling.profile_stage("unused", device=object()):
        pass
    clock.assert_not_called()


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("device_type", [None, "cpu", "cuda"])
def test_profile_records_duration_and_preserves_exception(
    monkeypatch, fails, device_type
):
    import torch

    monkeypatch.setattr(profiling, "_ENABLED", True)
    monkeypatch.setattr(profiling.time, "perf_counter", Mock(side_effect=[10, 10.25]))
    sync = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    log = Mock()
    monkeypatch.setattr(profiling.logger, "info", log)
    device = SimpleNamespace(type=device_type) if device_type else None

    def run():
        with profiling.profile_stage("copy", device=device, nbytes=32):
            if fails:
                raise ValueError("original")

    if fails:
        with pytest.raises(ValueError, match="original"):
            run()
    else:
        run()
    record = json.loads(log.call_args.args[1])
    assert record["elapsed_s"] == 0.25
    assert record["succeeded"] is not fails
    assert record["nbytes"] == 32
    assert sync.call_count == ((1 if fails else 2) if device_type == "cuda" else 0)


@pytest.mark.parametrize("configured", [False, True])
def test_profile_uses_configured_file_or_stderr(
    monkeypatch, tmp_path, capsys, configured
):
    import logging
    import runpy

    parent = logging.Logger("profile-parent")
    logger = logging.Logger("profile-child")
    logger.parent = parent
    path = tmp_path / "server.log"
    handler = logging.FileHandler(path) if configured else None
    if handler:
        parent.addHandler(handler)
    monkeypatch.setenv("XINFERENCE_XAVIER_PROFILE", "1")
    with monkeypatch.context() as patch:
        patch.setattr(logging, "getLogger", lambda name: logger)
        namespace = runpy.run_path(profiling.__file__)
    try:
        with namespace["profile_stage"]("test"):
            pass
        if configured:
            assert "Xavier profile:" in path.read_text()
            assert not capsys.readouterr().err
            assert not logger.handlers
        else:
            assert "Xavier profile:" in capsys.readouterr().err
    finally:
        if handler:
            handler.close()
        for fallback in logger.handlers:
            fallback.close()
