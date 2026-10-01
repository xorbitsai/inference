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
def test_profile_records_duration_and_preserves_exception(monkeypatch, fails):
    import torch

    monkeypatch.setattr(profiling, "_ENABLED", True)
    monkeypatch.setattr(profiling.time, "perf_counter", Mock(side_effect=[10, 10.25]))
    sync = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    log = Mock()
    monkeypatch.setattr(profiling.logger, "info", log)
    device = SimpleNamespace(type="cuda")

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
    assert sync.call_count == (1 if fails else 2)
