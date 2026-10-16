# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.parametrize("mode", ["enabled", "disabled", "already-frozen"])
def test_snapshot_gc_preserves_new_request_cycles_and_external_owner(mode):
    source = Path(__file__).parents[3] / "sglang" / "gc_lifecycle.py"
    script = r"""
import gc, runpy, sys, weakref
guard = runpy.run_path(sys.argv[1])["InitializationGCFreeze"]()
mode = sys.argv[2]
if mode == "disabled":
    gc.disable()
if mode == "already-frozen":
    gc.collect()
    gc.freeze()
was_enabled, was_frozen = gc.isenabled(), bool(gc.get_freeze_count())
class Library:
    pass
library = Library()
library.cycle = library
library_ref = weakref.ref(library)
try:
    guard.start()
    assert gc.isenabled() == was_enabled
    del library
    gc.collect()
    assert library_ref() is not None
    request = Library()
    request.cycle = request
    request_ref = weakref.ref(request)
    guard.start()
    del request
    gc.collect()
    assert request_ref() is None, "new request cycles must remain collectible"
finally:
    guard.close()
    guard.close()
assert gc.isenabled() == was_enabled
assert bool(gc.get_freeze_count()) == was_frozen
if was_frozen:
    gc.unfreeze()
gc.collect()
assert library_ref() is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(source), mode],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.asyncio
async def test_actor_starts_gc_guard_and_closes_it_after_cleanup_error():
    from ..transfer import TransferActor

    guard = SimpleNamespace(start=Mock(), close=Mock())
    actor = SimpleNamespace(
        init_rank=Mock(),
        _snapshot_gc_freeze=guard,
        close_gpu_caches_v1=AsyncMock(),
        _layer_send_tasks_v1=set(),
        _context=SimpleNamespace(
            closeConnections=Mock(side_effect=RuntimeError("stop"))
        ),
    )
    await TransferActor.__post_create__(actor)
    guard.start.assert_called_once()
    with pytest.raises(RuntimeError, match="stop"):
        await TransferActor.__pre_destroy__(actor)
    guard.close.assert_called_once()


@pytest.mark.parametrize("v1,gpu_budget", [(True, None), (True, 64), (False, None)])
@pytest.mark.asyncio
async def test_model_only_freezes_ordinary_v1_snapshot_runtime(
    monkeypatch, v1, gpu_budget
):
    import xoscar as xo

    from xinference.core.model import ModelActor
    from xinference.model.llm.vllm import core

    class Model:
        _is_vllm_v1 = Mock(return_value=v1)
        init_xavier = AsyncMock()

    monkeypatch.setattr(core, "VLLMModel", Model)
    create = AsyncMock(return_value=SimpleNamespace(address="local"))
    monkeypatch.setattr(xo, "create_actor", create)
    actor = SimpleNamespace(
        _model=Model(),
        address="local",
        _xavier_config={"rank": 1, "gpu_cache_bytes": gpu_budget},
    )
    await ModelActor.start_transfer_for_vllm(actor, ["local"])
    assert create.await_args.kwargs["freeze_initialization_gc"] == (
        v1 and gpu_budget is None
    )
