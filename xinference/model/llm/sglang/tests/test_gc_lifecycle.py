# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("mode", ["enabled", "disabled", "already-frozen"])
def test_gc_freeze_preserves_request_collection_and_external_owner(mode):
    # Collection is process-wide. Exercise real cycles in an isolated process.
    script = r"""
import gc, sys, weakref
from xinference.model.llm.sglang.gc_lifecycle import InitializationGCFreeze

mode = sys.argv[1]
if mode == "disabled":
    gc.disable()
if mode == "already-frozen":
    gc.collect()
    gc.freeze()
was_enabled, was_frozen = gc.isenabled(), bool(gc.get_freeze_count())
guard = InitializationGCFreeze()
class Library:
    pass
library = Library()
library.cycle = library
library_ref = weakref.ref(library)
try:
    guard.start()
    assert gc.get_freeze_count() > 0
    assert gc.isenabled() == was_enabled
    del library
    gc.collect()
    assert library_ref() is not None, "new library graph must be frozen"
    class Request:
        pass
    request = Request()
    request.cycle = request
    ref = weakref.ref(request)
    # Starting again must not freeze request objects created after initialization.
    guard.start()
    del request
    gc.collect()
    assert ref() is None, "request cycles must remain collectible"
finally:
    guard.close()
    guard.close()
assert gc.isenabled() == was_enabled
assert bool(gc.get_freeze_count()) == was_frozen
if was_frozen:
    gc.unfreeze()  # The external owner ends its own lifetime.
gc.collect()
assert library_ref() is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script, mode], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_engine_shutdown_restores_gc_after_failure():
    from ..core import SGLANGModel

    model = object.__new__(SGLANGModel)
    model._engine = SimpleNamespace(
        pid=1, shutdown=Mock(side_effect=RuntimeError("stop"))
    )
    model._gc_freeze = SimpleNamespace(close=Mock())
    cache = model._weight_cache = SimpleNamespace(stop=Mock())
    with pytest.raises(RuntimeError, match="stop"):
        model.stop()
    model._gc_freeze.close.assert_called_once()
    cache.stop.assert_called_once()
    assert model._weight_cache is None
