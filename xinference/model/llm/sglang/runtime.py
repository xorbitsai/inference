# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Prepare SGLang Runtime's cache path and authenticated startup requests."""

import os
import threading
from functools import wraps

_RUNTIME_START_LOCK = threading.Lock()


def create_runtime(runtime_class, **config):
    # SGLang compares resolved dependencies against its unresolved staging path.
    cache_dir = os.environ.get("SGLANG_JIT_CACHE_DIR") or "~/.cache/sglang/jit"
    os.environ["SGLANG_JIT_CACHE_DIR"] = os.path.realpath(os.path.expanduser(cache_dir))
    api_key = config.get("api_key")
    if not api_key:
        return runtime_class(**config)

    from sglang.lang.backend import runtime_endpoint

    # SGLang 0.5.21 passes api_key to the HTTP server but constructs its
    # RuntimeEndpoint without it. Scope the credential bridge to this startup
    # in the model process; restore the SDK class even when startup fails.
    with _RUNTIME_START_LOCK:
        original = runtime_endpoint.RuntimeEndpoint

        @wraps(original)
        def authenticated_endpoint(*args, **kwargs):
            if len(args) < 2:
                kwargs.setdefault("api_key", api_key)
            return original(*args, **kwargs)

        runtime_endpoint.RuntimeEndpoint = authenticated_endpoint
        try:
            return runtime_class(**config)
        finally:
            runtime_endpoint.RuntimeEndpoint = original
