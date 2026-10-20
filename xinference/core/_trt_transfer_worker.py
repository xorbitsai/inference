# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Pre-bootstrap a GPU transfer worker without loading TRT's UCX libraries."""

import sys

from ..engine_hooks import register_engine_registration_hook


def _transfer_engines(engines):
    # This child only transfers buffers. Probing TRT during model-registry
    # initialization would load a second UCX beside Python NIXL's runtime.
    if "TRT-LLM" in engines:
        engines["TRT-LLM"] = []


if "xinference.model" not in sys.modules:
    register_engine_registration_hook("LLM", _transfer_engines)
