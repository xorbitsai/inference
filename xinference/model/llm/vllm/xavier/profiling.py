# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Compatibility alias for the shared Xavier profiling module."""

import sys
from typing import TYPE_CHECKING

from ...xavier import profiling as _implementation

if TYPE_CHECKING:
    from ...xavier.profiling import profile_stage as profile_stage

sys.modules[__name__] = _implementation
