# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Compatibility alias for the shared Xavier block_tracker module."""

import sys
from typing import TYPE_CHECKING

from ...xavier import block_tracker as _implementation

if TYPE_CHECKING:
    from ...xavier.block_tracker import BlockTracker as BlockTracker
    from ...xavier.block_tracker import VLLMBlockTracker as VLLMBlockTracker

sys.modules[__name__] = _implementation
