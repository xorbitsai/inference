# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Compatibility alias for the shared Xavier snapshot module."""

import sys
from typing import TYPE_CHECKING

from ...xavier.backends.torch import snapshot as _implementation

if TYPE_CHECKING:
    from ...xavier.backends.torch.snapshot import KVSnapshotStore as KVSnapshotStore
    from ...xavier.backends.torch.snapshot import block_major_view as block_major_view

sys.modules[__name__] = _implementation
