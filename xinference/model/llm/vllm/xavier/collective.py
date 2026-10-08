# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Compatibility alias for the shared Xavier collective module."""

import sys
from typing import TYPE_CHECKING

from ...xavier import collective as _implementation

if TYPE_CHECKING:
    from ...xavier.collective import CollectiveRank as CollectiveRank

sys.modules[__name__] = _implementation
