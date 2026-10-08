# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Compatibility alias for the shared Xavier direct_history module."""

import sys
from typing import TYPE_CHECKING

from ...xavier.backends.torch import direct_history as _implementation

if TYPE_CHECKING:
    from ...xavier.backends.torch.direct_history import (
        DirectHistoryMixin as DirectHistoryMixin,
    )
    from ...xavier.backends.torch.direct_history import HistoryStore as HistoryStore

sys.modules[__name__] = _implementation
