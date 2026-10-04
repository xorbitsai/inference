# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Compatibility alias for the shared Xavier direct_handoff module."""

import sys
from typing import TYPE_CHECKING

from ...xavier.backends.torch import direct_handoff as _implementation

if TYPE_CHECKING:
    from ...xavier.backends.torch.direct_handoff import (
        DirectGPUTransfer as DirectGPUTransfer,
    )
    from ...xavier.backends.torch.direct_handoff import DirectRequest as DirectRequest

sys.modules[__name__] = _implementation
