# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Compatibility alias for the shared Xavier gpu_transfer module."""

import sys
from typing import TYPE_CHECKING

from ...xavier.backends.torch import gpu_transfer as _implementation

if TYPE_CHECKING:
    from ...xavier.backends.torch.gpu_transfer import GPUTransfer as GPUTransfer
    from ...xavier.backends.torch.gpu_transfer import (
        GPUTransferMixin as GPUTransferMixin,
    )
    from ...xavier.backends.torch.gpu_transfer import (
        finish_before_cancel as finish_before_cancel,
    )

sys.modules[__name__] = _implementation
