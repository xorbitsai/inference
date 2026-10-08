# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Compatibility alias for the shared Xavier request_transfer module."""

import sys
from typing import TYPE_CHECKING

from ...xavier.backends.torch import request_transfer as _implementation

if TYPE_CHECKING:
    from ...xavier.backends.torch.request_transfer import LayerRead as LayerRead
    from ...xavier.backends.torch.request_transfer import batch_reads as batch_reads
    from ...xavier.backends.torch.request_transfer import pack_reads as pack_reads
    from ...xavier.backends.torch.request_transfer import unpack_reads as unpack_reads

sys.modules[__name__] = _implementation
