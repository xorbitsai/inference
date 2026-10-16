# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Compatibility alias for the shared Xavier utils module."""

import sys
from typing import TYPE_CHECKING

from ...xavier import utils as _implementation

if TYPE_CHECKING:
    from ...xavier.utils import hash_block_tokens as hash_block_tokens

sys.modules[__name__] = _implementation
