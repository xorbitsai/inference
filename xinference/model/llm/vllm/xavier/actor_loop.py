# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Compatibility alias for the shared Xavier actor_loop module."""

import sys
from typing import TYPE_CHECKING

from ...xavier import actor_loop as _implementation

if TYPE_CHECKING:
    from ...xavier.actor_loop import acquire_actor_loop as acquire_actor_loop
    from ...xavier.actor_loop import release_actor_loop as release_actor_loop

sys.modules[__name__] = _implementation
