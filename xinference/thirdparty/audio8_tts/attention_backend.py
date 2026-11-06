# SPDX-License-Identifier: Apache-2.0
"""Attention backend selection for the Audio8 TTS adapter.

FA3 kernels are built for Hopper only. On devices that ship no FA3 kernel
image the portable FlashInfer / SDPA path is selected automatically so that a
default deployment starts without extra configuration. ``fa3`` remains the
default everywhere else, and an explicit environment override always wins.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

ATTENTION_BACKEND_ENV = "AUDIO8_TTS_ATTENTION_BACKEND"
DEFAULT_ATTENTION_BACKEND = "fa3"
PORTABLE_ATTENTION_BACKEND = "flashinfer"

# Consumer Blackwell (RTX 50 series) reports compute capability (12, 0) and has
# no FA3 kernel image, so flash_attn_with_kvcache fails at launch with
# "no kernel image is available for execution on the device".
_CAPABILITIES_WITHOUT_FA3 = frozenset({(12, 0)})


def _device_capability() -> Optional[Tuple[int, int]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability()
    except Exception:  # never block startup on a probe
        logger.debug("Could not query CUDA compute capability", exc_info=True)
        return None


@functools.lru_cache(maxsize=1)
def fa3_kernels_available() -> bool:
    """Whether the current device is expected to have an FA3 kernel image."""
    capability = _device_capability()
    if capability is None:
        return True
    if capability in _CAPABILITIES_WITHOUT_FA3:
        logger.info(
            "Compute capability %s has no FA3 kernel image; defaulting to the "
            "'%s' attention backend. Set %s to override.",
            capability,
            PORTABLE_ATTENTION_BACKEND,
            ATTENTION_BACKEND_ENV,
        )
        return False
    return True


def resolve_attention_backend() -> str:
    """Return the attention backend name, honouring the environment override."""
    override = os.getenv(ATTENTION_BACKEND_ENV)
    if override:
        return override.lower()
    if fa3_kernels_available():
        return DEFAULT_ATTENTION_BACKEND
    return PORTABLE_ATTENTION_BACKEND
