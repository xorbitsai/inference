"""
vLLM post-install patches registry.

Patches are auto-discovered: any module in this package that exposes a
top-level ``PATCH`` attribute (instance of VllmPatch) is registered automatically.
"""
import importlib
import logging
import pkgutil
from typing import List, Optional

from ._base import VllmPatch

logger = logging.getLogger(__name__)

# Auto-discover patch modules
_PATCHES: List[VllmPatch] = []
for _info in pkgutil.iter_modules(__path__):
    if _info.name.startswith("_"):
        continue
    _mod = importlib.import_module(f".{_info.name}", __name__)
    if hasattr(_mod, "PATCH"):
        _PATCHES.append(_mod.PATCH)


def apply_vllm_patches(
    env_path: str,
    model_name: Optional[str] = None,
    model_family: Optional[str] = None,
) -> None:
    """
    Apply applicable vLLM patches for the given model.

    Args:
        env_path: Path to the virtual environment root.
        model_name: Model name (for logging).
        model_family: Model family string, used to filter patches.
    """
    for patch in _PATCHES:
        if patch.model_families and model_family:
            if model_family.lower() not in patch.model_families:
                logger.debug(
                    "Skipping patch '%s' (not applicable to %s)",
                    patch.name,
                    model_family,
                )
                continue

        try:
            applied = patch.fn(env_path)
            if applied:
                logger.info(
                    "Applied patch '%s' for %s", patch.name, model_name or "unknown"
                )
        except Exception:
            logger.exception("Failed to apply patch '%s'", patch.name)
