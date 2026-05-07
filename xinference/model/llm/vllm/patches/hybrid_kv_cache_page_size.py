"""
Fix hybrid attention KV cache page size for Qwen3.5/3.6 MoE models.

Problem: vLLM 0.20.x raises NotImplementedError when page sizes across
different layer types are not evenly divisible.

Solution: Replace the hard error with LCM-based padding.

Removal condition: vLLM merges fix for vllm-project/vllm#37121 / #38041.
"""

import glob
import logging
import os
from typing import Optional

from ._base import VllmPatch

logger = logging.getLogger(__name__)

# Original code to match (the problematic raise)
_KV_CACHE_ORIGINAL = """\
            if max_page_size % layer_page_size != 0:
                raise NotImplementedError(
                    "The page size of the layer is not divisible by the "
                    "maximum page size. Cannot unify by adjusting block_size."
                )
            ratio = max_page_size // layer_page_size
            new_block_size = layer_spec.block_size * ratio
            new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec"""

# Marker to detect already-patched files
_KV_CACHE_PATCHED_MARKER = "# [xinference-patch] hybrid KV cache LCM padding"


def _get_vllm_site_packages_path(env_path: str) -> Optional[str]:
    """Find the vLLM installation path within a virtual environment."""
    import subprocess

    # Method 1: use the venv's own Python to locate vllm precisely
    # This handles .pth injected parent site-packages and uv-managed envs
    for python_name in ("python3", "python"):
        python_bin = os.path.join(env_path, "bin", python_name)
        if not os.path.exists(python_bin):
            continue
        try:
            result = subprocess.run(
                [
                    python_bin,
                    "-c",
                    "import importlib.util; spec = importlib.util.find_spec('vllm'); "
                    "print(spec.submodule_search_locations[0]) if spec else exit(1)",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                vllm_path = result.stdout.strip()
                if os.path.isdir(vllm_path):
                    return vllm_path
        except Exception:
            pass

    # Method 2: fallback to glob pattern (standard venv layout)
    patterns = [
        os.path.join(env_path, "lib", "python*", "site-packages", "vllm"),
        os.path.join(env_path, "Lib", "site-packages", "vllm"),  # Windows
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    return None


def patch_kv_cache_hybrid_page_size(env_path: str) -> bool:
    """
    Patch vLLM's unify_kv_cache_spec_page_size for hybrid attention models.

    Returns:
        True if patch was applied, False if skipped.
    """
    vllm_path = _get_vllm_site_packages_path(env_path)
    if not vllm_path:
        logger.debug("vLLM not found in %s, skipping KV cache patch", env_path)
        return False

    target_file = os.path.join(vllm_path, "v1", "core", "kv_cache_utils.py")
    if not os.path.exists(target_file):
        logger.debug("kv_cache_utils.py not found at %s, skipping", target_file)
        return False

    with open(target_file, "r") as f:
        content = f.read()

    # Already patched?
    if _KV_CACHE_PATCHED_MARKER in content:
        logger.debug("KV cache hybrid patch already applied in %s", env_path)
        return False

    # Check if the original problematic code exists
    if _KV_CACHE_ORIGINAL not in content:
        logger.debug(
            "Original KV cache code not found (may be a newer vLLM with official fix), "
            "skipping patch for %s",
            env_path,
        )
        return False

    # Apply patch
    patched_section = f"""\
            {_KV_CACHE_PATCHED_MARKER}
            # Check if all smaller pages divide max evenly
            smaller_sizes = sorted(
                ps for ps in page_sizes if ps < max_page_size
            )
            if all(max_page_size % ps == 0 for ps in smaller_sizes):
                target_page_size = max_page_size
            else:
                # Hybrid model: use LCM-based padding
                # Reference: vllm-project/vllm#40128
                import math as _math
                smaller_lcm = _math.lcm(*smaller_sizes)
                target_page_size = (
                    (max_page_size + smaller_lcm - 1) // smaller_lcm
                ) * smaller_lcm
                logger.info(
                    "Hybrid KV cache: padding max page size %d -> %d "
                    "(LCM=%d, overhead=%.3f%%)",
                    max_page_size,
                    target_page_size,
                    smaller_lcm,
                    (target_page_size - max_page_size) / max_page_size * 100,
                )

            new_kv_cache_spec = {{}}
            for layer_name, layer_spec in kv_cache_spec.items():
                layer_page = layer_spec.page_size_bytes
                if layer_page == target_page_size:
                    new_kv_cache_spec[layer_name] = layer_spec
                elif (
                    layer_page < target_page_size
                    and target_page_size % layer_page == 0
                ):
                    ratio = target_page_size // layer_page
                    new_block_size = layer_spec.block_size * ratio
                    new_spec = replace(layer_spec, block_size=new_block_size)
                    assert new_spec.page_size_bytes == target_page_size
                    new_kv_cache_spec[layer_name] = new_spec
                else:
                    try:
                        new_spec = replace(
                            layer_spec, page_size_padded=target_page_size
                        )
                    except TypeError:
                        raise NotImplementedError(
                            f"Cannot pad page size for "
                            f"{{type(layer_spec).__name__}}: "
                            f"page_size_padded not supported. "
                            f"layer_page={{layer_page}}, "
                            f"target={{target_page_size}}"
                        )
                    assert new_spec.page_size_bytes == target_page_size
                    new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec"""

    new_content = content.replace(_KV_CACHE_ORIGINAL, patched_section)

    if new_content == content:
        logger.warning("KV cache patch replacement failed for %s", env_path)
        return False

    with open(target_file, "w") as f:
        f.write(new_content)

    logger.info(
        "Applied hybrid KV cache patch to %s (vllm-project/vllm#37121)",
        target_file,
    )
    return True


# --- Registration ---
PATCH = VllmPatch(
    name="hybrid_kv_cache_page_size",
    fn=patch_kv_cache_hybrid_page_size,
    description="LCM-based padding for hybrid attention KV cache page sizes",
    removal_condition="vLLM merges fix for vllm-project/vllm#37121 / #38041",
    architectures={
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
    },
)
