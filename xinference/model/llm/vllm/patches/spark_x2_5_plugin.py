"""Keep the out-of-tree Spark-X2.5 plugin compatible with vLLM 0.29."""

import glob
import logging
import os
from typing import Optional

from ._base import VllmPatch

logger = logging.getLogger(__name__)

_PLUGIN_RELATIVE_PATH = os.path.join("vllm_spark2_5_plugin", "spark2_5.py")
_ORIGINAL = """        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )"""
_PATCHED = """        # [xinference-patch] vLLM 0.29 derives tied-weight aliases itself.
        loader = AutoWeightsLoader(self)"""


def _get_plugin_file(env_path: str) -> Optional[str]:
    patterns = [
        os.path.join(
            env_path, "lib", "python*", "site-packages", _PLUGIN_RELATIVE_PATH
        ),
        os.path.join(env_path, "Lib", "site-packages", _PLUGIN_RELATIVE_PATH),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def patch_spark_x2_5_plugin(env_path: str) -> bool:
    """Remove the ``skip_prefixes`` argument removed by vLLM 0.29."""
    target_file = _get_plugin_file(env_path)
    if not target_file:
        logger.debug("Spark-X2.5 vLLM plugin not found in %s", env_path)
        return False

    with open(target_file, "r") as f:
        content = f.read()
    if "[xinference-patch] vLLM 0.29" in content:
        return False
    if _ORIGINAL not in content:
        logger.debug(
            "Spark-X2.5 plugin has no compatible patch target: %s", target_file
        )
        return False

    with open(target_file, "w") as f:
        f.write(content.replace(_ORIGINAL, _PATCHED))
    logger.info("Patched Spark-X2.5 vLLM plugin: %s", target_file)
    return True


PATCH = VllmPatch(
    name="spark_x2_5_plugin",
    fn=patch_spark_x2_5_plugin,
    description="Remove the obsolete AutoWeightsLoader skip_prefixes argument",
    removal_condition="vllm-spark2-5-plugin supports vLLM 0.29 natively",
    architectures={"Spark2_5ForCausalLM"},
)
