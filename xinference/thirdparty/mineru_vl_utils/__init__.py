import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.14"
__version_info__ = (0, 1, 14)

__lazy_attrs__ = {
    "MinerUClient": (".mineru_client", "MinerUClient"),
    "MinerUSamplingParams": (".mineru_client", "MinerUSamplingParams"),
    "MinerULogitsProcessor": (".logits_processor.vllm_v1_no_repeat_ngram", "VllmV1NoRepeatNGramLogitsProcessor"),
}

if TYPE_CHECKING:
    # Rename for future compatibility
    from .logits_processor.vllm_v1_no_repeat_ngram import (
        VllmV1NoRepeatNGramLogitsProcessor as MinerULogitsProcessor,
    )
    from .mineru_client import MinerUClient, MinerUSamplingParams


def __getattr__(name: str):
    if name in __lazy_attrs__:
        module_name, attr_name = __lazy_attrs__[name]
        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")


__all__ = [
    "MinerUClient",
    "MinerUSamplingParams",
    "MinerULogitsProcessor",
    "__version__",
    "__version_info__",
]
