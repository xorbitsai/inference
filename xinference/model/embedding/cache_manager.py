import os
from typing import TYPE_CHECKING

from ..cache_manager import CacheManager

if TYPE_CHECKING:
    from .core import EmbeddingModelFamilyV2


class EmbeddingCacheManager(CacheManager):
    def __init__(self, model_family: "EmbeddingModelFamilyV2"):
        from ..llm.cache_manager import LLMCacheManager

        super().__init__(model_family)
        # Composition design mode for avoiding duplicate code
        self.cache_helper = LLMCacheManager(model_family)

        spec = self._model_family.model_specs[0]
        model_dir_name = (
            f"{self._model_family.model_name}-{spec.model_format}-{spec.quantization}"
        )
        self._cache_dir = os.path.join(self._v2_cache_dir_prefix, model_dir_name)
        self.cache_helper._cache_dir = self._cache_dir

    def cache(self) -> str:
        spec = self._model_family.model_specs[0]
        if spec.model_uri is not None:
            return self.cache_helper.cache_uri()
        else:
            if spec.model_hub == "huggingface":
                return self.cache_helper.cache_from_huggingface()
            elif spec.model_hub == "modelscope":
                return self.cache_helper.cache_from_modelscope()
            else:
                raise ValueError(f"Unknown model hub: {spec.model_hub}")
