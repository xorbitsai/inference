import os
from typing import TYPE_CHECKING

from ..cache_manager import CacheManager

if TYPE_CHECKING:
    from .core import EmbeddingModelFamilyV2


class EmbeddingCacheManager(CacheManager):
    @staticmethod
    def get_cache_dir_for_spec(model_name: str, spec) -> str:
        """Compute a cache path without initializing or creating directories."""
        from ...constants import XINFERENCE_CACHE_DIR

        return os.path.join(
            XINFERENCE_CACHE_DIR,
            "v2",
            f"{model_name}-{spec.model_format}-{spec.quantization}",
        )

    def __init__(self, model_family: "EmbeddingModelFamilyV2"):
        from ..llm.cache_manager import LLMCacheManager

        super().__init__(model_family)
        # Composition design mode for avoiding duplicate code
        spec = self._model_family.model_specs[0]
        self.cache_helper = LLMCacheManager(
            model_family,
            multimodal_projector=getattr(spec, "multimodal_projector", None),
        )
        self._cache_dir = self.get_cache_dir_for_spec(
            self._model_family.model_name, spec
        )
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
