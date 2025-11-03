import os
from typing import TYPE_CHECKING

from ..cache_manager import CacheManager

if TYPE_CHECKING:
    from .core import RerankModelFamilyV2


class RerankCacheManager(CacheManager):
    def __init__(self, model_family: "RerankModelFamilyV2"):
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

    @staticmethod
    def is_model_from_builtin_dir(model_name: str, model_type: str) -> bool:
        """
        Check if a model comes from the builtin directory for rerank models.
        """
        return CacheManager.is_model_from_builtin_dir(model_name, model_type)

    @staticmethod
    def resolve_model_source(
        model_name: str, model_type: str, builtin_model_names=None
    ) -> str:
        """
        Resolve the source of a rerank model.
        """
        return CacheManager.resolve_model_source(
            model_name, model_type, builtin_model_names
        )

    @staticmethod
    def is_builtin_model(
        model_name: str, model_type: str, builtin_model_names=None
    ) -> bool:
        """
        Determine if a rerank model should be considered builtin.
        """
        return CacheManager.is_builtin_model(
            model_name, model_type, builtin_model_names
        )
