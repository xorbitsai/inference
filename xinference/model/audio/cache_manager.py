from typing import TYPE_CHECKING

from ..cache_manager import CacheManager

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2


class AudioCacheManager(CacheManager):
    def __init__(self, model_family: "AudioModelFamilyV2"):
        super().__init__(model_family)

    @staticmethod
    def is_model_from_builtin_dir(model_name: str, model_type: str) -> bool:
        """
        Check if a model comes from the builtin directory for audio models.
        """
        return CacheManager.is_model_from_builtin_dir(model_name, model_type)

    @staticmethod
    def resolve_model_source(
        model_name: str, model_type: str, builtin_model_names=None
    ) -> str:
        """
        Resolve the source of an audio model.
        """
        return CacheManager.resolve_model_source(
            model_name, model_type, builtin_model_names
        )

    @staticmethod
    def is_builtin_model(
        model_name: str, model_type: str, builtin_model_names=None
    ) -> bool:
        """
        Determine if an audio model should be considered builtin.
        """
        return CacheManager.is_builtin_model(
            model_name, model_type, builtin_model_names
        )
