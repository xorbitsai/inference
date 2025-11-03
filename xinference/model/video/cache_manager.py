from typing import TYPE_CHECKING

from ..cache_manager import CacheManager

if TYPE_CHECKING:
    from .core import VideoModelFamilyV2


class VideoCacheManager(CacheManager):
    def __init__(self, model_family: "VideoModelFamilyV2"):
        super().__init__(model_family)

    @staticmethod
    def is_model_from_builtin_dir(model_name: str, model_type: str) -> bool:
        """
        Check if a model comes from the builtin directory for video models.
        """
        return CacheManager.is_model_from_builtin_dir(model_name, model_type)

    @staticmethod
    def resolve_model_source(
        model_name: str, model_type: str, builtin_model_names=None
    ) -> str:
        """
        Resolve the source of a video model.
        """
        return CacheManager.resolve_model_source(
            model_name, model_type, builtin_model_names
        )

    @staticmethod
    def is_builtin_model(
        model_name: str, model_type: str, builtin_model_names=None
    ) -> bool:
        """
        Determine if a video model should be considered builtin.
        """
        return CacheManager.is_builtin_model(
            model_name, model_type, builtin_model_names
        )
