from typing import TYPE_CHECKING

from ..custom import ModelRegistry

if TYPE_CHECKING:
    from .core import FlexibleModelSpec


class FlexibleModelRegistry(ModelRegistry):
    model_type = "flexible"

    def __init__(self):
        from .core import FLEXIBLE_MODELS

        super().__init__()
        self.models = FLEXIBLE_MODELS
        self.builtin_models = []

    def register(self, model_spec: "FlexibleModelSpec", persist: bool):
        from ..cache_manager import CacheManager
        from ..utils import is_valid_model_name, is_valid_model_uri

        if not is_valid_model_name(model_spec.model_name):
            raise ValueError(f"Invalid model name {model_spec.model_name}.")

        model_uri = model_spec.model_uri
        if model_uri and not is_valid_model_uri(model_uri):
            raise ValueError(f"Invalid model URI {model_uri}.")

        if model_spec.launcher_args:
            try:
                model_spec.parser_args()
            except Exception:
                raise ValueError(
                    f"Invalid model launcher args {model_spec.launcher_args}."
                )

        with self.lock:
            for model_name in [spec.model_name for spec in self.models]:
                if model_spec.model_name == model_name:
                    raise ValueError(
                        f"Model name conflicts with existing model {model_spec.model_name}"
                    )
            self.models.append(model_spec)

        if persist:
            cache_manager = CacheManager(model_spec)
            cache_manager.register_custom_model(self.model_type)


def get_flexible_models():
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("flexible")
    return registry.get_custom_models()


def register_flexible_model(model_spec: "FlexibleModelSpec", persist: bool):
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("flexible")
    registry.register(model_spec, persist)


def unregister_flexible_model(model_name: str, raise_error: bool = True):
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("flexible")
    registry.unregister(model_name, raise_error)
