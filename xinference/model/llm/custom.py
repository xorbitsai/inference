import logging
from typing import TYPE_CHECKING, List

from ..custom import ModelRegistry

if TYPE_CHECKING:
    from .llm_family import LLMFamilyV1


logger = logging.getLogger(__name__)


UD_LLM_FAMILIES: List["LLMFamilyV1"] = []


class LLMModelRegistry(ModelRegistry):
    model_type = "llm"

    def __init__(self):
        from .llm_family import BUILTIN_LLM_FAMILIES

        super().__init__()
        self.models = UD_LLM_FAMILIES
        self.builtin_models = BUILTIN_LLM_FAMILIES

    def register(self, llm_family: "LLMFamilyV1", persist: bool):
        from ..utils import is_valid_model_name, is_valid_model_uri
        from . import generate_engine_config_by_model_family
        from .cache_manager import LLMCacheManager

        if not is_valid_model_name(llm_family.model_name):
            raise ValueError(f"Invalid model name {llm_family.model_name}.")

        for spec in llm_family.model_specs:
            model_uri = spec.model_uri
            if model_uri and not is_valid_model_uri(model_uri):
                raise ValueError(f"Invalid model URI {model_uri}.")

        with self.lock:
            for family in self.builtin_models + self.models:
                if llm_family.model_name == family.model_name:
                    raise ValueError(
                        f"Model name conflicts with existing model {family.model_name}"
                    )

            UD_LLM_FAMILIES.append(llm_family)
            generate_engine_config_by_model_family(llm_family)

        if persist:
            cache_manager = LLMCacheManager(llm_family)
            cache_manager.register_custom_model(self.model_type)

    def unregister(self, model_name: str, raise_error: bool = True):
        from .cache_manager import LLMCacheManager
        from .llm_family import LLM_ENGINES

        with self.lock:
            llm_family = self.find_model(model_name)
            if llm_family:
                UD_LLM_FAMILIES.remove(llm_family)
                del LLM_ENGINES[model_name]

                _llm_family = llm_family.copy()
                for spec in llm_family.model_specs:
                    _llm_family.model_specs = [spec]
                    cache_manager = LLMCacheManager(_llm_family)
                    cache_manager.unregister_custom_model(self.model_type)
            else:
                if raise_error:
                    raise ValueError(f"Model {model_name} not found")
                else:
                    logger.warning(f"Custom model {model_name} not found")


def get_user_defined_llm_families():
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("llm")
    return registry.get_custom_models()


def register_llm(llm_family: "LLMFamilyV1", persist: bool):
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("llm")
    registry.register(llm_family, persist)


def unregister_llm(model_name: str, raise_error: bool = True):
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("llm")
    registry.unregister(model_name, raise_error)
