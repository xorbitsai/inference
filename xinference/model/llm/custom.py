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
        self.builtin_models = [x.model_name for x in BUILTIN_LLM_FAMILIES]

    def add_ud_model(self, model_spec):
        from . import generate_engine_config_by_model_family

        self.models.append(model_spec)
        generate_engine_config_by_model_family(model_spec)

    def check_model_uri(self, llm_family: "LLMFamilyV1"):
        from ..utils import is_valid_model_uri

        for spec in llm_family.model_specs:
            model_uri = spec.model_uri
            if model_uri and not is_valid_model_uri(model_uri):
                raise ValueError(f"Invalid model URI {model_uri}.")

    def remove_ud_model(self, llm_family: "LLMFamilyV1"):
        from .llm_family import LLM_ENGINES

        UD_LLM_FAMILIES.remove(llm_family)
        del LLM_ENGINES[llm_family.model_name]

    def remove_ud_model_files(self, llm_family: "LLMFamilyV1"):
        from .cache_manager import LLMCacheManager

        _llm_family = llm_family.copy()
        for spec in llm_family.model_specs:
            _llm_family.model_specs = [spec]
            cache_manager = LLMCacheManager(_llm_family)
            cache_manager.unregister_custom_model(self.model_type)


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
