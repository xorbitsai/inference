# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import tempfile
import threading
from typing import TYPE_CHECKING, List, cast

from ..custom import ModelRegistry

if TYPE_CHECKING:
    from .llm_family import LLMFamilyV2


logger = logging.getLogger(__name__)


UD_LLM_FAMILIES: List["LLMFamilyV2"] = []


def _volume_ignores_case(path: str) -> bool:
    """Check case aliases inside the cache parent, conservatively on errors.

    Use existing entries when possible. Otherwise create the parent and a
    temporary probe inside it: looking up the parent's own name could test
    another filesystem or directory's case rules.
    """
    parent = os.path.dirname(os.path.abspath(path))

    def entry_ignores_case(name: str) -> bool:
        original_path = os.path.join(parent, name)
        original = os.lstat(original_path)
        try:
            alternate = os.lstat(os.path.join(parent, name.swapcase()))
        except FileNotFoundError:
            # A disappearing original is not evidence of sensitivity.
            os.lstat(original_path)
            return False
        return os.path.samestat(original, alternate)

    try:
        with os.scandir(parent) as entries:
            names = {entry.name for entry in entries}
        for name in names:
            if not name.isascii() or name.swapcase() == name:
                continue
            if name.swapcase() in names:
                # Distinct entries prove sensitivity, even if hard-linked.
                return False
            return entry_ignores_case(name)
    except OSError:
        pass

    try:
        os.makedirs(parent, exist_ok=True)
        probe = tempfile.mkdtemp(prefix=".xinference-CaseProbe-", dir=parent)
        try:
            return entry_ignores_case(os.path.basename(probe))
        finally:
            # Remove only our entry; the cache manager also needs the parent.
            os.rmdir(probe)
    except OSError as exc:
        logger.debug(
            "Cannot probe case sensitivity of cache parent %s: %s", parent, exc
        )
        return True


class LLMModelRegistry(ModelRegistry):
    model_type = "llm"

    def __init__(self):
        from .llm_family import BUILTIN_LLM_FAMILIES

        super().__init__()
        # Engine matching may resolve a custom parent while register() holds
        # this lock, which re-enters get_custom_models() on the same thread.
        self.lock = threading.RLock()
        self.models = UD_LLM_FAMILIES
        self.builtin_models = [x.model_name for x in BUILTIN_LLM_FAMILIES]

    def check_cache_identity(self, llm_family: "LLMFamilyV2"):
        from .cache_manager import LLMCacheManager
        from .llm_family import BUILTIN_LLM_FAMILIES

        # The cache directory replaces "." with "_" in the model name, so
        # "qwen2_5-instruct" would be served the files cached for the built-in
        # "qwen2.5-instruct". Compare the directories themselves: specs of one
        # family (e.g. its Hugging Face and ModelScope entries) share one by design.
        def cache_dirs(family: "LLMFamilyV2"):
            return {
                LLMCacheManager.get_cache_dir_for_spec(family.model_name, spec)
                for spec in family.model_specs
            }

        wanted = cache_dirs(llm_family)
        if not wanted:
            return
        # Keep case-only variants apart only when the cache parent supplies
        # evidence that its entries are case-sensitive.
        ignores_case = _volume_ignores_case(min(wanted))

        def key(path: str) -> str:
            return path.casefold() if ignores_case else path

        wanted_keys = {key(path) for path in wanted}
        for family in [*BUILTIN_LLM_FAMILIES, *cast(List["LLMFamilyV2"], self.models)]:
            shared = {path for path in cache_dirs(family) if key(path) in wanted_keys}
            if shared:
                raise ValueError(
                    f"Model name conflicts with existing model {family.model_name}: "
                    f"{llm_family.model_name} would use the same cache directory, "
                    f"{os.path.basename(min(shared))}."
                )

    def add_ud_model(self, model_spec):
        from . import generate_engine_config_by_model_family

        self.models.append(model_spec)
        generate_engine_config_by_model_family(model_spec)

    def check_model_uri(self, llm_family: "LLMFamilyV2"):
        from ..utils import is_valid_model_uri

        for spec in llm_family.model_specs:
            model_uri = spec.model_uri
            if model_uri and not is_valid_model_uri(model_uri):
                raise ValueError(f"Invalid model URI {model_uri}.")

    def remove_ud_model(self, llm_family: "LLMFamilyV2"):
        from .llm_family import LLM_ENGINES

        UD_LLM_FAMILIES.remove(llm_family)
        del LLM_ENGINES[llm_family.model_name]

    def remove_ud_model_files(self, llm_family: "LLMFamilyV2"):
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


def register_llm(llm_family: "LLMFamilyV2", persist: bool):
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("llm")
    registry.register(llm_family, persist)


def unregister_llm(model_name: str, raise_error: bool = True):
    from ..custom import RegistryManager

    registry = RegistryManager.get_registry("llm")
    registry.unregister(model_name, raise_error)
