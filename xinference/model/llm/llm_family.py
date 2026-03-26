# Copyright 2022-2026 XProbe Inc.
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
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from typing_extensions import Annotated, Literal

from ..._compat import (
    ROOT_KEY,
    BaseModel,
    ErrorWrapper,
    Field,
    Protocol,
    StrBytes,
    ValidationError,
    load_str_bytes,
    validator,
)
from ...constants import XINFERENCE_CACHE_DIR
from ..core import VirtualEnvSettings
from ..utils import (
    ModelInstanceInfoMixin,
    download_from_csghub,
    download_from_modelscope,
    download_from_openmind_hub,
    retry_download,
)
from . import LLM

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_LENGTH = 2048
BUILTIN_LLM_PROMPT_STYLE: Dict[str, Dict[str, Any]] = {}
BUILTIN_LLM_MODEL_CHAT_FAMILIES: Set[str] = set()
BUILTIN_LLM_MODEL_GENERATE_FAMILIES: Set[str] = set()
BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES: Set[str] = set()


class LlamaCppLLMSpecV2(BaseModel):
    model_format: Literal["ggufv2"]
    # Must in order that `str` first, then `int`
    model_size_in_billions: Union[str, int]
    quantization: str
    multimodal_projectors: Optional[List[str]]
    model_id: Optional[str]
    model_file_name_template: str
    model_file_name_split_template: Optional[str]
    quantization_parts: Optional[Dict[str, List[str]]]
    model_hub: str = "huggingface"
    model_uri: Optional[str]
    model_revision: Optional[str]
    # for MOE model, illustrates the activated model size
    activated_size_in_billions: Optional[Union[str, int]]

    @validator("model_size_in_billions", "activated_size_in_billions", pre=False)
    def validate_model_size_with_radix(cls, v: object) -> object:
        if isinstance(v, str):
            if (
                "_" in v
            ):  # for example, "1_8" just returns "1_8", otherwise int("1_8") returns 18
                return v
            else:
                return int(v)
        return v


class PytorchLLMSpecV2(BaseModel):
    model_format: Literal["pytorch", "gptq", "awq", "fp4", "fp8", "bnb"]
    # Must in order that `str` first, then `int`
    model_size_in_billions: Union[str, int]
    quantization: str
    model_id: Optional[str]
    model_hub: str = "huggingface"
    model_uri: Optional[str]
    model_revision: Optional[str]
    # for MOE model, illustrates the activated model size
    activated_size_in_billions: Optional[Union[str, int]]

    @validator("model_size_in_billions", "activated_size_in_billions", pre=False)
    def validate_model_size_with_radix(cls, v: object) -> object:
        if isinstance(v, str):
            if (
                "_" in v
            ):  # for example, "1_8" just returns "1_8", otherwise int("1_8") returns 18
                return v
            else:
                return int(v)
        return v


class MLXLLMSpecV2(BaseModel):
    model_format: Literal["mlx"]
    # Must in order that `str` first, then `int`
    model_size_in_billions: Union[str, int]
    quantization: str
    model_id: Optional[str]
    model_hub: str = "huggingface"
    model_uri: Optional[str]
    model_revision: Optional[str]
    # for MOE model, illustrates the activated model size
    activated_size_in_billions: Optional[Union[str, int]]

    @validator("model_size_in_billions", "activated_size_in_billions", pre=False)
    def validate_model_size_with_radix(cls, v: object) -> object:
        if isinstance(v, str):
            if (
                "_" in v
            ):  # for example, "1_8" just returns "1_8", otherwise int("1_8") returns 18
                return v
            else:
                return int(v)
        return v


class LLMFamilyV2(BaseModel, ModelInstanceInfoMixin):
    version: Literal[2]
    context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH
    model_name: str
    model_lang: List[str]
    model_ability: List[
        Literal[
            "embed",
            "generate",
            "chat",
            "tools",
            "vision",
            "audio",
            "omni",
            "reasoning",
            "hybrid",
        ]
    ]
    model_description: Optional[str]
    # reason for not required str here: legacy registration
    model_family: Optional[str]
    model_specs: List["LLMSpecV1"]
    chat_template: Optional[str]
    stop_token_ids: Optional[List[int]]
    stop: Optional[List[str]]
    architectures: Optional[List[str]]
    reasoning_start_tag: Optional[str]
    reasoning_end_tag: Optional[str]
    cache_config: Optional[dict]
    virtualenv: Optional[VirtualEnvSettings]
    tool_parser: Optional[str]

    class Config:
        extra = "allow"

    def _resolve_architectures(self) -> Optional[List[str]]:
        if self.architectures:
            return self.architectures
        if not self.model_family:
            return None
        for family in BUILTIN_LLM_FAMILIES:
            if family.model_name == self.model_family:
                return family.architectures
        return None

    def has_architecture(self, *architectures: str) -> bool:
        resolved = self._resolve_architectures()
        if not architectures or not resolved:
            return False
        return any(arch in resolved for arch in architectures)

    def matches_supported_architectures(
        self, supported_architectures: List[str]
    ) -> bool:
        resolved = self._resolve_architectures()
        if not resolved:
            return False
        return any(arch in supported_architectures for arch in resolved)

    def to_description(self):
        spec = self.model_specs[0]
        return {
            "model_type": "LLM",
            "address": getattr(self, "address", None),
            "accelerators": getattr(self, "accelerators", None),
            "model_name": self.model_name,
            "model_lang": self.model_lang,
            "model_ability": self.model_ability,
            "model_description": self.model_description,
            "model_format": spec.model_format,
            "model_size_in_billions": spec.model_size_in_billions,
            "model_family": self.model_family or self.model_name,
            "quantization": spec.quantization,
            "multimodal_projector": getattr(self, "multimodal_projector", None),
            "model_hub": spec.model_hub,
            "revision": spec.model_revision,
            "context_length": self.context_length,
        }

    def to_version_info(self):
        """
        Entering this function means it is already bound to a model instance,
        so there is only one spec.
        """
        from .cache_manager import LLMCacheManager
        from .utils import get_model_version

        spec = self.model_specs[0]
        multimodal_projector = getattr(self, "multimodal_projector", None)
        cache_manager = LLMCacheManager(self, multimodal_projector)

        return {
            "model_version": get_model_version(
                self.model_name,
                spec.model_format,
                spec.model_size_in_billions,
                spec.quantization,
            ),
            "model_file_location": cache_manager.get_cache_dir(),
            "cache_status": cache_manager.get_cache_status(),
            "quantization": spec.quantization,
            "multimodal_projector": multimodal_projector,
            "model_format": spec.model_format,
            "model_size_in_billions": spec.model_size_in_billions,
        }


class CustomLLMFamilyV2(LLMFamilyV2):
    @classmethod
    def parse_raw(
        cls: Any,
        b: StrBytes,
        *,
        content_type: Optional[str] = None,
        encoding: str = "utf8",
        proto: Protocol = None,
        allow_pickle: bool = False,
    ) -> LLMFamilyV2:
        # See source code of BaseModel.parse_raw
        try:
            obj = load_str_bytes(
                b,
                proto=proto,
                content_type=content_type,
                encoding=encoding,
                allow_pickle=allow_pickle,
                json_loads=cls.__config__.json_loads,
            )
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            raise ValidationError([ErrorWrapper(e, loc=ROOT_KEY)], cls)
        llm_spec: CustomLLMFamilyV2 = cls.parse_obj(obj)
        vision_model_names: Set[str] = {
            family.model_name
            for family in BUILTIN_LLM_FAMILIES
            if "vision" in family.model_ability
        }

        # check model_family
        if llm_spec.model_family is None:
            raise ValueError(
                f"You must specify `model_family` when registering custom LLM models."
            )
        assert isinstance(llm_spec.model_family, str)
        # TODO: Currently, tool call and vision models cannot be registered if it is not the builtin model_family
        if (
            "tools" in llm_spec.model_ability
            and llm_spec.model_family not in BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES
        ):
            raise ValueError(
                f"`model_family` for tool call model must be one of the following values: \n"
                f"{', '.join(list(BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES))}"
            )
        if (
            "vision" in llm_spec.model_ability
            and llm_spec.model_family not in vision_model_names
        ):
            raise ValueError(
                f"`model_family` for multimodal model must be one of the following values: \n"
                f"{', '.join(list(vision_model_names))}"
            )
        # set chat_template when it is the builtin model family
        if llm_spec.chat_template is None and "chat" in llm_spec.model_ability:
            llm_spec.chat_template = llm_spec.model_family

        # handle chat_template when user choose existing model_family
        if (
            llm_spec.chat_template is not None
            and llm_spec.chat_template in BUILTIN_LLM_PROMPT_STYLE
        ):
            llm_spec.stop_token_ids = BUILTIN_LLM_PROMPT_STYLE[llm_spec.chat_template][
                "stop_token_ids"
            ]
            llm_spec.stop = BUILTIN_LLM_PROMPT_STYLE[llm_spec.chat_template]["stop"]
            llm_spec.reasoning_start_tag = BUILTIN_LLM_PROMPT_STYLE[
                llm_spec.chat_template
            ].get("reasoning_start_tag")
            llm_spec.reasoning_end_tag = BUILTIN_LLM_PROMPT_STYLE[
                llm_spec.chat_template
            ].get("reasoning_end_tag")
            llm_spec.chat_template = BUILTIN_LLM_PROMPT_STYLE[llm_spec.chat_template][
                "chat_template"
            ]

        # check model ability, registering LLM only provides generate and chat
        # but for vision models, we add back the abilities so that
        # gradio chat interface can be generated properly
        if (
            llm_spec.model_family in vision_model_names
            and "vision" not in llm_spec.model_ability
        ):
            llm_spec.model_ability.append("vision")

        return llm_spec


LLMSpecV1 = Annotated[
    Union[LlamaCppLLMSpecV2, PytorchLLMSpecV2, MLXLLMSpecV2],
    Field(discriminator="model_format"),
]

LLMFamilyV2.update_forward_refs()
CustomLLMFamilyV2.update_forward_refs()


LLAMA_CLASSES: List[Type[LLM]] = []

BUILTIN_LLM_FAMILIES: List["LLMFamilyV2"] = []

SGLANG_CLASSES: List[Type[LLM]] = []
TRANSFORMERS_CLASSES: List[Type[LLM]] = []
VLLM_CLASSES: List[Type[LLM]] = []
MLX_CLASSES: List[Type[LLM]] = []
LMDEPLOY_CLASSES: List[Type[LLM]] = []

LLM_ENGINES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
SUPPORTED_ENGINES: Dict[str, List[Type[LLM]]] = {}


# Add decorator definition
def register_transformer(cls):
    """
    Decorator function to register a class as a transformer.

    This decorator appends the provided class to the TRANSFORMERS_CLASSES list.
    It is used to keep track of classes that are considered transformers.

    Args:
        cls (class): The class to be registered as a transformer.

    Returns:
        class: The same class that was passed in, after being registered.
    """
    # Append the class to the list of transformer classes
    TRANSFORMERS_CLASSES.append(cls)
    return cls


def cache_model_tokenizer_and_config(
    llm_family: LLMFamilyV2,
) -> str:
    """
    Download model config.json and tokenizers only
    """
    llm_spec = llm_family.model_specs[0]
    cache_dir = _get_cache_dir_for_model_mem(llm_family, llm_spec, "tokenizer_config")
    os.makedirs(cache_dir, exist_ok=True)
    patterns = ["tokenizer*", "config.json", "configuration*", "tokenization*"]
    if llm_spec.model_hub == "huggingface":
        from huggingface_hub import snapshot_download

        download_dir = retry_download(
            snapshot_download,
            llm_family.model_name,
            {
                "model_size": llm_spec.model_size_in_billions,
                "model_format": llm_spec.model_format,
            },
            llm_spec.model_id,
            revision=llm_spec.model_revision,
            allow_patterns=patterns,
            local_dir=cache_dir,
        )
    elif llm_spec.model_hub == "modelscope":
        from modelscope.hub.snapshot_download import snapshot_download

        download_dir = retry_download(
            snapshot_download,
            llm_family.model_name,
            {
                "model_size": llm_spec.model_size_in_billions,
                "model_format": llm_spec.model_format,
            },
            llm_spec.model_id,
            revision=llm_spec.model_revision,
            allow_patterns=patterns,
            local_dir=cache_dir,
        )
    else:
        raise NotImplementedError(
            f"Does not support download config.json and "
            f"tokenizer related files via {llm_spec.model_hub}"
        )
    return download_dir


def cache_model_config(llm_family: LLMFamilyV2):
    """Download model config.json into cache_dir,
    returns local filepath
    """
    llm_spec = llm_family.model_specs[0]
    cache_dir = _get_cache_dir_for_model_mem(llm_family, llm_spec, "model_mem")
    config_file = os.path.join(cache_dir, "config.json")
    if not os.path.islink(config_file) and not os.path.exists(config_file):
        os.makedirs(cache_dir, exist_ok=True)
        if llm_spec.model_hub == "huggingface":
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=llm_spec.model_id, filename="config.json", local_dir=cache_dir
            )
        else:
            from modelscope.hub.file_download import model_file_download

            download_path = model_file_download(
                model_id=llm_spec.model_id, file_path="config.json"
            )
            os.symlink(download_path, config_file)
    return config_file


def _get_cache_dir_for_model_mem(
    llm_family: LLMFamilyV2,
    llm_spec: "LLMSpecV1",
    category: str,
    create_if_not_exist=True,
):
    """
    Get file dir for special usage, like `cal-model-mem` and download partial files for

    e.g. for cal-model-mem, (might called from supervisor / cli)
    Temporary use separate dir from worker's cache_dir, due to issue of different style of symlink.
    """
    cache_dir_name = (
        f"{llm_family.model_name}-{llm_spec.model_format}"
        f"-{llm_spec.model_size_in_billions}b-{llm_spec.quantization}"
    )
    cache_dir = os.path.realpath(
        os.path.join(XINFERENCE_CACHE_DIR, "v2", category, cache_dir_name)
    )
    if create_if_not_exist and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def match_model_size(
    model_size: Union[int, str], spec_model_size: Union[int, str]
) -> bool:
    if isinstance(model_size, str):
        model_size = model_size.replace("_", ".")
    if isinstance(spec_model_size, str):
        spec_model_size = spec_model_size.replace("_", ".")

    if model_size == spec_model_size:
        return True

    try:
        ms = int(model_size)
        ss = int(spec_model_size)
        return ms == ss
    except ValueError:
        return False


def convert_model_size_to_float(
    model_size_in_billions: Union[float, int, str],
) -> float:
    if isinstance(model_size_in_billions, str):
        if "_" in model_size_in_billions:
            ms = model_size_in_billions.replace("_", ".")
            return float(ms)
        elif "." in model_size_in_billions:
            return float(model_size_in_billions)
        else:
            return int(model_size_in_billions)
    return model_size_in_billions


def match_llm(
    model_name: str,
    model_format: Optional[str] = None,
    model_size_in_billions: Optional[Union[int, str]] = None,
    quantization: Optional[str] = None,
    download_hub: Optional[
        Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
) -> Optional[LLMFamilyV2]:
    """
    Find an LLM family, spec, and quantization that satisfy given criteria.
    """
    from .custom import get_user_defined_llm_families

    user_defined_llm_families = get_user_defined_llm_families()

    def _match_quantization(q: Union[str, None], quant: str):
        # Currently, the quantization name could include both uppercase and lowercase letters,
        # so it is necessary to ensure that the case sensitivity does not
        # affect the matching results.
        if q is None or q.lower() != quant.lower():
            return None
        return quant

    def _apply_format_to_model_id(_spec: "LLMSpecV1", q: str) -> "LLMSpecV1":
        # Different quantized versions of some models use different model ids,
        # Here we check the `{}` in the model id to format the id.
        if _spec.model_id and "{" in _spec.model_id:
            _spec.model_id = _spec.model_id.format(quantization=q)
        return _spec

    def _get_model_specs(
        _model_specs: List["LLMSpecV1"], hub: str
    ) -> List["LLMSpecV1"]:
        return [x for x in _model_specs if x.model_hub == hub]

    # priority: download_hub > download_from_modelscope() and download_from_csghub()
    # set base model
    families = BUILTIN_LLM_FAMILIES + user_defined_llm_families

    for family in families:
        if model_name != family.model_name:
            continue

        # prepare possible quantization matching options
        if download_hub is not None:
            if download_hub == "huggingface":
                model_specs = _get_model_specs(family.model_specs, download_hub)
            else:
                model_specs = _get_model_specs(
                    family.model_specs, download_hub
                ) + _get_model_specs(family.model_specs, "huggingface")
        else:
            if download_from_modelscope():
                model_specs = _get_model_specs(
                    family.model_specs, "modelscope"
                ) + _get_model_specs(family.model_specs, "huggingface")
            elif download_from_openmind_hub():
                model_specs = _get_model_specs(
                    family.model_specs, "openmind_hub"
                ) + _get_model_specs(family.model_specs, "huggingface")
            elif download_from_csghub():
                model_specs = _get_model_specs(
                    family.model_specs, "csghub"
                ) + _get_model_specs(family.model_specs, "huggingface")
            else:
                model_specs = _get_model_specs(family.model_specs, "huggingface")

        for spec in model_specs:
            # check model_format and model_size_in_billions
            if (
                model_format
                and model_format != spec.model_format
                or model_size_in_billions
                and not match_model_size(
                    model_size_in_billions, spec.model_size_in_billions
                )
            ):
                continue

            # Check quantization
            matched_quantization = _match_quantization(quantization, spec.quantization)
            if quantization and matched_quantization is None:
                continue
            _llm_family = family.copy()
            if quantization:
                _llm_family.model_specs = [
                    _apply_format_to_model_id(spec, matched_quantization)
                ]
                return _llm_family
            else:
                # TODO: If user does not specify quantization, just use the first one
                _q = "none" if spec.model_format == "pytorch" else spec.quantization
                _llm_family.model_specs = [_apply_format_to_model_id(spec, _q)]
                return _llm_family
    return None


def check_engine_by_spec_parameters(
    model_engine: str,
    model_name: str,
    model_format: str,
    model_size_in_billions: Union[str, int],
    quantization: str,
) -> Type[LLM]:
    def get_model_engine_from_spell(engine_str: str) -> str:
        for engine in LLM_ENGINES[model_name].keys():
            if engine.lower() == engine_str.lower():
                return engine
        return engine_str

    if model_name not in LLM_ENGINES:
        raise ValueError(f"Model {model_name} not found.")
    model_engine = get_model_engine_from_spell(model_engine)
    if model_engine not in LLM_ENGINES[model_name]:
        raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")
    match_params = LLM_ENGINES[model_name][model_engine]
    for param in match_params:
        if (
            model_name == param["model_name"]
            and model_format == param["model_format"]
            and model_size_in_billions == param["model_size_in_billions"]
            and quantization in param["quantizations"]
        ):
            return param["llm_class"]
    raise ValueError(
        f"Model {model_name} cannot be run on engine {model_engine}, with format {model_format}, size {model_size_in_billions} and quantization {quantization}."
    )


def check_engine_by_spec_parameters_with_virtual_env(
    model_engine: str,
    model_name: str,
    model_format: str,
    model_size_in_billions: Union[str, int],
    quantization: str,
    llm_family: Optional["LLMFamilyV2"] = None,
) -> Type[LLM]:
    from ..utils import _collect_virtualenv_engine_markers

    def get_model_engine_from_spell(engine_str: str) -> str:
        for engine in LLM_ENGINES[model_name].keys():
            if engine.lower() == engine_str.lower():
                return engine
        return engine_str

    def _select_llm_spec(
        family: "LLMFamilyV2",
    ) -> Optional["LLMSpecV1"]:
        for spec in family.model_specs:
            if model_format != spec.model_format:
                continue
            if not match_model_size(
                model_size_in_billions, spec.model_size_in_billions
            ):
                continue
            if quantization and quantization.lower() != spec.quantization.lower():
                continue
            return spec
        return None

    def _select_engine_class_fallback(
        engine_classes: List[Type[LLM]],
        family: "LLMFamilyV2",
    ) -> Optional[Type[LLM]]:
        if not engine_classes:
            return None
        abilities = set(getattr(family, "model_ability", []) or [])
        preferred_tokens: List[Tuple[str, ...]] = []
        if any(a in abilities for a in ("vision", "audio", "omni")):
            preferred_tokens.append(("multi", "vision", "omni"))
        if "chat" in abilities:
            preferred_tokens.append(("chat",))
        for tokens in preferred_tokens:
            for engine_cls in engine_classes:
                cls_name = engine_cls.__name__.lower()
                if any(token in cls_name for token in tokens):
                    return engine_cls
        return engine_classes[0]

    if model_name not in LLM_ENGINES:
        raise ValueError(f"Model {model_name} not found.")
    if llm_family is None:
        llm_family = next(
            (f for f in BUILTIN_LLM_FAMILIES if f.model_name == model_name), None
        )
    engine_markers = _collect_virtualenv_engine_markers(llm_family)
    if engine_markers and model_engine.lower() not in engine_markers:
        raise ValueError(
            f"Engine {model_engine} is not listed in virtualenv packages for model {model_name}."
        )
    model_engine = get_model_engine_from_spell(model_engine)
    if model_engine not in LLM_ENGINES[model_name]:
        if model_engine.lower() in engine_markers:
            if model_engine.lower() == "mlx" and model_format != "mlx":
                raise ValueError(
                    f"Engine {model_engine} only supports mlx format, got {model_format}."
                )
            if llm_family is None:
                raise ValueError(f"Model {model_name} not found.")
            llm_spec = _select_llm_spec(llm_family)
            if llm_spec is None:
                raise ValueError(
                    f"Model {model_name} cannot be run on engine {model_engine}, "
                    f"with format {model_format}, size {model_size_in_billions} "
                    f"and quantization {quantization}."
                )
            for engine_name, engine_classes in SUPPORTED_ENGINES.items():
                if engine_name.lower() == model_engine.lower() and engine_classes:
                    engine_cls = _select_engine_class_fallback(
                        engine_classes, llm_family
                    )
                    if engine_cls is None:
                        raise ValueError(
                            f"Model {model_name} cannot be run on engine {model_engine}, "
                            f"with format {model_format}, size {model_size_in_billions} "
                            f"and quantization {quantization}."
                        )
                    logger.warning(
                        "Bypassing engine compatibility checks for %s due to virtualenv marker.",
                        model_engine,
                    )
                    return engine_cls
        raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")
    match_params = LLM_ENGINES[model_name][model_engine]
    for param in match_params:
        if (
            model_name == param["model_name"]
            and model_format == param["model_format"]
            and model_size_in_billions == param["model_size_in_billions"]
            and quantization in param["quantizations"]
        ):
            return param["llm_class"]
    if model_engine.lower() in engine_markers:
        if model_engine.lower() == "mlx" and model_format != "mlx":
            raise ValueError(
                f"Engine {model_engine} only supports mlx format, got {model_format}."
            )
        if llm_family is None:
            raise ValueError(f"Model {model_name} not found.")
        llm_spec = _select_llm_spec(llm_family)
        if llm_spec is None:
            raise ValueError(
                f"Model {model_name} cannot be run on engine {model_engine}, "
                f"with format {model_format}, size {model_size_in_billions} "
                f"and quantization {quantization}."
            )
        for engine_name, engine_classes in SUPPORTED_ENGINES.items():
            if engine_name.lower() == model_engine.lower() and engine_classes:
                engine_cls = _select_engine_class_fallback(engine_classes, llm_family)
                if engine_cls is None:
                    raise ValueError(
                        f"Model {model_name} cannot be run on engine {model_engine}, "
                        f"with format {model_format}, size {model_size_in_billions} "
                        f"and quantization {quantization}."
                    )
                logger.warning(
                    "Bypassing engine compatibility checks for %s due to virtualenv marker.",
                    model_engine,
                )
                return engine_cls
    raise ValueError(
        f"Model {model_name} cannot be run on engine {model_engine}, with format {model_format}, size {model_size_in_billions} and quantization {quantization}."
    )
