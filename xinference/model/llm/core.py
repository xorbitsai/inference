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

import abc
import copy
import inspect
import json
import logging
import os
import platform
import warnings
from abc import abstractmethod
from collections import defaultdict
from contextvars import ContextVar
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from ...core.utils import parse_replica_model_uid
from ...types import PeftModelConfig
from .reasoning_parser import KimiK3ReasoningParser, ReasoningParser
from .tool_parsers import TOOL_PARSERS

if TYPE_CHECKING:
    from .llm_family import LLMFamilyV2, LLMSpecV1

logger = logging.getLogger(__name__)


LLM_VERSION_INFOS: Dict[str, List[Dict]] = defaultdict(list)


def get_llm_version_infos():
    import copy

    return copy.deepcopy(LLM_VERSION_INFOS)


class LLM(abc.ABC):
    _auto_yarn_env: Optional[str] = None
    supports_cache_salt = False
    allow_batch = False
    supports_tool_controls = False
    # whether the engine can run speculative decoding with a paired drafter
    support_draft_model = False

    def __init__(
        self,
        replica_model_uid: str,
        model_family: "LLMFamilyV2",
        model_path: str,
        *args,
        **kwargs,
    ):
        self.model_uid, self.rep_id = parse_replica_model_uid(replica_model_uid)
        self.raw_model_uid = replica_model_uid
        self.model_family = model_family
        self.model_spec = model_family.model_specs[0]
        self.quantization = model_family.model_specs[0].quantization
        self.model_path = model_path
        self.reasoning_parser = None
        self.tool_parser = None
        if args:
            raise ValueError(f"Unrecognized positional arguments: {args}")
        if kwargs:
            raise ValueError(f"Unrecognized keyword arguments: {kwargs}")

    def count_tokens(
        self, messages: List[Dict], generate_config: Optional[Dict] = None
    ) -> int:
        """Count the rendered chat prompt with the loaded model's tokenizer."""
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is None or not hasattr(self, "get_full_context"):
            raise NotImplementedError("This engine does not expose chat token counting")
        if any(
            isinstance(m.get("content"), list)
            and any(p.get("type") != "text" for p in m["content"])
            for m in messages
        ):
            raise NotImplementedError(
                "This engine does not expose multimodal token counting"
            )
        config = copy.deepcopy(generate_config or {})
        context = self._get_chat_template_kwargs_from_generate_config(config, self.reasoning_parser) or {}  # type: ignore[attr-defined]
        if config.get("tools"):
            context["tools"] = config["tools"]
        template = self.model_family.chat_template or getattr(
            tokenizer, "chat_template", None
        )
        prompt = self.get_full_context(copy.deepcopy(messages), template, tokenizer=tokenizer, **context)  # type: ignore[attr-defined]
        prepare_prompt = getattr(self, "_prepare_reasoning_prompt", None)
        if prepare_prompt is not None:
            prompt = prepare_prompt(prompt, config)
        return len(tokenizer.encode(prompt))

    def prepare_anthropic_request(
        self, messages: List[Dict], config: Dict
    ) -> Dict[str, Any]:
        """Validate backend capabilities before starting an Anthropic response."""
        config = copy.deepcopy(config)
        from ...api.protocols.anthropic_capabilities import plan_anthropic_compatibility

        requested_features = frozenset(config.pop("_anthropic_requested_features", []))
        content_metrics = config.pop("_anthropic_content_metrics", {})
        capabilities = self.get_anthropic_capabilities()
        engine = capabilities.engine
        plan = plan_anthropic_compatibility(
            requested_features, content_metrics, capabilities
        )
        for decision in plan.decisions:
            logger.info(
                "Anthropic compatibility decision: model=%s engine=%s "
                "feature=%s action=%s reason=%s revision=%s",
                self.model_uid,
                engine,
                decision.feature,
                decision.action,
                decision.reason,
                capabilities.revision,
            )
        if not plan.allowed:
            rejected = next(
                decision for decision in plan.decisions if decision.action == "rejected"
            )
            if rejected.feature == "vision.max_images_per_prompt":
                raise ValueError(
                    f"Model {self.model_uid} supports at most "
                    f"{plan.effective_max_images} images per prompt, but "
                    f"{content_metrics.get('total_images', 0)} were provided "
                    f"({content_metrics.get('direct_images', 0)} direct images, "
                    f"{content_metrics.get('tool_result_images', 0)} tool-result "
                    f"images, {content_metrics.get('document_pages', 0)} PDF pages)"
                )
            if rejected.feature == "vision":
                raise ValueError("Image input requires a model with vision support")
            raise NotImplementedError(
                f"Anthropic feature {rejected.feature} is not supported by "
                f"the {engine} engine"
            )
        context = config.get("chat_template_kwargs") or {}
        if isinstance(context, str):
            context = json.loads(context)
        config["chat_template_kwargs"] = {**context, "_anthropic_request": True}
        if engine in {"vllm", "sglang"}:
            config["_anthropic_request"] = True
        if config.get("response_format") and engine not in {
            "vllm",
            "sglang",
            "llama_cpp",
            "mlx",
        }:
            raise NotImplementedError(
                f"Structured JSON output is not supported by the {engine} engine"
            )
        if (
            config.get("parallel_tool_calls") is False
            and not self.supports_tool_controls
        ):
            raise NotImplementedError(
                f"disable_parallel_tool_use is not supported by the {engine} engine"
            )
        if not self.supports_tool_controls:
            config.pop("parallel_tool_calls", None)
        choice = config.pop("tool_choice", None)
        if choice == "none":
            config.pop("tools", None)
        elif (
            choice is not None and choice != "auto" and not self.supports_tool_controls
        ):
            raise NotImplementedError(
                f"Forced tool_choice is not supported by the {engine} engine"
            )
        elif choice is not None and self.supports_tool_controls:
            config["tool_choice"] = choice
        cache_control = config.get("_anthropic_cache_control")
        has_images = any(
            isinstance(m.get("content"), list)
            and any(p.get("type") == "image_url" for p in m["content"])
            for m in messages
        )
        cache_control_applied = (
            cache_control is not None
            and (
                hasattr(self, "_prepare_explicit_prompt_cache")
                or hasattr(self, "_prepare_reusable_prompt_cache")
            )
            and not has_images
            and getattr(self, "_xavier", None) is None
            and getattr(self, "_cache_has_prompt_cache_api", True)
            and getattr(self, "_model_config", {}).get("prompt_cache_size", 2) > 0
        )
        if cache_control is not None and not cache_control_applied:
            # Markers are cache hints: keep other engines' automatic caching
            # usable, and expose the difference through the response header.
            config.pop("_anthropic_cache_control", None)
        if has_images and "vision" not in self.model_family.model_ability:
            raise ValueError("Image input requires a model with vision support")
        # Effort is a soft preference. Engines without a native effort API use
        # explicit prompt guidance, without forcing thinking on non-thinking runs.
        effort = config.pop("reasoning_effort", None)
        prepared = copy.deepcopy(messages)
        if effort and effort != "medium":
            guidance = {
                "low": "Use brief reasoning and prioritize a direct answer.",
                "high": "Reason carefully and check your work before answering.",
                "xhigh": "Use thorough reasoning and check difficult edge cases.",
                "max": "Use the reasoning needed to solve the task thoroughly.",
            }[effort]
            if prepared and prepared[0].get("role") == "system":
                prepared[0]["content"] = prepared[0]["content"] + "\n" + guidance
            else:
                prepared.insert(0, {"role": "system", "content": guidance})
            if cache_control_applied and cache_control is not None:
                for checkpoint in [
                    cache_control,
                    *cache_control.get("breakpoints", []),
                ]:
                    prefix = checkpoint["messages"]
                    if prefix and prefix[0].get("role") == "system":
                        prefix[0]["content"] += "\n" + guidance
                    else:
                        prefix.insert(0, {"role": "system", "content": guidance})
        try:
            input_tokens = self.count_tokens(prepared, config)
        except NotImplementedError:
            input_tokens = 0
        return {
            "messages": prepared,
            "generate_config": config,
            "input_tokens": input_tokens,
            "cache_control_applied": cache_control_applied,
        }

    def get_anthropic_capabilities(self):
        """Describe Anthropic features for this loaded model instance."""
        from ...api.protocols.anthropic_capabilities import (
            AnthropicBackendCapabilities,
            SupportLevel,
        )

        abilities = set(getattr(self.model_family, "model_ability", []) or [])
        engine = type(self).__module__.split(".")[-2]
        explicit_limit = getattr(self, "_explicit_limit_mm_per_prompt", None)
        raw_limit = (
            explicit_limit
            if explicit_limit is not None
            else getattr(self, "_model_config", {}).get("limit_mm_per_prompt")
        )
        max_images = raw_limit.get("image") if isinstance(raw_limit, dict) else None
        reasoning = (
            "reasoning" in abilities
            or getattr(self, "reasoning_parser", None) is not None
        )
        tools = "tools" in abilities or getattr(self, "tool_parser", None) is not None
        return AnthropicBackendCapabilities(
            engine=engine,
            vision=(
                SupportLevel.NATIVE
                if "vision" in abilities or "omni" in abilities
                else SupportLevel.UNSUPPORTED
            ),
            tools=SupportLevel.NATIVE if tools else SupportLevel.UNSUPPORTED,
            parallel_tools=(
                SupportLevel.NATIVE
                if tools and self.supports_tool_controls
                else SupportLevel.UNSUPPORTED
            ),
            reasoning=(SupportLevel.NATIVE if reasoning else SupportLevel.UNSUPPORTED),
            adaptive_thinking=(
                SupportLevel.TRANSLATED if reasoning else SupportLevel.UNSUPPORTED
            ),
            structured_output=(
                SupportLevel.NATIVE
                if engine in {"vllm", "sglang", "llama_cpp", "mlx"}
                else SupportLevel.UNSUPPORTED
            ),
            prompt_cache=(
                SupportLevel.NATIVE
                if hasattr(self, "_prepare_explicit_prompt_cache")
                or hasattr(self, "_prepare_reusable_prompt_cache")
                else SupportLevel.DEGRADED
            ),
            max_images_per_prompt=max_images,
        )

    @classmethod
    @abstractmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        raise NotImplementedError

    @staticmethod
    def _is_darwin_and_apple_silicon():
        return platform.system() == "Darwin" and platform.processor() == "arm"

    @staticmethod
    def _is_linux():
        return platform.system() == "Linux"

    @staticmethod
    @lru_cache
    def _has_cuda_device():
        """
        Use pynvml to impl this interface.
        DO NOT USE torch to impl this, which will lead to some unexpected errors.
        """
        from pynvml import nvmlDeviceGetCount, nvmlInit, nvmlShutdown

        device_count = 0
        try:
            nvmlInit()
            device_count = nvmlDeviceGetCount()
        except Exception:
            pass
        finally:
            try:
                nvmlShutdown()
            except Exception:
                pass

        return device_count > 0

    @staticmethod
    @lru_cache
    def _has_mlu_device():
        """
        Use cnmon command to detect MLU devices.
        DO NOT USE torch to impl this, which will lead to some unexpected errors.
        """
        try:
            import subprocess

            result = subprocess.run(
                ["cnmon", "info"], capture_output=True, text=True, timeout=5
            )
            return "Card 0" in result.stdout
        except Exception:
            return False

    @staticmethod
    @lru_cache
    def _has_vacc_device():
        """
        Use glob command to detect VACC devices.
        DO NOT USE torch to impl this, which will lead to some unexpected errors.
        """
        try:
            import glob

            return len(glob.glob("/dev/vacc*")) > 0
        except Exception:
            return False

    @staticmethod
    @lru_cache
    def _has_musa_device():
        """
        Use pymtml to impl this interface.
        DO NOT USE torch to impl this, which will lead to some unexpected errors.
        """
        try:
            from pymtml import nvmlDeviceGetCount, nvmlInit, nvmlShutdown
        except Exception:
            return False

        device_count = 0
        try:
            nvmlInit()
            device_count = nvmlDeviceGetCount()
        except Exception:
            pass
        finally:
            try:
                nvmlShutdown()
            except Exception:
                pass

        return device_count > 0

    @staticmethod
    @lru_cache
    def _get_cuda_count():
        from ...device_utils import get_available_device_env_name
        from ...utils import cuda_count

        env_name = get_available_device_env_name()
        if env_name is None:
            return cuda_count()

        cuda_visible_devices = os.getenv(env_name, None)
        if cuda_visible_devices is None:
            return cuda_count()

        if cuda_visible_devices == "-1":
            return 0
        else:
            return len(cuda_visible_devices.split(","))

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        lib_result = cls.check_lib()
        if lib_result != True:
            return False
        match_result = cls.match_json(llm_family, llm_spec, quantization)
        return match_result == True

    @classmethod
    @abstractmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        raise NotImplementedError

    def prepare_parse_reasoning_content(
        self, reasoning_content: bool, enable_thinking: bool = True
    ):
        if "hybrid" not in self.model_family.model_ability and not enable_thinking:
            enable_thinking = True
            warnings.warn(
                "enable_thinking cannot be disabled for non hybrid model, will be ignored"
            )
        abilities = self.model_family.model_ability or []
        if self.model_family.model_name == "Kimi-K3":
            self.reasoning_parser = KimiK3ReasoningParser(  # type: ignore
                reasoning_content
            )
            return
        auto_insert_start_tag = "hybrid" not in abilities
        # Initialize reasoning parser if model has reasoning ability
        self.reasoning_parser = ReasoningParser(  # type: ignore
            reasoning_content,
            self.model_family.reasoning_start_tag,  # type: ignore
            self.model_family.reasoning_end_tag,  # type: ignore
            enable_thinking=enable_thinking,
            auto_insert_start_tag=auto_insert_start_tag,
            allow_disable_thinking=(
                getattr(self.model_family, "model_family", None)
                or self.model_family.model_name
            )
            != "glm-5.3",
        )

    def prepare_parse_tool_calls(self):
        if self.model_family.tool_parser is None:
            return
        if self.model_family.tool_parser not in TOOL_PARSERS:
            return
        tool_parser = TOOL_PARSERS[self.model_family.tool_parser]
        self.tool_parser = tool_parser()


# Context variable for passing per-request chat context (e.g., chat_template_kwargs).
# This variable should be set at the beginning of each chat or stream_chat call.
# It allows downstream components (e.g., reasoning_parser) to access request-specific
# settings like 'enable_thinking', without requiring those values to be passed explicitly
# through every function layer.
#
# The context is automatically isolated per thread or coroutine, so concurrent requests
# will not interfere with each other.
chat_context_var: ContextVar[dict] = ContextVar("chat_context_var", default={})


def generate_llm_version_info(llm_family: "LLMFamilyV2") -> Dict[str, List[Dict]]:
    res = defaultdict(list)
    # Use model_specs from huggingface, as HuggingFace is the most comprehensive.
    hf_specs = [
        spec for spec in llm_family.model_specs if spec.model_hub == "huggingface"
    ]
    for spec in hf_specs:
        _llm_family = llm_family.copy()
        _llm_family.model_specs = [spec]
        multimodal_projectors = getattr(spec, "multimodal_projectors", None)
        if multimodal_projectors:
            for mmproj in multimodal_projectors:
                _llm_family.multimodal_projector = mmproj
                res[_llm_family.model_name].append(_llm_family.to_version_info())
        else:
            res[_llm_family.model_name].append(_llm_family.to_version_info())
    return res


def parse_bool_launch_arg(value: object) -> bool:
    """The Web UI submits additional parameters as plain strings, so ``"false"``
    must not be read as a truthy value."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


GEMMA_4_NUM_SPECULATIVE_TOKENS_BY_SIZE = {
    "2": 2,
    "4": 4,
    "12": 4,
    "26": 4,
    "31": 4,
}


def get_model_speculative_tokens_default(
    model_name: object, model_size: object, fallback: int
) -> int:
    """Return the model recipe's speculation depth when one is registered."""
    if model_name == "gemma-4":
        return GEMMA_4_NUM_SPECULATIVE_TOKENS_BY_SIZE.get(str(model_size), fallback)
    return fallback


def parse_num_speculative_tokens(value: object) -> Optional[int]:
    """Validate the engine-neutral speculation depth.

    ``None`` means the launch left it unset, so each engine applies its own
    default. Anything else has to be a positive integer: the Web UI submits
    additional parameters as strings, and a zero or negative depth is a
    contradiction with speculative decoding being enabled — substituting a
    default for it would silently ignore what the caller asked for.
    """
    if value is None or value == "":
        return None
    if isinstance(value, float) and not value.is_integer():
        # truncating 1.5 to 1 would be the same silent substitution
        raise ValueError(
            f"num_speculative_tokens must be a positive integer, got {value!r}"
        )
    try:
        parsed = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        raise ValueError(
            f"num_speculative_tokens must be a positive integer, got {value!r}"
        )
    if parsed <= 0:
        raise ValueError(
            f"num_speculative_tokens must be a positive integer, got {parsed}"
        )
    return parsed


def create_llm_model_instance(
    model_uid: str,
    model_name: str,
    model_engine: Optional[str],
    model_format: Optional[str] = None,
    model_size_in_billions: Optional[Union[int, str]] = None,
    quantization: Optional[str] = None,
    peft_model_config: Optional[PeftModelConfig] = None,
    download_hub: Optional[
        Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> LLM:
    from .cache_manager import LLMCacheManager
    from .llm_family import (
        check_engine_by_spec_parameters,
        check_engine_by_spec_parameters_with_virtual_env,
        match_llm,
    )

    if model_engine is None:
        raise ValueError("model_engine is required for LLM model")
    from .context_length import configure_auto_yarn, map_context_length

    # Validate early, but retain the per-request value while deriving YaRN.
    # llama.cpp's native n_ctx is the total capacity across all parallel slots.
    map_context_length(model_engine, kwargs)
    llm_family = match_llm(
        model_name, model_format, model_size_in_billions, quantization, download_hub
    )

    if not llm_family:
        raise ValueError(
            f"Model not found, name: {model_name}, format: {model_format}, "
            f"size: {model_size_in_billions}, quantization: {quantization}"
        )

    enable_virtual_env = kwargs.pop("enable_virtual_env", None)
    if enable_virtual_env is None:
        from ...constants import XINFERENCE_ENABLE_VIRTUAL_ENV

        enable_virtual_env = XINFERENCE_ENABLE_VIRTUAL_ENV
    if enable_virtual_env:
        llm_cls = check_engine_by_spec_parameters_with_virtual_env(
            model_engine,
            llm_family.model_name,
            llm_family.model_specs[0].model_format,
            llm_family.model_specs[0].model_size_in_billions,
            llm_family.model_specs[0].quantization,
            llm_family=llm_family,
        )
    else:
        llm_cls = check_engine_by_spec_parameters(
            model_engine,
            llm_family.model_name,
            llm_family.model_specs[0].model_format,
            llm_family.model_specs[0].model_size_in_billions,
            llm_family.model_specs[0].quantization,
        )
    logger.debug(f"Launching {model_uid} with {llm_cls.__name__}")

    multimodal_projector = kwargs.get("multimodal_projector")
    if not model_path:
        cache_manager = LLMCacheManager(llm_family, multimodal_projector)
        model_path = cache_manager.cache()

    kwargs, context_env = configure_auto_yarn(
        llm_family, model_engine, kwargs, model_path
    )
    kwargs = map_context_length(model_engine, kwargs)

    # Speculative decoding: download the drafter checkpoint paired with this
    # spec (e.g. Gemma 4 ``*-it-assistant`` for MTP) unless the user points at
    # a local one. ``enable_mtp`` is not forwarded to the engine config.
    enable_mtp = parse_bool_launch_arg(kwargs.pop("enable_mtp", False))
    draft_quantization = kwargs.pop("draft_quantization", None)
    if (
        enable_mtp or kwargs.get("draft_model_path")
    ) and not llm_cls.support_draft_model:
        raise ValueError(
            f"Speculative decoding is not supported by engine {model_engine} "
            f"for model {model_name}."
        )
    if enable_mtp and not kwargs.get("draft_model_path"):
        kwargs["draft_model_path"] = LLMCacheManager(
            llm_family, use_draft_model=True, draft_quantization=draft_quantization
        ).cache()
    if kwargs.get("draft_model_path"):
        logger.info(
            "Launching %s with speculative decoding, drafter: %s",
            model_uid,
            kwargs["draft_model_path"],
        )

    peft_model = peft_model_config.peft_model if peft_model_config else None
    if peft_model is not None:
        if "peft_model" in inspect.signature(llm_cls.__init__).parameters:
            model = llm_cls(
                model_uid,
                llm_family,
                model_path,
                kwargs,
                peft_model,
            )
        else:
            logger.warning(
                f"Model not supported with lora, name: {model_name}, format: {model_format}, engine: {model_engine}. "
                f"Load this without lora."
            )
            model = llm_cls(model_uid, llm_family, model_path, kwargs)
    else:
        model = llm_cls(model_uid, llm_family, model_path, kwargs)
    model._auto_yarn_env = context_env
    return model
