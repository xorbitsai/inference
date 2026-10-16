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
import json
from threading import Thread
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from ....core.model import register_batching_multimodal_models
from ....types import ChatCompletion, ChatCompletionChunk, PytorchGenerateConfig
from ...scheduler.request import InferenceRequest
from ...utils import cache_clean
from ..llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from .core import PytorchChatModel, register_non_default_model
from .direct_chat import PytorchDirectChatMixin


@register_batching_multimodal_models("gemma-4")
@register_transformer
@register_non_default_model(
    "Gemma4ForConditionalGeneration", "Gemma4UnifiedForConditionalGeneration"
)
class Gemma4ChatModel(PytorchDirectChatMixin, PytorchChatModel):
    GEMMA4_ARCHITECTURES = {"Gemma4ForConditionalGeneration"}
    GEMMA4_UNIFIED_ARCHITECTURES = {"Gemma4UnifiedForConditionalGeneration"}
    GEMMA4_MIN_TRANSFORMERS_VERSION = "5.5.0"
    GEMMA4_UNIFIED_MIN_TRANSFORMERS_VERSION = "5.10.0"

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        result = super().check_lib()
        if result is not True:
            return result

        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) < Version(
            cls.GEMMA4_MIN_TRANSFORMERS_VERSION
        ):
            return (
                False,
                f"Gemma-4 requires transformers>={cls.GEMMA4_MIN_TRANSFORMERS_VERSION}",
            )
        return True

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV2", model_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if model_spec.model_format not in ["pytorch", "gptq", "awq", "bnb", "fp4"]:
            return (
                False,
                "Gemma4 transformer supports pytorch/gptq/awq/bnb/fp4 formats only",
            )
        if not model_family.has_architecture(
            *cls.GEMMA4_ARCHITECTURES, *cls.GEMMA4_UNIFIED_ARCHITECTURES
        ):
            return (
                False,
                f"Model architectures {model_family.architectures} are not Gemma-4-it",
            )
        return True

    @classmethod
    def _is_unified_model_spec(cls, model_spec: "LLMSpecV1") -> bool:
        model_id = getattr(model_spec, "model_id", None) or ""
        return "gemma-4-12b" in model_id.lower()

    @classmethod
    def _ensure_model_spec_transformers_version(cls, model_spec: "LLMSpecV1") -> None:
        if not cls._is_unified_model_spec(model_spec):
            return

        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) < Version(
            cls.GEMMA4_UNIFIED_MIN_TRANSFORMERS_VERSION
        ):
            raise ImportError(
                "Gemma-4 unified Transformers backend requires "
                f"transformers>={cls.GEMMA4_UNIFIED_MIN_TRANSFORMERS_VERSION}"
            )

    def _load_model(self, **kwargs):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._ensure_model_spec_transformers_version(self.model_spec)

        processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=kwargs["trust_remote_code"],
            revision=kwargs["revision"],
            padding_side="left",
        )
        tokenizer = processor.tokenizer
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    "Gemma-4 tokenizer requires either a pad token or an EOS token"
                )
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        if self._is_unified_model_spec(self.model_spec):
            from transformers import AutoModelForMultimodalLM

            model_cls = AutoModelForMultimodalLM
        else:
            model_cls = AutoModelForCausalLM
        model = model_cls.from_pretrained(
            self.model_path,
            **kwargs,
        )
        self._processor = processor
        self._device = model.device
        return model, tokenizer

    def _get_full_prompt(self, messages: List[Dict], tools, generate_config: dict):
        return self._transform_messages(messages)

    def _get_processor_chat_template_kwargs(self, generate_config: Optional[Dict]):
        template_kwargs = (
            self._get_chat_template_kwargs_from_generate_config(
                generate_config, getattr(self, "reasoning_parser", None)
            )
            or {}
        )
        tools = (generate_config or {}).get("tools")
        if tools:
            template_kwargs["tools"] = tools
        return template_kwargs

    def get_batching_prefill_compatibility_key(self, req: InferenceRequest) -> str:
        generate_config = req.generate_config or {}
        template_inputs = {
            "tools": generate_config.get("tools"),
            "chat_template_kwargs": generate_config.get("chat_template_kwargs"),
        }
        return json.dumps(template_inputs, sort_keys=True, default=repr)

    def build_inputs_from_messages(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ):
        messages = self._transform_messages(messages)
        template_kwargs = self._get_processor_chat_template_kwargs(generate_config)
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **template_kwargs,
        ).to(self._device)
        return inputs

    def build_generate_kwargs(
        self,
        generate_config: Dict,
    ) -> Dict[str, Any]:
        return dict(
            max_new_tokens=generate_config.get("max_tokens") or 512,
            temperature=generate_config.get("temperature", 1),
        )

    def build_streaming_iter(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ) -> Tuple[Iterator, int]:
        from transformers import TextIteratorStreamer

        inputs = self.build_inputs_from_messages(messages, generate_config)
        configs = self.build_generate_kwargs(generate_config)

        streamer = TextIteratorStreamer(
            self._tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {"streamer": streamer, **inputs, **configs}
        t = Thread(target=self._model.generate, kwargs=gen_kwargs)
        t.start()
        return streamer, len(inputs.input_ids[0])

    @cache_clean
    async def _direct_chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        return self.build_direct_chat_result(messages, generate_config)

    def build_prefill_kwargs(
        self, prompts: List, req_list: List[InferenceRequest]
    ) -> Dict:
        template_kwargs = self._get_processor_chat_template_kwargs(
            req_list[0].generate_config
        )
        inputs = self._processor.apply_chat_template(
            prompts,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            **template_kwargs,
        ).to(self._device)

        for i, r in enumerate(req_list):
            input_ids = inputs["input_ids"][i]
            if "attention_mask" in inputs:
                attention_mask = inputs["attention_mask"][i].bool()
                real_len = int(attention_mask.sum().item())
                r.padding_len = attention_mask.numel() - real_len
                r.extra_kwargs["attention_mask_seq_len"] = real_len
                r.prompt_tokens = input_ids[attention_mask].tolist()
            else:
                r.prompt_tokens = input_ids.tolist()
                r.padding_len = 0

        input_ids = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape
        position_ids = self.build_prefill_position_ids(batch_size, seq_len, req_list)

        return {**inputs, "position_ids": position_ids}
