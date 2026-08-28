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
import uuid
from threading import Thread
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

from ....core.model import register_batching_multimodal_models
from ....types import ChatCompletion, ChatCompletionChunk, CompletionChunk
from ...scheduler.request import InferenceRequest
from ...utils import cache_clean
from ..llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from ..utils import generate_completion, generate_completion_chunk
from .core import PytorchChatModel, register_non_default_model


@register_batching_multimodal_models("gemma-4")
@register_transformer
@register_non_default_model(
    "Gemma4ForConditionalGeneration", "Gemma4UnifiedForConditionalGeneration"
)
class Gemma4ChatModel(PytorchChatModel):
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
        if cls._is_unified_model_spec(model_spec):
            import transformers
            from packaging.version import Version

            if Version(transformers.__version__) < Version(
                cls.GEMMA4_UNIFIED_MIN_TRANSFORMERS_VERSION
            ):
                return (
                    False,
                    "Gemma-4 unified Transformers backend requires "
                    f"transformers>={cls.GEMMA4_UNIFIED_MIN_TRANSFORMERS_VERSION}",
                )
        return True

    @classmethod
    def _is_unified_model_spec(cls, model_spec: "LLMSpecV1") -> bool:
        model_id = getattr(model_spec, "model_id", None) or ""
        return "gemma-4-12b" in model_id.lower()

    def _load_model(self, **kwargs):
        from transformers import AutoModelForCausalLM, AutoProcessor

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

    def build_inputs_from_messages(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ):
        messages = self._transform_messages(messages)
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
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

    def generate_non_streaming(
        self,
        messages: List[Dict],
        generate_config: Optional[Dict] = None,
    ) -> ChatCompletion:
        generate_config = generate_config if generate_config else {}
        tools = generate_config.get("tools", None)
        streamer, prompt_tokens = self.build_streaming_iter(messages, generate_config)
        completion_tokens, total_tokens = 0, 0
        res = ""
        for i, new_text in enumerate(streamer):
            completion_tokens = i
            total_tokens = prompt_tokens + completion_tokens
            res += new_text
        completion = generate_completion(
            self.model_uid,
            res,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens if prompt_tokens != -1 else -1,
            total_tokens=total_tokens if prompt_tokens != -1 else -1,
        )
        if tools and self.tool_parser:
            return self._post_process_completion(
                self.model_family,
                self.model_uid,
                completion,
            )
        return self._to_chat_completion(completion, self.reasoning_parser)

    def generate_streaming(
        self,
        messages: List[Dict],
        generate_config: Optional[Dict] = None,
    ) -> Iterator[CompletionChunk]:
        generate_config = generate_config if generate_config else {}
        tools = generate_config.get("tools", None)
        use_tool_calls = bool(tools and self.tool_parser)
        streamer, prompt_tokens = self.build_streaming_iter(messages, generate_config)
        stream_options = generate_config.pop("stream_options", None)
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )

        completion_id = str(uuid.uuid1())
        completion_tokens, total_tokens = 0, 0
        previous_texts = [""]
        previous_tools_texts = [""]
        tool_call_state = {"seen": False}
        i = 0
        for i, new_text in enumerate(streamer):
            completion_tokens = i
            total_tokens = prompt_tokens + completion_tokens
            completion_chunk = generate_completion_chunk(
                chunk_text=new_text,
                finish_reason=None,
                chunk_id=completion_id,
                model_uid=self.model_uid,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens if prompt_tokens != -1 else -1,
                total_tokens=total_tokens if prompt_tokens != -1 else -1,
                has_choice=True,
                has_content=True,
            )
            if use_tool_calls:
                chat_chunk = self._to_chat_completion_chunk(
                    completion_chunk,
                    self.reasoning_parser,
                    previous_texts,
                    ensure_role=i == 0,
                )
                processed_chunk = self._post_process_completion_chunk(
                    self.model_family,
                    self.model_uid,
                    chat_chunk,
                    previous_texts=previous_tools_texts,
                    tool_call_state=tool_call_state,
                )
                if processed_chunk:
                    yield cast(CompletionChunk, processed_chunk)
            else:
                yield completion_chunk

        completion_chunk = generate_completion_chunk(
            chunk_text=None,
            finish_reason="stop",
            chunk_id=completion_id,
            model_uid=self.model_uid,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens if prompt_tokens != -1 else -1,
            total_tokens=total_tokens if prompt_tokens != -1 else -1,
            has_choice=True,
            has_content=False,
        )
        if use_tool_calls:
            chat_chunk = self._to_chat_completion_chunk(
                completion_chunk,
                self.reasoning_parser,
                previous_texts,
                ensure_role=i == 0,
            )
            processed_chunk = self._post_process_completion_chunk(
                self.model_family,
                self.model_uid,
                chat_chunk,
                previous_texts=previous_tools_texts,
                tool_call_state=tool_call_state,
            )
            if processed_chunk:
                yield cast(CompletionChunk, processed_chunk)
        else:
            yield completion_chunk

        if include_usage:
            yield generate_completion_chunk(
                chunk_text=None,
                finish_reason=None,
                chunk_id=completion_id,
                model_uid=self.model_uid,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens if prompt_tokens != -1 else -1,
                total_tokens=total_tokens if prompt_tokens != -1 else -1,
                has_choice=False,
                has_content=False,
            )

    @cache_clean
    async def _direct_chat(
        self,
        messages: List[Dict],
        generate_config: Optional[Dict] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        stream = generate_config.get("stream", False) if generate_config else False
        return (
            self._to_chat_completion_chunks(
                self.generate_streaming(messages, generate_config)
            )
            if stream
            else self.generate_non_streaming(messages, generate_config)
        )

    def build_prefill_kwargs(
        self, prompts: List, req_list: List[InferenceRequest]
    ) -> Dict:
        inputs = self._processor.apply_chat_template(
            prompts,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
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
