# Copyright 2022-2023 XProbe Inc.
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
import base64
import logging
import operator
import tempfile
import typing
import uuid
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer

from ....core.scheduler import InferenceRequest
from ....model.utils import select_device
from ....types import ChatCompletion, ChatCompletionChunk, CompletionChunk
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import generate_chat_completion, generate_completion_chunk
from .core import PytorchChatModel, PytorchGenerateConfig
from .utils import pad_prefill_tokens

logger = logging.getLogger(__name__)


class QwenVLChatModel(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = None
        self._model = None
        self._device = None

    @classmethod
    def match(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        llm_family = model_family.model_family or model_family.model_name
        if "qwen-" in llm_family and "vision" in model_family.model_ability:
            return True
        return False

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig

        if self._check_tensorizer_integrity():
            self._model, self._tokenizer = self._load_tensorizer(
                code_revision=self.model_spec.model_revision
            )
            self._apply_lora()
            return

        device = self._pytorch_model_config.get("device", "auto")
        device = select_device(device)
        self._device = device
        # for multiple GPU, set back to auto to make multiple devices work
        device = "auto" if device == "cuda" else device

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            code_revision=self.model_spec.model_revision,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device,
            trust_remote_code=True,
            code_revision=self.model_spec.model_revision,
        ).eval()

        # Specify hyperparameters for generation
        self._model.generation_config = GenerationConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            code_revision=self.model_spec.model_revision,
        )
        self._apply_lora()
        self._save_tensorizer(code_revision=self.model_spec.model_revision)

    def _message_content_to_qwen(self, content) -> str:
        def _ensure_url(_url):
            if _url.startswith("data:"):
                logging.info("Parse url by base64 decoder.")
                # https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
                # e.g. f"data:image/jpeg;base64,{base64_image}"
                _type, data = _url.split(";")
                _, ext = _type.split("/")
                data = data[len("base64,") :]
                data = base64.b64decode(data.encode("utf-8"))

                with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
                    f.write(data)
                logging.info("Dump base64 data to %s", f.name)
                return f.name
            else:
                if len(_url) > 2048:
                    raise Exception(f"Image url is too long, {len(_url)} > 2048.")
                return _url

        if not isinstance(content, str):
            # TODO(codingl2k1): Optimize _ensure_url
            content = [
                (
                    {"image": _ensure_url(c["image_url"]["url"]), "type": "image"}
                    if c.get("type") == "image_url"
                    else c
                )
                for c in content
            ]
            content = sorted(content, key=operator.itemgetter("type"))
            return self._tokenizer.from_list_format(content)
        return content

    def _get_prompt_and_chat_history(self, messages: List[Dict]):
        qwen_history = []
        query_to_response: List = []
        for message in messages[:-1]:
            role = message["role"]
            content = self._message_content_to_qwen(message["content"])
            if len(query_to_response) == 0 and role == "user":
                query_to_response.append(content)
            if len(query_to_response) == 1 and role == "assistant":
                query_to_response.append(content)
            if len(query_to_response) == 2:
                qwen_history.append(query_to_response)
                query_to_response = []
        prompt = self._message_content_to_qwen(messages[-1]["content"])
        return prompt, qwen_history

    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        prompt, qwen_history = self._get_prompt_and_chat_history(messages)

        stream = generate_config.get("stream", False) if generate_config else False
        stream_options = (
            generate_config.pop("stream_options", None) if generate_config else None
        )
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )
        if stream:
            it = self._generate_stream(prompt, qwen_history, include_usage)  # type: ignore
            return self._to_chat_completion_chunks(it)
        else:
            return self._generate(prompt, qwen_history)  # type: ignore

    def _generate(self, prompt: str, qwen_history: List) -> ChatCompletion:
        response, history = self._model.chat(
            self._tokenizer, query=prompt, history=qwen_history
        )
        return generate_chat_completion(self.model_uid, response)

    def _generate_stream(
        self, prompt: str, qwen_history: List, include_usage
    ) -> Iterator[CompletionChunk]:
        response_generator = self._model.chat_stream(
            self._tokenizer, query=prompt, history=qwen_history
        )
        completion_id = str(uuid.uuid1())
        prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
        input_ids = self._tokenizer(prompt, allowed_special="all").input_ids
        prompt_tokens = len(input_ids)
        full_response = ""
        for response in response_generator:
            inc_content = response[len(full_response) :]
            full_response = response
            completion_tokens = completion_tokens + 1
            total_tokens = prompt_tokens + completion_tokens
            yield generate_completion_chunk(
                chunk_text=inc_content,
                finish_reason=None,
                chunk_id=completion_id,
                model_uid=self.model_uid,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        yield generate_completion_chunk(
            chunk_text=None,
            finish_reason="stop",
            chunk_id=completion_id,
            model_uid=self.model_uid,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            has_choice=True,
            has_content=False,
        )
        if include_usage:
            yield generate_completion_chunk(
                chunk_text=None,
                finish_reason=None,
                chunk_id=completion_id,
                model_uid=self.model_uid,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                has_choice=False,
                has_content=False,
            )

    @staticmethod
    def get_batch_size_and_seq_len_indexes_from_kv() -> Tuple[int, int]:
        """
        Qwen-vl is very special for its kv_cache impl.
        Its dimension is `bs * seq_len * head_num * dim`.
        See https://huggingface.co/Qwen/Qwen-VL-Chat/blob/main/modeling_qwen.py
        """
        return 0, 1

    @staticmethod
    @typing.no_type_check
    def make_context(
        tokenizer: PreTrainedTokenizer,
        query: str,
        history: List[Tuple[str, str]] = None,
        system: str = "",
        max_window_size: int = 6144,
        chat_format: str = "chatml",
    ):
        """
        This function is from https://huggingface.co/Qwen/Qwen-VL-Chat/blob/main/qwen_generation_utils.py.
        Use this function to get input_ids with image.
        """
        if history is None:
            history = []

        if chat_format == "chatml":
            im_start, im_end = "<|im_start|>", "<|im_end|>"
            im_start_tokens = [tokenizer.im_start_id]
            im_end_tokens = [tokenizer.im_end_id]
            nl_tokens = tokenizer.encode("\n")

            def _tokenize_str(role, content):
                return f"{role}\n{content}", tokenizer.encode(
                    role, allowed_special=set(tokenizer.IMAGE_ST)
                ) + nl_tokens + tokenizer.encode(
                    content, allowed_special=set(tokenizer.IMAGE_ST)
                )

            system_text, system_tokens_part = _tokenize_str("system", system)
            system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

            raw_text = ""
            context_tokens = []

            for turn_query, turn_response in reversed(history):
                query_text, query_tokens_part = _tokenize_str("user", turn_query)
                query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
                if turn_response is not None:
                    response_text, response_tokens_part = _tokenize_str(
                        "assistant", turn_response
                    )
                    response_tokens = (
                        im_start_tokens + response_tokens_part + im_end_tokens
                    )

                    next_context_tokens = (
                        nl_tokens + query_tokens + nl_tokens + response_tokens
                    )
                    prev_chat = f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
                else:
                    next_context_tokens = nl_tokens + query_tokens + nl_tokens
                    prev_chat = f"\n{im_start}{query_text}{im_end}\n"

                current_context_size = (
                    len(system_tokens) + len(next_context_tokens) + len(context_tokens)
                )
                if current_context_size < max_window_size:
                    context_tokens = next_context_tokens + context_tokens
                    raw_text = prev_chat + raw_text
                else:
                    break

            context_tokens = system_tokens + context_tokens
            raw_text = f"{im_start}{system_text}{im_end}" + raw_text
            context_tokens += (
                nl_tokens
                + im_start_tokens
                + _tokenize_str("user", query)[1]
                + im_end_tokens
                + nl_tokens
                + im_start_tokens
                + tokenizer.encode("assistant")
                + nl_tokens
            )
            raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

        elif chat_format == "raw":
            raw_text = query
            context_tokens = tokenizer.encode(raw_text)
        else:
            raise NotImplementedError(f"Unknown chat format {chat_format!r}")

        return raw_text, context_tokens

    def _get_full_prompt(self, messages: List[Dict], tools):
        prompt, qwen_history = self._get_prompt_and_chat_history(messages)
        _, context_tokens = self.make_context(self._tokenizer, prompt, qwen_history)
        return context_tokens

    def prepare_sanitize_generate_config(self, req: InferenceRequest):
        """
        Refer to https://huggingface.co/Qwen/Qwen-VL-Chat/blob/main/generation_config.json
        """
        raw_config = req.inference_kwargs.get("raw_params", {})
        top_p = raw_config.get("top_p", None)
        if top_p is None:
            raw_config["top_p"] = 0.3
        top_k = raw_config.get("top_k", None)
        if top_k is None:
            raw_config["top_k"] = 0
        return raw_config

    def build_prefill_inputs(self, prompts: List, req_list: List[InferenceRequest]):
        context_len = self.get_context_len()
        inputs = pad_prefill_tokens(prompts, context_len, req_list)
        input_ids = torch.as_tensor(
            pad_prefill_tokens(inputs, context_len, req_list), device=self._device
        )
        return input_ids

    def build_prefill_position_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        Qwen-vl fill `1` for position_ids padding
        """
        res = []
        for r in reqs:
            real_seq_len = seq_length - r.padding_len
            res.append(
                torch.cat(
                    [
                        torch.full((r.padding_len,), 1, dtype=torch.long),
                        torch.arange(0, real_seq_len, dtype=torch.long),
                    ]
                )
            )
            r.extra_kwargs["max_position_id"] = real_seq_len - 1
        return torch.stack(res).to(self._device)
