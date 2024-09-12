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
import functools
import json
import logging
import os
import time
import typing
import uuid
from io import BytesIO
from typing import AsyncGenerator, Dict, Iterator, List, Optional, Tuple, cast

import requests
from PIL import Image

from ...types import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionChunk,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
)
from .llm_family import (
    LlamaCppLLMSpecV1,
    LLMFamilyV1,
    LLMSpecV1,
    _get_cache_dir,
    get_cache_status,
)

logger = logging.getLogger(__name__)


QWEN_TOOL_CALL_FAMILY = [
    "qwen1.5-chat",
    "qwen1.5-moe-chat",
    "qwen2-instruct",
    "qwen2-moe-instruct",
]

GLM4_TOOL_CALL_FAMILY = [
    "glm4-chat",
    "glm4-chat-1m",
]

QWEN_TOOL_CALL_SYMBOLS = ["<tool_call>", "</tool_call>"]


class ChatModelMixin:
    @staticmethod
    @functools.lru_cache
    def _compile_jinja_template(chat_template):
        """
        Copied from transformers source code.
        """
        try:
            from jinja2.exceptions import TemplateError
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError:
            raise ImportError("xinference requires jinja2 to be installed.")

        def raise_exception(message):
            raise TemplateError(message)

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        return jinja_env.from_string(chat_template)

    def _build_from_raw_template(
        self, messages: List, chat_template: str, **kwargs
    ) -> str:
        compiled_template = self._compile_jinja_template(chat_template)
        rendered = compiled_template.render(
            messages=messages, add_generation_prompt=True, **kwargs
        )
        return rendered

    def get_full_context(
        self, messages: List, chat_template: str, tokenizer=None, **kwargs
    ) -> str:
        if tokenizer is not None:
            try:
                full_context = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    chat_template=chat_template,
                    add_generation_prompt=True,
                    **kwargs,
                )
                return full_context
            except Exception as e:
                logger.warning(
                    f"tokenizer.apply_chat_template error. Maybe this is an old model: {e}"
                )
                return self._build_from_raw_template(messages, chat_template, **kwargs)
        else:
            # build from jinja
            # Compilation function uses a cache to avoid recompiling the same template
            return self._build_from_raw_template(messages, chat_template, **kwargs)

    @staticmethod
    def get_specific_prompt(model_family: str, messages: List[Dict]):
        """
        Inspired by FastChat. Format chat history into a prompt according to the prompty style of
        different models.
        """
        _messages = [x for x in messages]  # copy for not modifying the origin messages
        _messages.append({"role": "assistant", "content": ""})

        if model_family == "internvl2":
            system_prompt = (
                messages[0]["content"] if messages[0]["role"] == "system" else ""
            )
            intra_message_sep = "<|im_end|>"
            ret = (
                "<s>"
                if system_prompt == ""
                else "<s><|im_start|>system\n"
                + system_prompt
                + intra_message_sep
                + "\n"
            )
            images = []  # type: ignore
            for message in _messages:
                role = "<|im_start|>" + message["role"]
                content = message["content"]
                if isinstance(content, str):
                    if content:
                        ret += role + "\n" + content + intra_message_sep + "\n"
                    else:
                        ret += role + "\n"
                elif isinstance(content, list):
                    text = ""
                    image_urls = []
                    for c in content:
                        c_type = c.get("type")
                        if c_type == "text":
                            text = c["text"]
                        elif c_type == "image_url":
                            image_urls.append(c["image_url"]["url"])
                    image_futures = []
                    from concurrent.futures import ThreadPoolExecutor

                    with ThreadPoolExecutor() as executor:
                        for image_url in image_urls:
                            fut = executor.submit(_decode_image, image_url)
                            image_futures.append(fut)
                    images = [fut.result() for fut in image_futures]
                    if len(image_futures) == 0:
                        ret += role + "\n" + text + intra_message_sep + "\n"
                    else:
                        ret += (
                            role + "\n" + f"<image>\n{text}" + intra_message_sep + "\n"
                        )

            return ret, images
        else:
            raise ValueError(f"Invalid model family: {model_family}")

    @classmethod
    def _to_chat_completion_chunk(cls, chunk: CompletionChunk) -> ChatCompletionChunk:
        choices = chunk.get("choices")
        if (
            chunk.get("object") == "chat.completion.chunk"
            and choices
            and "delta" in choices[0]
        ):
            # Already a ChatCompletionChunk, we don't need to convert chunk.
            return cast(ChatCompletionChunk, chunk)
        chat_chunk = {
            "id": "chat" + chunk["id"],
            "model": chunk["model"],
            "created": chunk["created"],
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": i,
                    "delta": {
                        **(
                            {"content": choice["text"]}
                            if ("text" in choice and choice["finish_reason"] is None)
                            else {}
                        ),
                        **(
                            {"tool_calls": choice["tool_calls"]}
                            if "tool_calls" in choice
                            else {}
                        ),
                    },
                    "finish_reason": choice["finish_reason"],
                }
                for i, choice in enumerate(chunk["choices"])
            ],
        }
        return cast(ChatCompletionChunk, chat_chunk)

    @classmethod
    def _get_first_chat_completion_chunk(
        cls, chunk: CompletionChunk
    ) -> ChatCompletionChunk:
        chat_chunk = {
            "id": "chat" + chunk["id"],
            "model": chunk["model"],
            "created": chunk["created"],
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": i,
                    "delta": {
                        "role": "assistant",
                        "content": "",
                    },
                    "finish_reason": None,
                }
                for i, choice in enumerate(chunk["choices"])
            ],
        }
        return cast(ChatCompletionChunk, chat_chunk)

    @classmethod
    def _get_final_chat_completion_chunk(
        cls, chunk: CompletionChunk
    ) -> ChatCompletionChunk:
        chat_chunk = {
            "id": "chat" + chunk["id"],
            "model": chunk["model"],
            "created": chunk["created"],
            "object": "chat.completion.chunk",
            "choices": [],
        }
        usage = chunk.get("usage")
        if usage is not None:
            chat_chunk["usage"] = usage
        return cast(ChatCompletionChunk, chat_chunk)

    @classmethod
    def _to_chat_completion_chunks(
        cls,
        chunks: Iterator[CompletionChunk],
    ) -> Iterator[ChatCompletionChunk]:
        for i, chunk in enumerate(chunks):
            if i == 0:
                yield cls._get_first_chat_completion_chunk(chunk)
            # usage
            choices = chunk.get("choices")
            if not choices:
                yield cls._get_final_chat_completion_chunk(chunk)
            else:
                yield cls._to_chat_completion_chunk(chunk)

    @classmethod
    async def _async_to_chat_completion_chunks(
        cls,
        chunks: AsyncGenerator[CompletionChunk, None],
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        i = 0
        async for chunk in chunks:
            if i == 0:
                yield cls._get_first_chat_completion_chunk(chunk)
            # usage
            choices = chunk.get("choices")
            if not choices:
                yield cls._get_final_chat_completion_chunk(chunk)
            else:
                yield cls._to_chat_completion_chunk(chunk)
            i += 1

    @staticmethod
    def _to_chat_completion(completion: Completion) -> ChatCompletion:
        return {
            "id": "chat" + completion["id"],
            "object": "chat.completion",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": choice["text"],
                    },
                    "finish_reason": choice["finish_reason"],
                }
                for i, choice in enumerate(completion["choices"])
            ],
            "usage": completion["usage"],
        }

    @staticmethod
    def _eval_glm_chat_arguments(c) -> List[Tuple]:
        """
        Currently, glm4 tool call only supports one function
        """
        try:
            if isinstance(c, dict):
                return [(None, c["name"], c["arguments"])]
        except KeyError:
            logger.error("Can't parse glm output: %s", c)
            return [(str(c), None, None)]
        else:
            return [(str(c), None, None)]

    @classmethod
    def _handle_qwen_tool_result(cls, text: str) -> List[Tuple]:
        text: str = text.strip()  # type: ignore
        contents: List[str] = text.split(QWEN_TOOL_CALL_SYMBOLS[1])
        results: List[Tuple] = []
        for content in contents:
            content = content.strip()
            if content:
                if content.startswith(QWEN_TOOL_CALL_SYMBOLS[0]):
                    content = content[len(QWEN_TOOL_CALL_SYMBOLS[0]) :]
                content = content.strip()
                try:
                    res = json.loads(content)
                    results.append((None, res["name"], res["arguments"]))
                except Exception as e:
                    logger.error(
                        "Can't parse single qwen tool call output: %s. Error: %s",
                        content,
                        e,
                    )
                    results.append((content, None, None))
        return results

    @classmethod
    def _eval_qwen_chat_arguments(cls, c) -> List[Tuple]:
        text = c["choices"][0]["text"]
        return cls._handle_qwen_tool_result(text)

    @classmethod
    def _eval_tool_arguments(cls, model_family, c):
        family = model_family.model_family or model_family.model_name
        if family in GLM4_TOOL_CALL_FAMILY:
            result = cls._eval_glm_chat_arguments(c)
        elif family in QWEN_TOOL_CALL_FAMILY:
            result = cls._eval_qwen_chat_arguments(c)
        else:
            raise Exception(
                f"Model {model_family.model_name} is not support tool calls."
            )
        logger.debug(f"Tool call content: {result}")
        return result

    @classmethod
    def _tool_calls_completion_chunk(cls, model_family, model_uid, c):
        _id = str(uuid.uuid4())
        tool_result = cls._eval_tool_arguments(model_family, c)
        tool_calls = []
        failed_contents = []
        for content, func, args in tool_result:
            if func:
                tool_calls.append(
                    [
                        {
                            "id": f"call_{_id}",
                            "type": "function",
                            "function": {
                                "name": func,
                                "arguments": json.dumps(args, ensure_ascii=False),
                            },
                        }
                    ]
                )
            else:
                failed_contents.append(content)
        finish_reason = "tool_calls" if tool_calls else "stop"
        d = {
            "role": "assistant",
            "content": ". ".join(failed_contents) if failed_contents else None,
            "tool_calls": tool_calls,
        }
        try:
            usage = c.get("usage")
            assert "prompt_tokens" in usage
        except Exception:
            usage = {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            }
        return {
            "id": "chat" + f"cmpl-{_id}",
            "model": model_uid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "delta": d,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }

    @classmethod
    def _tool_calls_completion(cls, model_family, model_uid, c):
        _id = str(uuid.uuid4())
        tool_result = cls._eval_tool_arguments(model_family, c)

        tool_calls = []
        failed_contents = []
        for content, func, args in tool_result:
            if func:
                tool_calls.append(
                    {
                        "id": f"call_{_id}",
                        "type": "function",
                        "function": {
                            "name": func,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }
                )
            else:
                failed_contents.append(content)
        finish_reason = "tool_calls" if tool_calls else "stop"
        m = {
            "role": "assistant",
            "content": ". ".join(failed_contents) if failed_contents else None,
            "tool_calls": tool_calls,
        }
        try:
            usage = c.get("usage")
            assert "prompt_tokens" in usage
        except Exception:
            usage = {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            }
        return {
            "id": "chat" + f"cmpl-{_id}",
            "model": model_uid,
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "message": m,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }


def get_file_location(
    llm_family: LLMFamilyV1, spec: LLMSpecV1, quantization: str
) -> Tuple[str, bool]:
    cache_dir = _get_cache_dir(
        llm_family, spec, quantization, create_if_not_exist=False
    )
    cache_status = get_cache_status(llm_family, spec, quantization)
    if isinstance(cache_status, list):
        is_cached = None
        for q, cs in zip(spec.quantizations, cache_status):
            if q == quantization:
                is_cached = cs
                break
    else:
        is_cached = cache_status
    assert isinstance(is_cached, bool)

    if spec.model_format in ["pytorch", "gptq", "awq", "fp8", "mlx"]:
        return cache_dir, is_cached
    elif spec.model_format in ["ggufv2"]:
        assert isinstance(spec, LlamaCppLLMSpecV1)
        filename = spec.model_file_name_template.format(quantization=quantization)
        model_path = os.path.join(cache_dir, filename)
        return model_path, is_cached
    else:
        raise ValueError(f"Not supported model format {spec.model_format}")


def get_model_version(
    llm_family: LLMFamilyV1, llm_spec: LLMSpecV1, quantization: str
) -> str:
    return f"{llm_family.model_name}--{llm_spec.model_size_in_billions}B--{llm_spec.model_format}--{quantization}"


def _decode_image(_url):
    if _url.startswith("data:"):
        logging.info("Parse url by base64 decoder.")
        # https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
        # e.g. f"data:image/jpeg;base64,{base64_image}"
        _type, data = _url.split(";")
        _, ext = _type.split("/")
        data = data[len("base64,") :]
        data = base64.b64decode(data.encode("utf-8"))
        return Image.open(BytesIO(data)).convert("RGB")
    else:
        try:
            response = requests.get(_url)
        except requests.exceptions.MissingSchema:
            return Image.open(_url).convert("RGB")
        else:
            return Image.open(BytesIO(response.content)).convert("RGB")


@typing.no_type_check
def generate_completion_chunk(
    chunk_text: Optional[str],
    finish_reason: Optional[str],
    chunk_id: str,
    model_uid: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    has_choice: bool = True,
    has_content: bool = True,
):
    choices = []
    if has_choice:
        choices.append(
            CompletionChoice(
                text=chunk_text, index=0, logprobs=None, finish_reason=finish_reason
            )
            if has_content
            else CompletionChoice(index=0, logprobs=None, finish_reason=finish_reason)
        )
    return CompletionChunk(
        id=chunk_id,
        object="text_completion",
        created=int(time.time()),
        model=model_uid,
        choices=choices,
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


def generate_completion(
    model_uid: str,
    response: str,
    prompt_tokens=-1,
    completion_tokens=-1,
    total_tokens=-1,
    finish_reason="stop",
) -> Completion:
    return Completion(
        id=str(uuid.uuid1()),
        object="text_completion",
        created=int(time.time()),
        model=model_uid,
        choices=[
            CompletionChoice(
                text=response, index=0, logprobs=None, finish_reason=finish_reason
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


def generate_chat_completion(
    model_uid: str,
    response: str,
    prompt_tokens=-1,
    completion_tokens=-1,
    total_tokens=-1,
    finish_reason="stop",
) -> ChatCompletion:
    return ChatCompletion(
        id="chat" + str(uuid.uuid1()),
        object="chat.completion",
        created=int(time.time()),
        model=model_uid,
        choices=[
            ChatCompletionChoice(
                index=0,
                message={"role": "assistant", "content": response},
                finish_reason=finish_reason,
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


@functools.lru_cache
def get_stop_token_ids_from_config_file(model_path: str) -> Optional[List[int]]:
    from transformers import GenerationConfig as TransformersGenerationConfig

    transformers_config = TransformersGenerationConfig.from_pretrained(model_path)
    if transformers_config.eos_token_id is not None:
        stop_token_ids = (
            transformers_config.eos_token_id
            if isinstance(transformers_config.eos_token_id, list)
            else [transformers_config.eos_token_id]
        )
        return stop_token_ids
    return None


def parse_messages(messages: List[Dict]) -> Tuple:
    """
    Some older models still follow the old way of parameter passing.
    This function helps to parse out the needed information from OpenAI-compatible `messages`.
    """
    system_messages = [mess["content"] for mess in messages if mess["role"] == "system"]
    content_messages = [mess for mess in messages if mess["role"] != "system"]
    prompt = content_messages[-1]["content"]
    system_prompt = ". ".join(system_messages) if system_messages else None
    chat_history = content_messages[:-1]
    return prompt, system_prompt, chat_history
