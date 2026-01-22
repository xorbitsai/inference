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
import json
import logging
import typing
import uuid
from threading import Thread
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch

from ....types import ChatCompletion, ChatCompletionChunk, LoRA, PytorchGenerateConfig
from ...scheduler.request import InferenceRequest
from ..core import chat_context_var
from ..llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from ..utils import (
    GLM4_TOOL_CALL_FAMILY,
    generate_chat_completion,
    generate_completion_chunk,
)
from .core import PytorchChatModel, PytorchModelConfig, register_non_default_model

logger = logging.getLogger(__name__)


@register_transformer
@register_non_default_model("GlmForCausalLM")
class ChatglmPytorchChatModel(PytorchChatModel):
    GLM4_ARCHITECTURES = {"GlmForCausalLM", "Glm4ForCausalLM", "Glm4MoeForCausalLM"}

    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV2",
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        super().__init__(
            model_uid,
            model_family,
            model_path,
            pytorch_model_config=pytorch_model_config,
            peft_model=peft_model,
        )

    def _get_model_class(self):
        from transformers import AutoModel

        return AutoModel

    def _load_model(self, **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            error_message = "Failed to import module 'transformers'"
            installation_guide = [
                "Please make sure 'transformers' is installed. ",
                "You can install it by `pip install transformers`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=kwargs["trust_remote_code"],
            encode_special_tokens=True,
            revision=kwargs["revision"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **kwargs,
        )
        return model, tokenizer

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if llm_spec.model_format not in ["pytorch", "fp4"]:
            return False, "ChatGLM transformer only supports pytorch/fp4 format"
        if not llm_family.has_architecture(*cls.GLM4_ARCHITECTURES):
            return (
                False,
                f"Model architectures {llm_family.architectures} are not GLM4",
            )
        if "chat" not in llm_family.model_ability:
            return False, "ChatGLM transformer requires chat ability"
        return True

    def _handle_tools(self, messages, generate_config):
        """Convert openai tools to ChatGLM tools."""
        if self.model_family.model_name not in GLM4_TOOL_CALL_FAMILY:
            return None
        if generate_config is None:
            return None
        tools = generate_config.pop("tools", None)
        if tools is None:
            return None
        # Convert an iterable to a list
        tools = list(tools)
        tool_choice = generate_config.pop("tool_choice", "none")
        messages[:] = self._process_messages(
            messages, tools=tools, tool_choice=tool_choice
        )
        return tools

    @staticmethod
    def _process_messages(messages, tools=None, tool_choice="none"):
        # This method is adapted from https://github.com/THUDM/GLM-4/blob/main/basic_demo/openai_api_server.py
        _messages = messages
        processed_messages = []
        msg_has_sys = False

        def _filter_tools(_tool_choice, _tools):
            function_name = _tool_choice.get("function", {}).get("name", None)
            if not function_name:
                return []
            filtered_tools = [
                tool
                for tool in _tools
                if tool.get("function", {}).get("name") == function_name
            ]
            return filtered_tools

        if tool_choice != "none":
            if isinstance(tool_choice, dict):
                tools = _filter_tools(tool_choice, tools)

        if tools:
            processed_messages.append(
                {"role": "system", "content": None, "tools": tools}
            )
            msg_has_sys = True

        if isinstance(tool_choice, dict) and tools:
            processed_messages.append(
                {
                    "role": "assistant",
                    "metadata": tool_choice["function"]["name"],
                    "content": "",
                }
            )

        for m in _messages:
            role, content = m["role"], m["content"] or ""
            tool_calls = m.get("tool_calls")

            if role == "function":
                processed_messages.append({"role": "observation", "content": content})
            elif role == "tool":
                processed_messages.append(
                    {"role": "observation", "content": content, "function_call": True}
                )
            elif role == "assistant":
                if tool_calls:
                    for tool_call in tool_calls:
                        processed_messages.append(
                            {
                                "role": "assistant",
                                "metadata": tool_call.get("function", {}).get("name"),
                                "content": tool_call.get("function", {}).get(
                                    "arguments"
                                ),
                            }
                        )
                else:
                    for response in content.split("\n"):
                        if "\n" in response:
                            metadata, sub_content = response.split("\n", maxsplit=1)
                        else:
                            metadata, sub_content = "", response
                        processed_messages.append(
                            {
                                "role": role,
                                "metadata": metadata,
                                "content": sub_content.strip(),
                            }
                        )
            else:
                if role == "system" and msg_has_sys:
                    msg_has_sys = False
                    continue
                processed_messages.append({"role": role, "content": content})

        if not tools or tool_choice == "none":
            for m in _messages:
                if m["role"] == "system":
                    processed_messages.insert(
                        0, {"role": m["role"], "content": m["content"]}
                    )
                    break
        return processed_messages

    @staticmethod
    @typing.no_type_check
    def _process_response_non_streaming(
        output: str, tools: Union[Dict, List[Dict]] = None, use_tool: bool = False
    ) -> Union[str, dict]:
        """
        Copied from https://github.com/THUDM/GLM-4/blob/main/basic_demo/openai_api_server.py#L150
        """
        import re

        lines = output.strip().split("\n")
        arguments_json = None
        special_tools = ["cogview", "simple_browser"]
        tools = {tool["function"]["name"] for tool in tools} if tools else {}

        # 这是一个简单的工具比较函数，不能保证拦截所有非工具输出的结果，比如参数未对齐等特殊情况。
        ##TODO 如果你希望做更多判断，可以在这里进行逻辑完善。

        if len(lines) >= 2 and lines[1].startswith("{"):
            function_name = lines[0].strip()
            arguments = "\n".join(lines[1:]).strip()
            if function_name in tools or function_name in special_tools:
                try:
                    arguments_json = json.loads(arguments)
                    is_tool_call = True
                except json.JSONDecodeError:
                    is_tool_call = function_name in special_tools

                if is_tool_call and use_tool:
                    content = {
                        "name": function_name,
                        "arguments": json.dumps(
                            (
                                arguments_json
                                if isinstance(arguments_json, dict)
                                else arguments
                            ),
                            ensure_ascii=False,
                        ),
                    }
                    if function_name == "simple_browser":
                        search_pattern = re.compile(
                            r'search\("(.+?)"\s*,\s*recency_days\s*=\s*(\d+)\)'
                        )
                        match = search_pattern.match(arguments)
                        if match:
                            content["arguments"] = json.dumps(
                                {
                                    "query": match.group(1),
                                    "recency_days": int(match.group(2)),
                                },
                                ensure_ascii=False,
                            )
                    elif function_name == "cogview":
                        content["arguments"] = json.dumps(
                            {"prompt": arguments}, ensure_ascii=False
                        )

                    return content
        return output.strip()

    @staticmethod
    def _process_response_streaming(output, tools, end=False):
        # Copy from https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/modeling_chatglm.py
        content = ""
        if not tools and end:
            return None
        for response in output.split("<|assistant|>"):
            if "\n" in response:
                metadata, content = response.split("\n", maxsplit=1)
            else:
                metadata, content = "", response
            if not metadata.strip():
                if tools and any(t.startswith(response) for t in tools) and not end:
                    # Waiting for tool call complete.
                    return None
                content = content.strip()
                content = content.replace("[[训练时间]]", "2023年")
            else:
                if tools and metadata in tools and not end:
                    return None
                metadata = metadata.strip()
                if tools and metadata in tools and end:
                    try:
                        parameters = json.loads(content)
                        content = {"name": metadata.strip(), "arguments": parameters}
                    except json.JSONDecodeError:
                        content = {"name": metadata.strip(), "content": content}
                else:
                    content = {"name": metadata.strip(), "content": content}
        return content

    @torch.inference_mode()
    def _stream_chat(self, inputs, tools, **kwargs):
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        tools = {tool["function"]["name"] for tool in tools} if tools else {}
        generation_kwargs = dict(inputs, streamer=streamer)
        generation_kwargs.update(kwargs)
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        response = ""
        for token in streamer:
            response += token
            if response and response[-1] != "�":
                new_response = self._process_response_streaming(
                    response, tools, end=False
                )
                if new_response is None:
                    continue
                yield new_response
        if tools:
            new_response = self._process_response_streaming(response, tools, end=True)
            if new_response:
                yield new_response

    @staticmethod
    def _get_generate_kwargs(generate_config):
        kwargs: Dict[str, Any] = {}  # type: ignore
        generate_config = generate_config or {}
        temperature = generate_config.get("temperature")
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        top_p = generate_config.get("top_p")
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        max_new_tokens = generate_config.get("max_tokens")
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = int(max_new_tokens)
        else:
            kwargs["max_new_tokens"] = 1024
        do_sample = generate_config.get("do_sample")
        if do_sample is not None:
            kwargs["do_sample"] = bool(do_sample)
        top_k = generate_config.get("top_k")
        if top_k is not None:
            kwargs["top_k"] = top_k
        repetition_penalty = generate_config.get("repetition_penalty")
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = repetition_penalty
        return kwargs

    def chat(  # type: ignore
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        generate_config = generate_config or {}
        kwargs: Dict[str, Any] = self._get_generate_kwargs(generate_config)
        tools = self._handle_tools(messages, generate_config)
        has_tools = tools is not None
        stream = generate_config.get("stream", False)
        stream_options = generate_config.pop("stream_options", None)
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )
        inputs = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            chat_template=self.model_family.chat_template,
            add_generation_prompt=True,
            return_dict=True,
        )
        inputs = inputs.to(self._model.device)

        if not stream:
            with torch.no_grad():
                outputs = self._model.generate(**inputs, **kwargs)
                outputs = outputs[:, inputs["input_ids"].shape[1] :]
                response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                # In some cases, the response starts with `\n`
                if response.startswith("\n"):
                    response = response[1:]
            if has_tools:
                function_call = self._process_response_non_streaming(
                    response, tools, use_tool=True
                )
                return self._post_process_completion(
                    self.model_family, self.model_uid, function_call
                )
            else:
                return generate_chat_completion(self.model_uid, response)
        else:

            def _stream_generator():
                last_chunk_text_length = 0
                chunk_id = "chat-" + str(uuid.uuid1())
                prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
                prompt_tokens = len(inputs["input_ids"][0])
                for chunk_text in self._stream_chat(inputs, tools, **kwargs):
                    if tools and isinstance(chunk_text, dict):
                        yield self._post_process_completion_chunk(
                            self.model_family, self.model_uid, chunk_text
                        )
                        return
                    completion_tokens = completion_tokens + 1
                    total_tokens = prompt_tokens + completion_tokens
                    chunk_text = chunk_text[last_chunk_text_length:]
                    last_chunk_text_length += len(chunk_text)
                    yield generate_completion_chunk(
                        chunk_text,
                        finish_reason=None,
                        chunk_id=chunk_id,
                        model_uid=self.model_uid,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )
                yield generate_completion_chunk(
                    None,
                    finish_reason="stop",
                    chunk_id=chunk_id,
                    model_uid=self.model_uid,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    has_choice=True,
                    has_content=False,
                )
                if include_usage:
                    yield generate_completion_chunk(
                        None,
                        finish_reason=None,
                        chunk_id=chunk_id,
                        model_uid=self.model_uid,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        has_choice=False,
                    )

            return self._to_chat_completion_chunks(_stream_generator())

    def prepare_sanitize_generate_config(self, req: InferenceRequest):
        """
        Set temperature and top_p to 0.8 by default
        """
        raw_config = req.inference_kwargs.get("raw_params", {})
        temperature = raw_config.get("temperature", None)
        if temperature is None:
            raw_config["temperature"] = 0.8
        top_p = raw_config.get("top_p", None)
        if top_p is None:
            raw_config["top_p"] = 0.8

        return raw_config

    def prepare_batch_inference(self, req_list: List[InferenceRequest]):
        super(PytorchChatModel, self).prepare_batch_inference(req_list)
        for r in req_list:
            try:
                if not r.stopped and r.is_prefill:
                    tools = r.generate_config.get("tools", None)
                    tools = list(tools) if tools is not None else None
                    tool_choice = r.generate_config.get("tool_choice", "none")

                    chat_template_kwargs = (
                        self._get_chat_template_kwargs_from_generate_config(
                            r.generate_config, self.reasoning_parser
                        )
                        or {}
                    )
                    chat_context_var.set(chat_template_kwargs)
                    full_context_kwargs = chat_template_kwargs.copy()
                    r.prompt = self._process_messages(
                        r.prompt, tools=tools, tool_choice=tool_choice
                    )
                    assert isinstance(
                        r.prompt, list
                    ), "r.prompt must be a list after processing"
                    r.full_prompt = self.get_full_context(
                        r.prompt,
                        self.model_family.chat_template,  # type: ignore
                        tokenizer=self._tokenizer,
                        **full_context_kwargs,
                    )
                    if tools:
                        r.tools = tools
            except Exception as e:
                logger.exception(f"prepare inference error with {e}")
                r.stopped = True
                r.error_msg = str(e)

    def handle_chat_result_non_streaming(self, req: InferenceRequest):
        if req.tools:
            response = req.completion[0]["choices"][0]["text"]
            usage = req.completion[0]["usage"]
            function_call = self._process_response_non_streaming(
                response, req.tools, use_tool=True
            )
            req.completion[0] = self._post_process_completion(
                self.model_family, self.model_uid, function_call
            )
            req.completion[0]["usage"] = usage
        else:
            req.completion[0] = self._to_chat_completion(req.completion[0])

    def handle_chat_result_streaming(self, req: InferenceRequest):
        results = []
        tools = {tool["function"]["name"] for tool in req.tools} if req.tools else {}
        response = "".join(req.outputs)
        eos_pos = response.find("<eos_stream>")
        if eos_pos != -1:
            response = response[:eos_pos]

        if "<bos_stream>" in req.completion:
            bos_pos = req.completion.index("<bos_stream>")
            results.extend(
                self._get_first_chat_completion_chunk(req.completion[bos_pos + 1])
            )

        if req.stopped:
            if tools:
                new_response = self._process_response_streaming(
                    response, tools, end=True
                )
                if new_response:
                    if isinstance(new_response, dict):  # tool call case
                        chunk_id = [
                            c for c in req.completion if not isinstance(c, str)
                        ][0]["id"]
                        results.append(
                            self._post_process_completion_chunk(
                                self.model_family,
                                self.model_uid,
                                new_response,
                                chunk_id=chunk_id,
                            )
                        )
                    else:  # normal case
                        for c in req.completion:
                            if c == "<bos_stream>":
                                continue
                            elif c == "<eos_stream>":
                                break
                            else:
                                results.append(
                                    self._to_chat_completion_chunk(
                                        c, ensure_role=not results
                                    )
                                )
            else:
                for c in req.completion:
                    if c == "<bos_stream>":
                        continue
                    elif c == "<eos_stream>":
                        break
                    else:
                        results.append(
                            self._to_chat_completion_chunk(c, ensure_role=not results)
                        )
        else:
            if response and response[-1] != "�":
                new_response = self._process_response_streaming(
                    response, tools, end=False
                )
                if new_response is not None:  # normal case
                    for c in req.completion:
                        if c == "<bos_stream>":
                            continue
                        results.append(
                            self._to_chat_completion_chunk(c, ensure_role=not results)
                        )

        if req.stopped and req.include_usage:
            results.append(self._get_final_chat_completion_chunk(req.completion[-1]))
        req.completion = results
