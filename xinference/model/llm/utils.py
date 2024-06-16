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
import functools
import json
import logging
import os
import time
import uuid
from typing import AsyncGenerator, Dict, Iterator, List, Optional, Tuple, cast

from ...types import (
    SPECIAL_TOOL_PROMPT,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChunk,
)
from .llm_family import (
    GgmlLLMSpecV1,
    LLMFamilyV1,
    LLMSpecV1,
    PromptStyleV1,
    _get_cache_dir,
    get_cache_status,
)

logger = logging.getLogger(__name__)


class ChatModelMixin:
    @staticmethod
    def get_prompt(
        prompt: str,
        chat_history: List[ChatCompletionMessage],
        prompt_style: PromptStyleV1,
        tools: Optional[List[Dict]] = None,
    ) -> str:
        """
        Inspired by FastChat. Format chat history into a prompt according to the prompty style of
        different models.
        """
        assert prompt_style.roles is not None
        if prompt != SPECIAL_TOOL_PROMPT:
            chat_history.append(
                ChatCompletionMessage(role=prompt_style.roles[0], content=prompt)
            )
        chat_history.append(
            ChatCompletionMessage(role=prompt_style.roles[1], content="")
        )

        def get_role(role_name: str):
            if role_name == "user":
                return prompt_style.roles[0]
            elif role_name == "assistant":
                return prompt_style.roles[1]
            else:
                return role_name

        if prompt_style.style_name == "ADD_COLON_SINGLE":
            ret = prompt_style.system_prompt + prompt_style.intra_message_sep
            for message in chat_history:
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += role + ": " + content + prompt_style.intra_message_sep
                else:
                    ret += role + ":"
            return ret
        elif prompt_style.style_name == "ADD_COLON_TWO":
            seps = [prompt_style.intra_message_sep, prompt_style.inter_message_sep]
            ret = prompt_style.system_prompt + seps[0]
            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += role + ": " + content + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif prompt_style.style_name == "NO_COLON_TWO":
            seps = [prompt_style.intra_message_sep, prompt_style.inter_message_sep]
            ret = prompt_style.system_prompt
            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += role + content + seps[i % 2]
                else:
                    ret += role
            return ret
        elif prompt_style.style_name == "LLAMA2":
            seps = [prompt_style.intra_message_sep, prompt_style.inter_message_sep]
            ret = ""
            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    if i == 0:
                        ret += prompt_style.system_prompt + content
                    else:
                        ret += role + " " + content + seps[i % 2]
                else:
                    ret += role
            return ret
        elif prompt_style.style_name == "LLAMA3":
            ret = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                f"{prompt_style.intra_message_sep}{prompt_style.system_prompt}{prompt_style.inter_message_sep}"
            )
            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += (
                        f"<|start_header_id|>{role}<|end_header_id|>"
                        f"{prompt_style.intra_message_sep}{content}{prompt_style.inter_message_sep}"
                    )
                else:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>{prompt_style.intra_message_sep}"
            return ret
        elif prompt_style.style_name == "FALCON":
            ret = prompt_style.system_prompt
            for message in chat_history:
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += (
                        role
                        + ": "
                        + content.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif prompt_style.style_name == "MIXTRAL_V01":
            ret = ""
            for i, message in enumerate(chat_history):
                content = message["content"]
                if i % 2 == 0:  # user
                    ret += f"<s> [INST] {content} [/INST]"
                else:  # assistant
                    ret += f"{content} </s>"
            return ret
        elif prompt_style.style_name == "CHATGLM":
            round_add_n = 1 if prompt_style.intra_message_sep == "\n\n" else 0
            if prompt_style.system_prompt:
                ret = prompt_style.system_prompt + prompt_style.intra_message_sep
            else:
                ret = ""
            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message["content"]
                if i % 2 == 0:
                    ret += f"[Round {i // 2 + round_add_n}]{prompt_style.intra_message_sep}"
                if content:
                    ret += role + "：" + content + prompt_style.intra_message_sep
                else:
                    ret += role + "："
            return ret
        elif prompt_style.style_name == "CHATGLM3":
            prompts = (
                [f"<|system|>\n {prompt_style.system_prompt}"]
                if prompt_style.system_prompt
                else []
            )

            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message.get("content")
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    content = tool_calls[0]["function"]
                if content:
                    if role == "tool":
                        role = "observation"
                    prompts.append(f"<|{role}|>\n {content}")
                else:
                    prompts.append(f"<|{role}|>")
            return "\n".join(prompts)
        elif prompt_style.style_name == "XVERSE":
            ret = (
                f"<|system|> \n {prompt_style.system_prompt}"
                if prompt_style.system_prompt
                else ""
            )
            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += f"<|{role}|> \n {content}"
                else:
                    ret += f"<|{role}|>"
            return ret
        elif prompt_style.style_name == "QWEN":
            if tools:
                tool_desc = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""

                react_instruction = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""
                tools_text = []
                tools_name_text = []
                for func_info in tools:
                    parameters = []
                    fp = func_info["function"].get("parameters", {})
                    if fp:
                        required_parameters = fp.get("required", [])
                        for name, p in fp["properties"].items():
                            param = dict({"name": name}, **p)
                            if name in required_parameters:
                                param["required"] = True
                            parameters.append(param)

                    name = func_info["function"]["name"]
                    desc = func_info["function"]["description"]
                    tool_string = tool_desc.format(
                        name_for_model=name,
                        name_for_human=name,
                        # Hint: You can add the following format requirements in description:
                        #   "Format the arguments as a JSON object."
                        #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                        description_for_model=desc,
                        parameters=json.dumps(parameters, ensure_ascii=False),
                    )
                    tools_text.append(tool_string)
                    tools_name_text.append(name)
                tools_text_string = "\n\n".join(tools_text)
                tools_name_text_string = ", ".join(tools_name_text)
                tool_system = react_instruction.format(
                    tools_text=tools_text_string,
                    tools_name_text=tools_name_text_string,
                )
            else:
                tool_system = ""

            ret = f"<|im_start|>system\n{prompt_style.system_prompt}<|im_end|>"
            for message in chat_history:
                role = get_role(message["role"])
                content = message.get("content")

                ret += prompt_style.intra_message_sep
                if tools:
                    if role == "user":
                        if tool_system:
                            content = tool_system + f"\n\nQuestion: {content}"
                            tool_system = ""
                        else:
                            content = f"Question: {content}"
                    elif role == "assistant":
                        tool_calls = message.get("tool_calls")
                        if tool_calls:
                            func_call = tool_calls[0]["function"]
                            f_name, f_args = (
                                func_call["name"],
                                func_call["arguments"],
                            )
                            content = f"Thought: I can use {f_name}.\nAction: {f_name}\nAction Input: {f_args}"
                        elif content:
                            content = f"Thought: I now know the final answer.\nFinal answer: {content}"
                    elif role == "tool":
                        role = "function"
                        content = f"Observation: {content}"
                    else:
                        raise Exception(f"Unsupported message role: {role}")
                if content:
                    content = content.lstrip("\n").rstrip()
                    ret += f"<|im_start|>{role}\n{content}<|im_end|>"
                else:
                    ret += f"<|im_start|>{role}\n"
            return ret
        elif prompt_style.style_name == "CHATML":
            ret = (
                ""
                if prompt_style.system_prompt == ""
                else prompt_style.system_prompt + prompt_style.intra_message_sep + "\n"
            )
            for message in chat_history:
                role = get_role(message["role"])
                content = message["content"]

                if content:
                    ret += role + "\n" + content + prompt_style.intra_message_sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif prompt_style.style_name == "INTERNLM":
            seps = [prompt_style.intra_message_sep, prompt_style.inter_message_sep]
            ret = ""
            for i, message in enumerate(chat_history[:-2]):
                if i % 2 == 0:
                    ret += "<s>"
                role = get_role(message["role"])
                content = message["content"]
                ret += role + ":" + str(content) + seps[i % 2]
            if len(ret) == 0:
                ret += "<s>"
            ret += (
                chat_history[-2]["role"]
                + ":"
                + str(chat_history[-2]["content"])
                + seps[0]
            )
            ret += chat_history[-1]["role"] + ":"
            return ret
        elif prompt_style.style_name == "INTERNLM2":
            ret = (
                "<s>"
                if prompt_style.system_prompt == ""
                else "<s><|im_start|>system\n"
                + prompt_style.system_prompt
                + prompt_style.intra_message_sep
                + "\n"
            )
            for message in chat_history:
                role = get_role(message["role"])
                content = message["content"]

                if content:
                    ret += role + "\n" + content + prompt_style.intra_message_sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif prompt_style.style_name == "ADD_COLON_SINGLE_COT":
            ret = prompt_style.system_prompt + prompt_style.intra_message_sep
            for message in chat_history:
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += role + ": " + content + prompt_style.intra_message_sep
                else:
                    ret += role + ": Let's think step by step."
            return ret
        elif prompt_style.style_name == "INSTRUCTION":
            message = chat_history[-2]
            return prompt_style.system_prompt.format(message["content"])
        elif prompt_style.style_name == "DEEPSEEK_CHAT":
            seps = [prompt_style.intra_message_sep, prompt_style.inter_message_sep]
            ret = prompt_style.system_prompt
            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += role + ": " + content + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif prompt_style.style_name == "DEEPSEEK_CODER":
            sep = prompt_style.inter_message_sep
            ret = prompt_style.system_prompt + sep
            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += role + "\n" + content + sep
                else:
                    ret += role + "\n"
            return ret
        elif prompt_style.style_name == "GORILLA_OPENFUNCTIONS":
            if tools:
                gorilla_functions = []
                for tool in tools:
                    gorilla_functions.append(
                        {
                            "name": tool["function"]["name"],
                            "api_name": tool["function"]["name"],
                            "description": tool["function"]["description"],
                            "parameters": [
                                dict({"name": name}, **p)
                                for name, p in tool["function"]["parameters"][
                                    "properties"
                                ].items()
                            ],
                        }
                    )
                tools_string = json.dumps(gorilla_functions)
                return f"USER: <<question>> {prompt} <<function>> {tools_string}\nASSISTANT: "
            else:
                return f"USER: <<question>> {prompt}\nASSISTANT: "
        elif prompt_style.style_name == "orion":
            ret = "<s>"
            for i, message in enumerate(chat_history):
                content = message["content"]
                role = get_role(message["role"])
                if i % 2 == 0:  # Human
                    assert content is not None
                    ret += role + ": " + content + "\n\n"
                else:  # Assistant
                    if content:
                        ret += role + ": </s>" + content + "</s>"
                    else:
                        ret += role + ": </s>"
            return ret
        elif prompt_style.style_name == "gemma":
            ret = ""
            for message in chat_history:
                content = message["content"]
                role = get_role(message["role"])
                ret += "<start_of_turn>" + role + "\n"
                if content:
                    ret += content + "<end_of_turn>\n"
            return ret
        elif prompt_style.style_name == "CodeShell":
            ret = ""
            for message in chat_history:
                content = message["content"]
                role = get_role(message["role"])
                if content:
                    ret += f"{role}{content}|<end>|"
                else:
                    ret += f"{role}".rstrip()
            return ret
        elif prompt_style.style_name == "MINICPM-2B":
            ret = ""
            for message in chat_history:
                content = message["content"] or ""
                role = get_role(message["role"])
                if role == "user":
                    ret += "<用户>" + content.strip()
                else:
                    ret += "<AI>" + content.strip()
            return ret
        elif prompt_style.style_name == "PHI3":
            ret = f"<|system|>{prompt_style.intra_message_sep}{prompt_style.system_prompt}{prompt_style.inter_message_sep}"
            for message in chat_history:
                content = message["content"] or ""
                role = get_role(message["role"])
                if content:
                    ret += f"<|{role}|>{prompt_style.intra_message_sep}{content}{prompt_style.inter_message_sep}"
                else:
                    ret += f"<|{role}|>{prompt_style.intra_message_sep}"
            ret += "<|assistant|>\n"
            return ret
        elif prompt_style.style_name == "c4ai-command-r":
            ret = (
                f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>"
                f"{prompt_style.system_prompt}{prompt_style.inter_message_sep}"
            )
            for i, message in enumerate(chat_history):
                role = get_role(message["role"])
                content = message["content"]
                if content:
                    ret += f"{role}{content}{prompt_style.inter_message_sep}"
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid prompt style: {prompt_style.style_name}")

    @classmethod
    def _to_chat_completion_chunk(cls, chunk: CompletionChunk) -> ChatCompletionChunk:
        chat_chunk = {
            "id": "chat" + chunk["id"],
            "model": chunk["model"],
            "created": chunk["created"],
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": i,
                    "delta": {
                        "content": choice["text"],
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
    def _eval_gorilla_openfunctions_arguments(c, tools):
        tool_names = [tool["function"]["name"] for tool in tools]
        arguments = c["choices"][0]["text"]

        def tool_call(n, **kwargs):
            return None, n, kwargs

        try:
            a, b, c = eval(
                arguments, {n: functools.partial(tool_call, n) for n in tool_names}
            )
            return a, b, c
        except Exception as e:
            logger.error("Eval tool calls completion failed: %s", e)
            return arguments, None, None

    @staticmethod
    def _eval_glm_chat_arguments(c, tools):
        if isinstance(c[0], str):
            return c[0], None, None
        return None, c[0]["name"], c[0]["parameters"]

    @staticmethod
    def _eval_qwen_chat_arguments(c, tools):
        text = c["choices"][0]["text"]
        try:
            # Refer to:
            # https://github.com/QwenLM/Qwen/blob/main/examples/react_prompt.md
            # https://github.com/QwenLM/Qwen/blob/main/openai_api.py#L297
            func_name, func_args, content = "", "", ""
            i = text.rfind("\nAction:")
            j = text.rfind("\nAction Input:")
            k = text.rfind("\nObservation:")
            t = max(
                text.rfind("\nThought:", 0, i), text.rfind("Thought:", 0, i)
            )  # find the last thought just before Action, considering the Thought at the very beginning
            if 0 <= i < j:  # If the text has `Action` and `Action input`,
                if k < j:  # but does not contain `Observation`,
                    # then it is likely that `Observation` is omitted by the LLM,
                    # because the output text may have discarded the stop word.
                    text = text.rstrip() + "\nObservation:"  # Add it back.
                    k = text.rfind("\nObservation:")
            if 0 <= t < i < j < k:
                func_name = text[i + len("\nAction:") : j].strip()
                func_args = text[j + len("\nAction Input:") : k].strip()
                content = text[
                    t + len("\nThought:") : i
                ].strip()  # len("\nThought:") and len("Thought:") both are OK since there is a space after :
            if func_name:
                return content, func_name, json.loads(func_args)
        except Exception as e:
            logger.error("Eval tool calls completion failed: %s", e)
        t = max(text.rfind("\nThought:"), text.rfind("Thought:"))
        z = max(text.rfind("\nFinal Answer:"), text.rfind("Final Answer:"))
        if z >= 0:
            text = text[
                z + len("\nFinal Answer:") :
            ]  # len("\nFinal Answer::") and len("Final Answer::") both are OK since there is a space after :
        else:
            text = text[
                t + len("\nThought:") :
            ]  # There is only Thought: no Final Answer:
        return text, None, None

    @classmethod
    def _eval_tool_arguments(cls, model_family, c, tools):
        family = model_family.model_family or model_family.model_name
        if family in ["gorilla-openfunctions-v1", "gorilla-openfunctions-v2"]:
            content, func, args = cls._eval_gorilla_openfunctions_arguments(c, tools)
        elif family in ["chatglm3", "glm4-chat"]:
            content, func, args = cls._eval_glm_chat_arguments(c, tools)
        elif family in ["qwen-chat", "qwen1.5-chat", "qwen2-instruct"]:
            content, func, args = cls._eval_qwen_chat_arguments(c, tools)
        else:
            raise Exception(
                f"Model {model_family.model_name} is not support tool calls."
            )
        logger.debug("Tool call content: %s, func: %s, args: %s", content, func, args)
        return content, func, args

    @classmethod
    def _tools_token_filter(cls, model_family):
        """
        Generates a filter function for Qwen series models to retain outputs after "\nFinal Answer:".

        Returns:
            A function that takes tokens (string output by the model so far) and delta (new tokens added) as input,
            returns the part after "\nFinal Answer:" if found, else returns delta.
        """
        family = model_family.model_family or model_family.model_name
        if family in ["qwen-chat", "qwen1.5-chat"]:
            # Encapsulating function to reset 'found' after each call
            found = False

            def process_tokens(tokens: str, delta: str):
                nonlocal found
                # Once "Final Answer:" is found, future tokens are allowed.
                if found:
                    return delta
                # Check if the token ends with "\nFinal Answer:" and update `found`.
                final_answer_idx = tokens.lower().rfind("\nfinal answer:")
                if final_answer_idx != -1:
                    found = True
                    return tokens[final_answer_idx + len("\nfinal answer:") :]
                return ""

            return process_tokens
        else:
            return lambda tokens, delta: delta

    @classmethod
    def _tool_calls_completion(cls, model_family, model_uid, c, tools):
        _id = str(uuid.uuid4())
        content, func, args = cls._eval_tool_arguments(model_family, c, tools)
        if func:
            m = {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": f"call_{_id}",
                        "type": "function",
                        "function": {
                            "name": func,
                            "arguments": json.dumps(args),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            m = {"role": "assistant", "content": content, "tool_calls": []}
            finish_reason = "stop"
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
    cache_dir = _get_cache_dir(llm_family, spec, create_if_not_exist=False)
    cache_status = get_cache_status(llm_family, spec)
    if isinstance(cache_status, list):
        is_cached = None
        for q, cs in zip(spec.quantizations, cache_status):
            if q == quantization:
                is_cached = cs
                break
    else:
        is_cached = cache_status
    assert isinstance(is_cached, bool)

    if spec.model_format in ["pytorch", "gptq", "awq"]:
        return cache_dir, is_cached
    elif spec.model_format in ["ggmlv3", "ggufv2"]:
        assert isinstance(spec, GgmlLLMSpecV1)
        filename = spec.model_file_name_template.format(quantization=quantization)
        model_path = os.path.join(cache_dir, filename)
        return model_path, is_cached
    else:
        raise ValueError(f"Not supported model format {spec.model_format}")


def get_model_version(
    llm_family: LLMFamilyV1, llm_spec: LLMSpecV1, quantization: str
) -> str:
    return f"{llm_family.model_name}--{llm_spec.model_size_in_billions}B--{llm_spec.model_format}--{quantization}"
