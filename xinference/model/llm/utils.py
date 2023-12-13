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
import time
import uuid
from typing import AsyncGenerator, Dict, Iterator, List, Optional

from xinference.model.llm.llm_family import PromptStyleV1

from ...types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChunk,
)

logger = logging.getLogger()


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
        chat_history.append(
            ChatCompletionMessage(role=prompt_style.roles[0], content=prompt)
        )
        chat_history.append(
            ChatCompletionMessage(role=prompt_style.roles[1], content="")
        )

        if prompt_style.style_name == "ADD_COLON_SINGLE":
            ret = prompt_style.system_prompt + prompt_style.intra_message_sep
            for message in chat_history:
                role = message["role"]
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
                role = message["role"]
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
                role = message["role"]
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
                role = message["role"]
                content = message["content"]
                if content:
                    if i == 0:
                        ret += prompt_style.system_prompt + content
                    else:
                        ret += role + " " + content + seps[i % 2]
                else:
                    ret += role
            return ret
        elif prompt_style.style_name == "FALCON":
            ret = prompt_style.system_prompt
            for message in chat_history:
                role = message["role"]
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
        elif prompt_style.style_name == "CHATGLM":
            round_add_n = 1 if prompt_style.intra_message_sep == "\n\n" else 0
            if prompt_style.system_prompt:
                ret = prompt_style.system_prompt + prompt_style.intra_message_sep
            else:
                ret = ""
            for i, message in enumerate(chat_history):
                role = message["role"]
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
                [f"<|system|>\n{prompt_style.system_prompt}"]
                if prompt_style.system_prompt
                else []
            )

            for i, message in enumerate(chat_history):
                role = message["role"]
                content = message["content"]
                if content:
                    prompts.append(f"<|{role}|>\n{content}")
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
                role = message["role"]
                content = message["content"]
                if content:
                    ret += f"<|{role}|> \n {content}"
                else:
                    ret += f"<|{role}|>"
            return ret
        elif prompt_style.style_name == "QWEN":
            if tools:
                tool_desc = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

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
                    required_parameters = func_info["function"]["parameters"].get(
                        "required", []
                    )
                    for name, p in func_info["function"]["parameters"][
                        "properties"
                    ].items():
                        param = dict({"name": name}, **p)
                        if name in required_parameters:
                            param["required"] = True
                        parameters.append(param)

                    name = func_info["function"]["name"]
                    desc = func_info["function"]["description"]
                    tool = tool_desc.format(
                        name_for_model=name,
                        name_for_human=name,
                        # Hint: You can add the following format requirements in description:
                        #   "Format the arguments as a JSON object."
                        #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                        description_for_model=desc,
                        parameters=json.dumps(parameters, ensure_ascii=False),
                    )
                    tools_text.append(tool)
                    tools_name_text.append(name)
                tools_text = "\n\n".join(tools_text)
                tools_name_text = ", ".join(tools_name_text)
                tool_system = "\n\n" + react_instruction.format(
                    tools_text=tools_text,
                    tools_name_text=tools_name_text,
                )
            else:
                tool_system = ""

            ret = f"<|im_start|>system\n{prompt_style.system_prompt}{tool_system}<|im_end|>"
            for message in chat_history:
                role = message["role"]
                content = message["content"]

                ret += prompt_style.intra_message_sep
                if content:
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
                role = message["role"]
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
                role = message["role"]
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
        elif prompt_style.style_name == "ADD_COLON_SINGLE_COT":
            ret = prompt_style.system_prompt + prompt_style.intra_message_sep
            for message in chat_history:
                role = message["role"]
                content = message["content"]
                if content:
                    ret += role + ": " + content + prompt_style.intra_message_sep
                else:
                    ret += role + ": Let's think step by step."
            return ret
        elif prompt_style.style_name == "INSTRUCTION":
            message = chat_history[-2]
            return prompt_style.system_prompt.format(message["content"])
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
        else:
            raise ValueError(f"Invalid prompt style: {prompt_style.style_name}")

    @classmethod
    def _to_chat_completion_chunk(cls, chunk: CompletionChunk) -> ChatCompletionChunk:
        return {
            "id": "chat" + chunk["id"],
            "model": chunk["model"],
            "created": chunk["created"],
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": i,
                    "delta": {
                        "content": choice["text"],
                    },
                    "finish_reason": choice["finish_reason"],
                }
                for i, choice in enumerate(chunk["choices"])
            ],
        }

    @classmethod
    def _get_first_chat_completion_chunk(
        cls, chunk: CompletionChunk
    ) -> ChatCompletionChunk:
        return {
            "id": "chat" + chunk["id"],
            "model": chunk["model"],
            "created": chunk["created"],
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": i,
                    "delta": {
                        "role": "assistant",
                    },
                    "finish_reason": None,
                }
                for i, choice in enumerate(chunk["choices"])
            ],
        }

    @classmethod
    def _to_chat_completion_chunks(
        cls,
        chunks: Iterator[CompletionChunk],
    ) -> Iterator[ChatCompletionChunk]:
        for i, chunk in enumerate(chunks):
            if i == 0:
                yield cls._get_first_chat_completion_chunk(chunk)
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
            return n, kwargs

        try:
            return eval(
                arguments, {n: functools.partial(tool_call, n) for n in tool_names}
            )
        except Exception as e:
            logger.error("Eval tool calls completion failed: %s", e)
            return arguments, arguments

    @staticmethod
    def _eval_chatglm3_arguments(c, tools):
        return c[0]["name"], c[0]["parameters"]

    @classmethod
    def _tool_calls_completion(cls, model_name, model_uid, c, tools):
        _id = str(uuid.uuid4())
        if model_name == "gorilla-openfunctions-v1":
            func, args = cls._eval_gorilla_openfunctions_arguments(c, tools)
        elif model_name == "chatglm3":
            func, args = cls._eval_chatglm3_arguments(c, tools)
        else:
            raise Exception(f"Model {model_name} is not support tool calls.")

        return {
            "id": "chat" + f"cmpl-{_id}",
            "model": model_uid,
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
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
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        }
