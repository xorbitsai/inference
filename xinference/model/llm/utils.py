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

from typing import Iterator, List

from xinference.model.llm.llm_family import PromptStyleV1

from ...types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChunk,
)


class ChatModelMixin:
    @staticmethod
    def get_prompt(
        prompt: str,
        chat_history: List[ChatCompletionMessage],
        prompt_style: PromptStyleV1,
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
        elif prompt_style.style_name == "QWEN":
            ret = f"<|im_start|>system\n{prompt_style.system_prompt}<|im_end|>"
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
                ret += role + ":" + content + seps[i % 2]
            if len(ret) == 0:
                ret += "<s>"
            ret += (
                chat_history[-2]["role"] + ":" + chat_history[-2]["content"] + seps[0]
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
        else:
            raise ValueError(f"Invalid prompt style: {prompt_style.style_name}")

    @staticmethod
    def _convert_chat_completion_chunks_to_chat(
        chunks: Iterator[CompletionChunk],
    ) -> Iterator[ChatCompletionChunk]:
        for i, chunk in enumerate(chunks):
            if i == 0:
                yield {
                    "id": "chat" + chunk["id"],
                    "model": chunk["model"],
                    "created": chunk["created"],
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            yield {
                "id": "chat" + chunk["id"],
                "model": chunk["model"],
                "created": chunk["created"],
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk["choices"][0]["text"],
                        },
                        "finish_reason": chunk["choices"][0]["finish_reason"],
                    }
                ],
            }

    @staticmethod
    def _convert_text_completion_to_chat(completion: Completion) -> ChatCompletion:
        return {
            "id": "chat" + completion["id"],
            "object": "chat.completion",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion["choices"][0]["text"],
                    },
                    "finish_reason": completion["choices"][0]["finish_reason"],
                }
            ],
            "usage": completion["usage"],
        }


def is_valid_model_name(model_name: str) -> bool:
    import re

    return re.match(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$", model_name) is not None
