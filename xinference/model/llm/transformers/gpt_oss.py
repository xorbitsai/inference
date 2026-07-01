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
import inspect
import logging
from typing import Dict, Iterator, List, Optional, Tuple, Union

from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    PytorchGenerateConfig,
    PytorchModelConfig,
)
from ..harmony import async_stream_harmony_chat_completion
from ..llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from .core import PytorchChatModel, register_non_default_model

logger = logging.getLogger(__name__)


@register_transformer
@register_non_default_model("GptOssForCausalLM")
class GPTOSSPytorchChatModel(PytorchChatModel):
    GPT_OSS_ARCHITECTURES = {"GptOssForCausalLM"}

    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        config = super()._sanitize_model_config(pytorch_model_config)
        config.setdefault("torch_dtype", "auto")
        return config  # type:ignore

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if llm_spec.model_format not in ["pytorch", "gptq", "awq", "bnb", "fp4"]:
            return (
                False,
                "GPT-OSS transformer supports pytorch/gptq/awq/bnb/fp4 formats only",
            )
        if not llm_family.has_architecture(*cls.GPT_OSS_ARCHITECTURES):
            return (
                False,
                f"Model architectures {llm_family.architectures} are not GPT-OSS",
            )
        if "chat" not in llm_family.model_ability:
            return False, "GPT-OSS transformer requires chat ability"
        return True

    async def chat(  # type:ignore
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        gen = super().chat(messages, generate_config=generate_config)

        if inspect.iscoroutine(gen):
            gen = await gen

        if inspect.isasyncgen(gen):
            # Streaming
            async def stream_parser():
                full_text = ""
                full_reasoning = ""

                async for parsed_chunk in async_stream_harmony_chat_completion(gen):
                    choices = parsed_chunk.get("choices")
                    if choices and len(choices) > 0:
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            full_text += delta["content"]
                        if delta.get("reasoning_content"):
                            full_reasoning += delta["reasoning_content"]
                    yield parsed_chunk

                logger.debug(
                    "Chat finished, content: %r, reasoning: %r",
                    full_text,
                    full_reasoning,
                )

            return stream_parser()

        else:
            # Non-streaming sync - handle single result
            async for parsed_completion in async_stream_harmony_chat_completion(gen):  # type: ignore
                return parsed_completion
