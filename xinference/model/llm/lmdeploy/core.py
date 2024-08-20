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
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Optional, TypedDict, Union

from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionUsage,
    LoRA,
)
from ..core import LLM
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import ChatModelMixin, _decode_image

logger = logging.getLogger(__name__)

LMDEPLOY_SUPPORTED_MODELS: List[str] = []
LMDEPLOY_SUPPORTED_CHAT_MODELS: List[str] = ["internvl2"]


def _message_content_to_intern(content):
    if not isinstance(content, str):
        texts = []
        image_urls = []
        for c in content:
            c_type = c.get("type")
            if c_type == "text":
                texts.append(c["text"])
            elif c_type == "image_url":
                image_urls.append(c["image_url"]["url"])
        image_futures = []
        with ThreadPoolExecutor() as executor:
            for image_url in image_urls:
                fut = executor.submit(_decode_image, image_url)
                image_futures.append(fut)
        images = [fut.result() for fut in image_futures]
        text = " ".join(texts)
        if len(images) == 0:
            return text
        elif len(images) == 1:
            return (text, images[-1])
        else:
            return (text, images)
    return content


def _get_prompt_and_chat_history(
    prompt: Union[str, List[Dict]],
    chat_history: Optional[List[ChatCompletionMessage]] = None,
):
    # Convert openai history to intern vl history
    history = []
    for h1, h2 in zip(*[iter(chat_history or [])] * 2):
        content1 = _message_content_to_intern(h1["content"])
        content2 = _message_content_to_intern(h2["content"])
        history.append((content1, content2))

    question = _message_content_to_intern(prompt)
    return question, history


class LMDEPLOYModelConfig(TypedDict, total=False):
    model_format: Optional[str]
    tp: Optional[int]
    session_len: Optional[int]
    max_batch_size: Optional[int]
    cache_max_entry_count: Optional[float]
    cache_block_seq_len: Optional[int]
    enable_prefix_caching: Optional[bool]
    quant_policy: Optional[int]
    rope_scaling_factor: Optional[float]
    use_logn_attn: Optional[bool]
    download_dir: Optional[str]
    revision: Optional[str]
    max_prefill_token_num: Optional[int]
    num_tokens_per_iter: Optional[int]
    max_prefill_iters: Optional[int]


class LMDEPLOYGenerateConfig(TypedDict, total=False):
    n: Optional[int]
    max_new_tokens: Optional[int]
    top_p: Optional[float]
    top_k: Optional[int]
    temperature: Optional[float]
    repetition_penalty: Optional[float]
    ignore_eos: Optional[bool]
    random_seed: Optional[int]
    stop_words: Optional[List[str]]
    bad_words: Optional[List[str]]
    min_new_tokens: Optional[int]
    skip_special_tokens: Optional[bool]
    logprobs: Optional[int]


class LMDEPLOYModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        model_config: Optional[LMDEPLOYModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        self._model_config: LMDEPLOYModelConfig = self._sanitize_model_config(
            model_config
        )
        if peft_model is not None:
            raise ValueError("LMDEPLOY engine has not supported lora yet.")

    def _sanitize_model_config(
        self, model_config: Optional[LMDEPLOYModelConfig]
    ) -> LMDEPLOYModelConfig:
        if model_config is None:
            model_config = LMDEPLOYModelConfig()
        model_config.setdefault("session_len", 8192)
        return model_config

    def load(self):
        try:
            import lmdeploy  # noqa: F401
        except ImportError:
            error_message = "Failed to import module 'lmdeploy'"
            installation_guide = [
                "Please make sure 'lmdeploy' is installed. ",
                "You can install it by `pip install lmdeploy`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
        raise ValueError("LMDEPLOY engine has not supported generate yet.")

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if llm_spec.model_format == "awq":
            # Currently, only 4-bit weight quantization is supported for AWQ, but got 8 bits.
            if "4" not in quantization:
                return False
        if llm_family.model_name not in LMDEPLOY_SUPPORTED_MODELS:
            return False
        return True

    def generate(
        self,
        prompt: str,
        generate_config: Optional[Dict] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        pass


class LMDEPLOYChatModel(LMDEPLOYModel, ChatModelMixin):
    def load(self):
        try:
            from lmdeploy import ChatTemplateConfig, TurbomindEngineConfig, pipeline
        except ImportError:
            error_message = "Failed to import module 'lmdeploy'"
            installation_guide = [
                "Please make sure 'lmdeploy' is installed. ",
                "You can install it by `pip install lmdeploy`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        chat_template_config = ChatTemplateConfig("internvl-internlm2")
        chat_template_config.meta_instruction = (
            self.model_family.prompt_style.system_prompt
        )
        self._model = pipeline(
            self.model_path,
            chat_template_config=chat_template_config,
            backend_config=TurbomindEngineConfig(**self._model_config),
        )

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if llm_spec.model_format == "awq":
            # Currently, only 4-bit weight quantization is supported for AWQ, but got 8 bits.
            if "4" not in quantization:
                return False
        if llm_family.model_name not in LMDEPLOY_SUPPORTED_CHAT_MODELS:
            return False
        return True

    async def async_chat(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[Dict] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        from lmdeploy.serve.async_engine import Session

        question, history = _get_prompt_and_chat_history(prompt, chat_history)

        session = Session()
        session._engine = self._model.engine
        session.history = history
        print(f"input session:{session}")

        # gen_config = GenerationConfig(**generate_config)
        sess = self._model.chat(question, session=session)

        chunk = Completion(
            id=str(uuid.uuid1()),
            object="text_completion",
            created=int(time.time()),
            model=self.model_uid,
            choices=[
                CompletionChoice(
                    index=0,
                    text=sess.response.text,
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=-1, completion_tokens=-1, total_tokens=-1
            ),
        )
        return self._to_chat_completion(chunk)
