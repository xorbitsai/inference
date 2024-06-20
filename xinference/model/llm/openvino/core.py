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
import os.path
from typing import Dict, Iterable, Iterator, List, Optional, TypedDict, Union

from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChunk,
    CreateCompletionTorch,
    LoRA,
)
from ...utils import select_device
from ..core import LLM
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import ChatModelMixin

logger = logging.getLogger(__name__)


class OpenVINOModelConfig(TypedDict, total=False):
    revision: Optional[str]
    device: str
    gpus: Optional[str]
    num_gpus: int
    max_gpu_memory: str
    gptq_ckpt: Optional[str]
    gptq_wbits: int
    gptq_groupsize: int
    gptq_act_order: bool
    trust_remote_code: bool


class OpenVINOGenerateConfig(TypedDict, total=False):
    temperature: float
    repetition_penalty: float
    top_p: float
    top_k: int
    stream: bool
    max_tokens: int
    echo: bool
    stop: Optional[Union[str, List[str]]]
    stop_token_ids: Optional[Union[int, List[int]]]
    stream_interval: int
    model: Optional[str]
    tools: Optional[List[Dict]]
    lora_name: Optional[str]
    stream_options: Optional[Union[dict, None]]
    request_id: Optional[str]


class OpenVINOModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        model_config: Optional[OpenVINOModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        self._use_fast_tokenizer = True
        self._model_config: OpenVINOModelConfig = self._sanitize_model_config(
            model_config
        )
        if peft_model is not None:
            raise ValueError("OpenVINO engine has not supported lora yet")

    def _sanitize_model_config(
        self, model_config: Optional[OpenVINOModelConfig]
    ) -> OpenVINOModelConfig:
        if model_config is None:
            model_config = OpenVINOModelConfig()
        model_config.setdefault("revision", self.model_spec.model_revision)
        model_config.setdefault("gptq_ckpt", None)
        model_config.setdefault("gptq_wbits", 16)
        model_config.setdefault("gptq_groupsize", -1)
        model_config.setdefault("gptq_act_order", False)
        model_config.setdefault("device", "auto")
        model_config.setdefault("trust_remote_code", True)
        return model_config

    def _sanitize_generate_config(
        self,
        generate_config: Optional[OpenVINOGenerateConfig],
    ) -> OpenVINOGenerateConfig:
        if generate_config is None:
            generate_config = OpenVINOGenerateConfig(**CreateCompletionTorch().dict())
        else:
            # Validate generate_config and fill default values to the generate config.
            generate_config = OpenVINOGenerateConfig(
                **CreateCompletionTorch(**generate_config).dict()
            )
        generate_config["model"] = self.model_uid
        return generate_config

    def _load_model(self, **kwargs):
        try:
            from optimum.intel import OVModelForCausalLM
        except ImportError:
            error_message = "Failed to import module 'optimum'"
            installation_guide = [
                "Please make sure 'optimum' is installed. ",
                "You can install it by `pip install optimum[openvino,nncf]`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=self._use_fast_tokenizer,
            trust_remote_code=kwargs["trust_remote_code"],
            revision=kwargs["revision"],
        )
        ov_path = self._convert_hf_to_ov(self.model_path, kwargs["revision"])
        model = OVModelForCausalLM.from_pretrained(ov_path)
        return model, tokenizer

    def _convert_hf_to_ov(self, model_path: str, revision: str) -> str:
        from optimum.intel import OVModelForCausalLM

        from ..llm_family import _generate_meta_file, valid_model_revision

        root_dir = os.path.dirname(os.path.dirname(model_path))
        ov_dir = os.path.join(
            root_dir,
            "{}-ov-{}b".format(
                self.model_family.model_name, self.model_spec.model_size_in_billions
            ),
        )
        meta_path = os.path.join(ov_dir, "__valid_download")
        if os.path.exists(meta_path):
            logger.info("Skip converting huggingface model to OpenVINO model")
            valid_model_revision(meta_path, revision)
            return ov_dir

        if not os.path.exists(ov_dir):
            os.makedirs(ov_dir)

        logger.info("Convert model to OpenVINO")
        model = OVModelForCausalLM.from_pretrained(model_path, export=True)
        model.save_pretrained(ov_dir)
        _generate_meta_file(
            meta_path, self.model_family, self.model_spec, self.quantization
        )
        return ov_dir

    def load(self):
        device = self._model_config.get("device", "auto")
        self._model_config["device"] = select_device(device)
        self._device = self._model_config["device"]

        kwargs = {}
        kwargs["revision"] = self._model_config.get(
            "revision", self.model_spec.model_revision
        )
        kwargs["trust_remote_code"] = self._model_config.get("trust_remote_code")

        self._model, self._tokenizer = self._load_model(**kwargs)

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if llm_spec.model_format not in ["pytorch"]:
            return False
        if "generate" not in llm_family.model_ability:
            return False
        return True

    def generate(
        self, prompt: str, generate_config: Optional[OpenVINOGenerateConfig] = None
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        from ..pytorch.utils import generate_stream

        def generator_wrapper(
            prompt: str, generate_config: OpenVINOGenerateConfig
        ) -> Iterator[CompletionChunk]:
            for completion_chunk, completion_usage in generate_stream(
                self.model_uid,
                self._model,
                self._tokenizer,
                prompt,
                self._device,
                generate_config,
            ):
                completion_chunk["usage"] = completion_usage
                yield completion_chunk

        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )

        generate_config = self._sanitize_generate_config(generate_config)

        assert self._model is not None
        assert self._tokenizer is not None

        stream = generate_config.get("stream", False)
        if not stream:
            for completion_chunk, completion_usage in generate_stream(
                self.model_uid,
                self._model,
                self._tokenizer,
                prompt,
                self._device,
                generate_config,
            ):
                pass
            completion = Completion(
                id=completion_chunk["id"],
                object=completion_chunk["object"],
                created=completion_chunk["created"],
                model=completion_chunk["model"],
                choices=completion_chunk["choices"],
                usage=completion_usage,
            )
            return completion
        else:
            return generator_wrapper(prompt, generate_config)


class OpenVINOChatModel(OpenVINOModel, ChatModelMixin):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        model_config: Optional[OpenVINOModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        super().__init__(
            model_uid,
            model_family,
            model_spec,
            quantization,
            model_path,
            model_config,
            peft_model,
        )

    def _sanitize_generate_config(
        self,
        generate_config: Optional[OpenVINOGenerateConfig],
    ) -> OpenVINOGenerateConfig:
        generate_config = super()._sanitize_generate_config(generate_config)
        if (
            (not generate_config.get("stop"))
            and self.model_family.prompt_style
            and self.model_family.prompt_style.stop
        ):
            generate_config["stop"] = self.model_family.prompt_style.stop.copy()
        if (
            generate_config.get("stop_token_ids", None) is None
            and self.model_family.prompt_style
            and self.model_family.prompt_style.stop_token_ids
        ):
            generate_config[
                "stop_token_ids"
            ] = self.model_family.prompt_style.stop_token_ids.copy()

        return generate_config

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if llm_spec.model_format not in ["pytorch"]:
            return False
        if "chat" not in llm_family.model_ability:
            return False
        return True

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[OpenVINOGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        tools = generate_config.pop("tools", []) if generate_config else None
        full_prompt = self._get_full_prompt(prompt, system_prompt, chat_history, tools)

        generate_config = self._sanitize_generate_config(generate_config)
        # TODO(codingl2k1): qwen hacky to set stop for function call.
        model_family = self.model_family.model_family or self.model_family.model_name
        if tools and model_family in ["qwen-chat", "qwen1.5-chat"]:
            stop = generate_config.get("stop")
            if isinstance(stop, str):
                generate_config["stop"] = [stop, "Observation:"]
            elif isinstance(stop, Iterable):
                assert not isinstance(stop, str)
                generate_config["stop"] = list(stop) + ["Observation:"]
            else:
                generate_config["stop"] = "Observation:"

        stream = generate_config.get("stream", False)
        if stream:
            it = self.generate(full_prompt, generate_config)
            assert isinstance(it, Iterator)
            return self._to_chat_completion_chunks(it)
        else:
            c = self.generate(full_prompt, generate_config)
            assert not isinstance(c, Iterator)
            if tools:
                return self._tool_calls_completion(
                    self.model_family, self.model_uid, c, tools
                )
            return self._to_chat_completion(c)

    def _get_full_prompt(self, prompt, system_prompt, chat_history, tools):
        assert self.model_family.prompt_style is not None
        prompt_style = self.model_family.prompt_style.copy()
        if system_prompt:
            prompt_style.system_prompt = system_prompt
        chat_history = chat_history or []
        full_prompt = ChatModelMixin.get_prompt(
            prompt, chat_history, prompt_style, tools=tools
        )
        return full_prompt
