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
from typing import Iterator, List, Optional, TypedDict, Union

from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChunk,
    Embedding,
    EmbeddingData,
    EmbeddingUsage,
)
from ..core import LLM
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import ChatModelMixin

logger = logging.getLogger(__name__)


class PytorchGenerateConfig(TypedDict, total=False):
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


class PytorchModelConfig(TypedDict, total=False):
    revision: Optional[str]
    device: str
    gpus: Optional[str]
    num_gpus: int
    max_gpu_memory: str
    gptq_ckpt: Optional[str]
    gptq_wbits: int
    gptq_groupsize: int
    gptq_act_order: bool


class PytorchModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
    ):
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        self._use_fast_tokenizer = True
        self._pytorch_model_config: PytorchModelConfig = self._sanitize_model_config(
            pytorch_model_config
        )

    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        if pytorch_model_config is None:
            pytorch_model_config = PytorchModelConfig()
        pytorch_model_config.setdefault("revision", self.model_spec.model_revision)
        pytorch_model_config.setdefault("gpus", None)
        pytorch_model_config.setdefault("num_gpus", 1)
        pytorch_model_config.setdefault("gptq_ckpt", None)
        pytorch_model_config.setdefault("gptq_wbits", 16)
        pytorch_model_config.setdefault("gptq_groupsize", -1)
        pytorch_model_config.setdefault("gptq_act_order", False)
        pytorch_model_config.setdefault("device", "auto")
        return pytorch_model_config

    def _sanitize_generate_config(
        self,
        pytorch_generate_config: Optional[PytorchGenerateConfig],
    ) -> PytorchGenerateConfig:
        if pytorch_generate_config is None:
            pytorch_generate_config = PytorchGenerateConfig()
        pytorch_generate_config.setdefault("temperature", 0.7)
        pytorch_generate_config.setdefault("repetition_penalty", 1.0)
        pytorch_generate_config.setdefault("max_tokens", 512)
        pytorch_generate_config.setdefault("stream_interval", 2)
        pytorch_generate_config["model"] = self.model_uid
        return pytorch_generate_config

    def _load_model(self, kwargs: dict):
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
            use_fast=self._use_fast_tokenizer,
            trust_remote_code=True,
            revision=kwargs["revision"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        return model, tokenizer

    def load(self):
        try:
            import torch
        except ImportError:
            raise ImportError(
                f"Failed to import module 'torch'. Please make sure 'torch' is installed.\n\n"
            )
        from .compression import load_compress_model

        quantization = self.quantization
        num_gpus = self._pytorch_model_config.get("num_gpus", 1)
        device = self._pytorch_model_config.get("device", "auto")
        self._pytorch_model_config["device"] = self._select_device(device)
        self._device = self._pytorch_model_config["device"]

        if self._device == "cpu":
            kwargs = {"torch_dtype": torch.float32}
        elif self._device == "cuda":
            kwargs = {"torch_dtype": torch.float16}
        elif self._device == "mps":
            kwargs = {"torch_dtype": torch.float16}
        else:
            raise ValueError(f"Device {self._device} is not supported in temporary")

        kwargs["revision"] = self._pytorch_model_config.get(
            "revision", self.model_spec.model_revision
        )

        if quantization != "none":
            if self._device == "cuda" and self._is_linux():
                kwargs["device_map"] = "auto"
                if quantization == "4-bit":
                    kwargs["load_in_4bit"] = True
                    kwargs["bnb_4bit_compute_dtype"] = torch.float16
                    kwargs["bnb_4bit_use_double_quant"] = True
                    kwargs["llm_int8_skip_modules"] = [
                        "lm_head",
                        "encoder",
                        "EncDecAttention",
                    ]
                elif quantization == "8-bit":
                    kwargs["load_in_8bit"] = True
                else:
                    raise ValueError(
                        f"Quantization {quantization} is not supported in temporary"
                    )
            else:
                if num_gpus != 1:
                    raise ValueError(f"Quantization is not supported for multi-gpu")
                elif quantization != "8-bit":
                    raise ValueError(
                        f"Only 8-bit quantization is supported if it is not linux system or cuda device"
                    )
                else:
                    self._model, self._tokenizer = load_compress_model(
                        model_path=self.model_path,
                        device=self._device,
                        torch_dtype=kwargs["torch_dtype"],
                        use_fast=self._use_fast_tokenizer,
                        revision=kwargs["revision"],
                    )
                    logger.debug(f"Model Memory: {self._model.get_memory_footprint()}")
                    return

        self._model, self._tokenizer = self._load_model(kwargs)

        if (
            self._device == "cuda" and num_gpus == 1 and quantization == "none"
        ) or self._device == "mps":
            self._model.to(self._device)
        logger.debug(f"Model Memory: {self._model.get_memory_footprint()}")

    def _select_device(self, device: str) -> str:
        try:
            import torch
        except ImportError:
            raise ImportError(
                f"Failed to import module 'torch'. Please make sure 'torch' is installed.\n\n"
            )

        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("cuda is unavailable in your environment")
        elif device == "mps":
            if not torch.backends.mps.is_available():
                raise ValueError("mps is unavailable in your environment")
        elif device == "cpu":
            pass
        else:
            raise ValueError(f"Device {device} is not supported in temporary")
        return device

    @classmethod
    def match(cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1") -> bool:
        if llm_spec.model_format != "pytorch":
            return False
        if llm_family.model_name in [
            "baichuan-chat",
            "vicuna-v1.3",
            "falcon",
            "falcon-instruct",
            "chatglm",
            "chatglm2",
            "chatglm2-32k",
            "llama-2",
            "llama-2-chat",
        ]:
            return False
        if "generate" not in llm_family.model_ability:
            return False
        return True

    def generate(
        self, prompt: str, generate_config: Optional[PytorchGenerateConfig] = None
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        from .utils import (
            generate_stream,
            generate_stream_chatglm,
            generate_stream_falcon,
        )

        def generator_wrapper(
            prompt: str, generate_config: PytorchGenerateConfig
        ) -> Iterator[CompletionChunk]:
            if "falcon" in self.model_family.model_name:
                for completion_chunk, _ in generate_stream_falcon(
                    self._model, self._tokenizer, prompt, self._device, generate_config
                ):
                    yield completion_chunk
            elif "chatglm" in self.model_family.model_name:
                for completion_chunk, _ in generate_stream_chatglm(
                    self._model, self._tokenizer, prompt, self._device, generate_config
                ):
                    yield completion_chunk
            else:
                for completion_chunk, _ in generate_stream(
                    self._model, self._tokenizer, prompt, self._device, generate_config
                ):
                    yield completion_chunk

        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )

        generate_config = self._sanitize_generate_config(generate_config)

        assert self._model is not None
        assert self._tokenizer is not None

        stream = generate_config.get("stream", False)
        if not stream:
            if "falcon" in self.model_family.model_name:
                for completion_chunk, completion_usage in generate_stream_falcon(
                    self._model, self._tokenizer, prompt, self._device, generate_config
                ):
                    pass
            elif "chatglm" in self.model_family.model_name:
                for completion_chunk, completion_usage in generate_stream_chatglm(
                    self._model, self._tokenizer, prompt, self._device, generate_config
                ):
                    pass
            else:
                for completion_chunk, completion_usage in generate_stream(
                    self._model, self._tokenizer, prompt, self._device, generate_config
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

    def create_embedding(self, input: Union[str, List[str]]) -> Embedding:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as e:
            raise ImportError(
                "Could not import torch. Please install it with `pip install torch`."
            ) from e

        if isinstance(input, str):
            inputs = [input]
        else:
            inputs = input

        tokenizer = self._tokenizer
        is_llama = "llama" in str(type(self._model))  # llama supports batch inference
        is_chatglm = "chatglm" in str(type(self._model))
        if is_llama:
            encoding = tokenizer.batch_encode_plus(
                inputs, padding=True, return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(self._device)
            attention_mask = encoding["attention_mask"].to(self._device)
            model_output = self._model(
                input_ids, attention_mask, output_hidden_states=True
            )
            data = model_output.hidden_states[-1]
            mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
            masked_embeddings = data * mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            seq_length = torch.sum(mask, dim=1)
            embedding = sum_embeddings / seq_length
            normalized_embeddings = F.normalize(embedding, p=2, dim=1)
            normalized_embeddings = normalized_embeddings.tolist()
            token_num = torch.sum(attention_mask).item()

            embedding_list = []
            for index, data in enumerate(normalized_embeddings):
                embedding_list.append(
                    EmbeddingData(index=index, object="embedding", embedding=data)
                )

            usage = EmbeddingUsage(prompt_tokens=token_num, total_tokens=token_num)

            ret = Embedding(
                object="list",
                model=self.model_uid,
                data=embedding_list,
                usage=usage,
            )

        else:
            embedding = []
            token_num = 0
            for index, text in enumerate(inputs):
                input_ids = tokenizer.encode(text, return_tensors="pt").to(self._device)
                model_output = self._model(input_ids, output_hidden_states=True)
                if is_chatglm:
                    data = (model_output.hidden_states[-1].transpose(0, 1))[0]
                else:
                    data = model_output.hidden_states[-1][0]
                data = F.normalize(torch.mean(data, dim=0), p=2, dim=0)
                data = data.tolist()

                embedding.append(
                    EmbeddingData(index=index, object="embedding", embedding=data)
                )
                token_num += len(input_ids[0])

            usage = EmbeddingUsage(prompt_tokens=token_num, total_tokens=token_num)
            ret = Embedding(
                object="list", model=self.model_uid, data=embedding, usage=usage
            )

        return ret


class PytorchChatModel(PytorchModel, ChatModelMixin):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
    ):
        super().__init__(
            model_uid,
            model_family,
            model_spec,
            quantization,
            model_path,
            pytorch_model_config,
        )

    def _sanitize_generate_config(
        self,
        pytorch_generate_config: Optional[PytorchGenerateConfig],
    ) -> PytorchGenerateConfig:
        pytorch_generate_config = super()._sanitize_generate_config(
            pytorch_generate_config
        )
        if (
            "stop" not in pytorch_generate_config
            and self.model_family.prompt_style
            and self.model_family.prompt_style.stop
        ):
            pytorch_generate_config["stop"] = self.model_family.prompt_style.stop.copy()
        if (
            "stop_token_ids" not in pytorch_generate_config
            and self.model_family.prompt_style
            and self.model_family.prompt_style.stop_token_ids
        ):
            pytorch_generate_config[
                "stop_token_ids"
            ] = self.model_family.prompt_style.stop_token_ids.copy()

        return pytorch_generate_config

    @classmethod
    def match(cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1") -> bool:
        if llm_spec.model_format != "pytorch":
            return False
        if llm_family.model_name in [
            "baichuan-chat",
            "vicuna-v1.3",
            "falcon",
            "falcon-instruct",
            "chatglm",
            "chatglm2",
            "chatglm2-32k",
            "llama-2",
            "llama-2-chat",
        ]:
            return False
        if "chat" not in llm_family.model_ability:
            return False
        return True

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        assert self.model_family.prompt_style is not None
        prompt_style = self.model_family.prompt_style.copy()
        if system_prompt:
            prompt_style.system_prompt = system_prompt
        chat_history = chat_history or []
        full_prompt = self.get_prompt(prompt, chat_history, prompt_style)

        generate_config = self._sanitize_generate_config(generate_config)

        stream = generate_config.get("stream", False)
        if stream:
            it = self.generate(full_prompt, generate_config)
            assert isinstance(it, Iterator)
            return self._convert_chat_completion_chunks_to_chat(it)
        else:
            c = self.generate(full_prompt, generate_config)
            assert not isinstance(c, Iterator)
            return self._convert_text_completion_to_chat(c)
