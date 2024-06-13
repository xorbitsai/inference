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

import json
import logging
import os
from functools import lru_cache
from typing import Iterable, Iterator, List, Optional, Union

from ....core.scheduler import InferenceRequest
from ....device_utils import (
    get_device_preferred_dtype,
    gpu_count,
    is_hf_accelerate_supported,
)
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CreateCompletionTorch,
    Embedding,
    EmbeddingData,
    EmbeddingUsage,
    LoRA,
    PytorchGenerateConfig,
    PytorchModelConfig,
)
from ...utils import select_device
from ..core import LLM
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import ChatModelMixin
from .utils import get_context_length, get_max_src_len

logger = logging.getLogger(__name__)

NON_DEFAULT_MODEL_LIST: List[str] = [
    "baichuan-chat",
    "baichuan-2-chat",
    "vicuna-v1.3",
    "falcon",
    "falcon-instruct",
    "chatglm",
    "chatglm2",
    "chatglm2-32k",
    "chatglm2-128k",
    "chatglm3",
    "chatglm3-32k",
    "chatglm3-128k",
    "glm4-chat",
    "glm4-chat-1m",
    "llama-2",
    "llama-2-chat",
    "internlm2-chat",
    "qwen-vl-chat",
    "OmniLMM",
    "yi-vl-chat",
    "deepseek-vl-chat",
    "internvl-chat",
    "mini-internvl-chat",
    "cogvlm2",
    "MiniCPM-Llama3-V-2_5",
    "glm-4v",
]


class PytorchModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        pytorch_model_config: Optional[PytorchModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        self._use_fast_tokenizer = True
        self._pytorch_model_config: PytorchModelConfig = self._sanitize_model_config(
            pytorch_model_config
        )
        self._peft_model = peft_model

    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        if pytorch_model_config is None:
            pytorch_model_config = PytorchModelConfig()
        pytorch_model_config.setdefault("revision", self.model_spec.model_revision)
        pytorch_model_config.setdefault("gptq_ckpt", None)
        pytorch_model_config.setdefault("gptq_wbits", 16)
        pytorch_model_config.setdefault("gptq_groupsize", -1)
        pytorch_model_config.setdefault("gptq_act_order", False)
        pytorch_model_config.setdefault("device", "auto")
        pytorch_model_config.setdefault("trust_remote_code", True)
        pytorch_model_config.setdefault("max_num_seqs", 16)
        return pytorch_model_config

    def _sanitize_generate_config(
        self,
        generate_config: Optional[PytorchGenerateConfig],
    ) -> PytorchGenerateConfig:
        if generate_config is None:
            generate_config = PytorchGenerateConfig(**CreateCompletionTorch().dict())
        else:
            # Validate generate_config and fill default values to the generate config.
            generate_config = PytorchGenerateConfig(
                **CreateCompletionTorch(**generate_config).dict()
            )
        generate_config["model"] = self.model_uid
        return generate_config

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
            use_fast=self._use_fast_tokenizer,
            trust_remote_code=kwargs["trust_remote_code"],
            revision=kwargs["revision"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        return model, tokenizer

    def _apply_lora(self):
        if self._peft_model is not None:
            try:
                from peft import PeftModel
            except ImportError:
                raise ImportError(
                    f"Failed to import 'PeftModel' from 'peft'. Please make sure 'peft' is installed.\n\n"
                )

            for i, peft_model in enumerate(self._peft_model):
                if i == 0:
                    self._model = PeftModel.from_pretrained(
                        self._model,
                        peft_model.local_path,
                        adapter_name=peft_model.lora_name,
                    )
                else:
                    self._model.load_adapter(
                        peft_model.local_path, adapter_name=peft_model.lora_name
                    )
                logger.info(
                    f"PEFT adaptor '{peft_model.lora_name}' successfully loaded for model '{self.model_uid}'."
                )

    def load(self):
        try:
            import torch
        except ImportError:
            raise ImportError(
                f"Failed to import module 'torch'. Please make sure 'torch' is installed.\n\n"
            )
        from .compression import load_compress_model

        quantization = self.quantization
        num_gpus = gpu_count()
        device = self._pytorch_model_config.get("device", "auto")
        self._pytorch_model_config["device"] = select_device(device)
        self._device = self._pytorch_model_config["device"]

        kwargs = {}

        dtype = get_device_preferred_dtype(self._device)

        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        else:
            raise ValueError(f"Device {self._device} is not supported in temporary")

        kwargs["revision"] = self._pytorch_model_config.get(
            "revision", self.model_spec.model_revision
        )
        kwargs["trust_remote_code"] = self._pytorch_model_config.get(
            "trust_remote_code"
        )
        model_format = self.model_spec.model_format

        is_device_map_auto = False

        # This is required for Intel GPU to actually work with accelerate device_map until
        # https://github.com/intel/intel-extension-for-pytorch/issues/522
        # is resolved
        max_memory_env = os.getenv("ACCELERATE_MAX_MEMORY", None)

        if max_memory_env is not None:
            max_memory_raw = json.loads(max_memory_env)
            max_memory = {
                int(k) if k.isdigit() else k: max_memory_raw[k] for k in max_memory_raw
            }
            kwargs["max_memory"] = max_memory

        if quantization != "none" and model_format == "pytorch":
            if self._device == "cuda" and self._is_linux():
                kwargs["device_map"] = "auto"
                is_device_map_auto = True
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
                if num_gpus != 1 and self._device == "cuda":
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

        if num_gpus > 0 and is_hf_accelerate_supported(self._device):
            kwargs.update({"device_map": "auto"})
            is_device_map_auto = True

        self._model, self._tokenizer = self._load_model(**kwargs)
        self._apply_lora()

        if not is_device_map_auto:
            self._model.to(self._device)
        logger.debug(f"Model Memory: {self._model.get_memory_footprint()}")

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if llm_spec.model_format not in ["pytorch", "gptq", "awq"]:
            return False
        model_family = llm_family.model_family or llm_family.model_name
        if model_family in NON_DEFAULT_MODEL_LIST:
            return False
        if "generate" not in llm_family.model_ability:
            return False
        return True

    def generate(
        self, prompt: str, generate_config: Optional[PytorchGenerateConfig] = None
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        from .utils import generate_stream, generate_stream_falcon

        model_family_name = self.model_family.model_name.lower()

        def generator_wrapper(
            prompt: str, generate_config: PytorchGenerateConfig
        ) -> Iterator[CompletionChunk]:
            if "falcon" in model_family_name:
                for completion_chunk, completion_usage in generate_stream_falcon(
                    self.model_uid,
                    self._model,
                    self._tokenizer,
                    prompt,
                    self._device,
                    generate_config,
                ):
                    completion_chunk["usage"] = completion_usage
                    yield completion_chunk
            else:
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

        lora_model = generate_config.pop("lora_name")

        if lora_model is not None and self._peft_model is not None:
            for lora in self._peft_model:
                if lora_model == lora.lora_name:
                    self._model.set_adapter(lora_model)
                    logger.info(f"Set lora model to {lora_model}")
                    break
            else:
                self._model.disable_adapter()
                logger.info(f"No lora model {lora_model} found, skip setting")

        stream = generate_config.get("stream", False)
        if not stream:
            if "falcon" in model_family_name:
                for completion_chunk, completion_usage in generate_stream_falcon(
                    self.model_uid,
                    self._model,
                    self._tokenizer,
                    prompt,
                    self._device,
                    generate_config,
                ):
                    pass
            else:
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

    @lru_cache
    def get_context_len(self):
        return get_context_length(self._model.config)

    def get_max_num_seqs(self) -> int:
        return self._pytorch_model_config.get("max_num_seqs")  # type: ignore

    def prepare_batch_inference(self, req_list: List[InferenceRequest]):
        # check some parameters
        for r in req_list:
            if r.sanitized_generate_config is None:
                r.sanitized_generate_config = self._sanitize_generate_config(
                    r.generate_config
                )
            if r.is_prefill:
                # check some generate params
                max_src_len = get_max_src_len(self.get_context_len(), r)  # type: ignore
                if max_src_len < 0:
                    r.stopped = True
                    r.error_msg = "Max tokens exceeds model's max length"
                    continue
                if r.stream_interval <= 0:
                    r.stopped = True
                    r.error_msg = "`stream_interval` must be greater than 0"
                    continue
                stop_str = r.sanitized_generate_config.get("stop", None)
                if stop_str and (
                    not (isinstance(stop_str, str) or isinstance(stop_str, Iterable))
                ):
                    r.stopped = True
                    r.error_msg = "Invalid `stop` field type"
                    continue

    def handle_batch_inference_results(self, req_list: List[InferenceRequest]):
        for req in req_list:
            if req.error_msg is None:
                # nothing need handle for non-stream case
                if req.stream:
                    results = []
                    for i, c in enumerate(req.completion):
                        if c == "<bos_stream>":
                            chunk = req.completion[i + 1]
                            results.append(
                                CompletionChunk(
                                    id=chunk["id"],
                                    object=chunk["object"],
                                    created=chunk["created"],
                                    model=chunk["model"],
                                    choices=[
                                        CompletionChoice(
                                            text="",
                                            index=0,
                                            logprobs=None,
                                            finish_reason=None,
                                        )
                                    ],
                                )
                            )
                            continue
                        elif c == "<eos_stream>":
                            break
                        else:
                            results.append(c)

                    if req.stopped and req.include_usage:
                        results.append(req.completion[-1])
                    req.completion = results

    def batch_inference(self, req_list: List[InferenceRequest]):
        from .utils import batch_inference_one_step

        self.prepare_batch_inference(req_list)
        context_len = self.get_context_len()
        assert isinstance(context_len, int)
        batch_inference_one_step(
            req_list,
            self.model_uid,
            self._model,
            self._tokenizer,
            self._device,
            context_len,
        )
        self.handle_batch_inference_results(req_list)

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
        tokenizer.pad_token = tokenizer.eos_token
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
        peft_model: Optional[List[LoRA]] = None,
    ):
        super().__init__(
            model_uid,
            model_family,
            model_spec,
            quantization,
            model_path,
            pytorch_model_config,
            peft_model,
        )

    def _sanitize_generate_config(
        self,
        generate_config: Optional[PytorchGenerateConfig],
    ) -> PytorchGenerateConfig:
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
        if llm_spec.model_format not in ["pytorch", "gptq", "awq"]:
            return False
        model_family = llm_family.model_family or llm_family.model_name
        if model_family in NON_DEFAULT_MODEL_LIST:
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

    def load(self):
        super().load()

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

    def prepare_batch_inference(self, req_list: List[InferenceRequest]):
        super().prepare_batch_inference(req_list)
        for r in req_list:
            r.full_prompt = self._get_full_prompt(
                r.prompt, r.system_prompt, r.chat_history, None
            )

    def handle_batch_inference_results(self, req_list: List[InferenceRequest]):
        for req in req_list:
            if req.stream and req.error_msg is None:
                if req.completion:
                    results = []
                    for i, c in enumerate(req.completion):
                        if c == "<bos_stream>":
                            results.append(
                                self._get_first_chat_completion_chunk(
                                    req.completion[i + 1]
                                )
                            )
                        elif c == "<eos_stream>":
                            break
                        else:
                            results.append(self._to_chat_completion_chunk(c))

                    if req.stopped and req.include_usage:
                        results.append(
                            self._get_final_chat_completion_chunk(req.completion[-1])
                        )
                    req.completion = results
