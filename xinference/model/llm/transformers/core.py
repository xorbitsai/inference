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
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import torch

from ....core.scheduler import InferenceRequest
from ....device_utils import (
    get_device_preferred_dtype,
    gpu_count,
    is_hf_accelerate_supported,
)
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CreateCompletionTorch,
    LoRA,
    PytorchGenerateConfig,
    PytorchModelConfig,
)
from ...utils import select_device
from ..core import LLM
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import QWEN_TOOL_CALL_FAMILY, ChatModelMixin
from .utils import get_context_length, get_max_src_len, pad_prefill_tokens

logger = logging.getLogger(__name__)

NON_DEFAULT_MODEL_LIST: List[str] = [
    "chatglm3",
    "chatglm3-32k",
    "chatglm3-128k",
    "glm4-chat",
    "glm4-chat-1m",
    "internlm2-chat",
    "internlm2.5-chat",
    "qwen-vl-chat",
    "OmniLMM",
    "yi-vl-chat",
    "deepseek-vl-chat",
    "internvl-chat",
    "internvl2",
    "cogvlm2",
    "cogvlm2-video-llama3-chat",
    "MiniCPM-Llama3-V-2_5",
    "MiniCPM-V-2.6",
    "glm-4v",
    "qwen2-vl-instruct",
    "qwen2-audio",
    "qwen2-audio-instruct",
    "deepseek-v2",
    "deepseek-v2-chat",
    "deepseek-v2.5",
    "deepseek-v2-chat-0628",
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
        pytorch_model_config.setdefault("enable_tensorizer", False)
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

    def _check_tensorizer_integrity(self):
        if not self._pytorch_model_config.get("enable_tensorizer"):
            return False

        from .tensorizer_utils import check_tensorizer_integrity

        integrity = check_tensorizer_integrity(
            self.model_path,
            [component[0] for component in self._get_components()],
        )
        logger.info(f"Tensorizer files integrity: {integrity} {self.model_uid}")
        return integrity

    def _load_tensorizer(self, **kwargs):
        enable_tensorizer = self._pytorch_model_config.get("enable_tensorizer", None)
        if enable_tensorizer:
            from .tensorizer_utils import load_from_tensorizer

            component_metadata = [
                (name, type, kwargs)
                for name, _, type, kwargs in self._get_components(**kwargs)
            ]
            model, tokenizer = load_from_tensorizer(
                self.model_path, component_metadata, self._get_model_class(), **kwargs
            )
            return model, tokenizer

    def _save_tensorizer(self, **kwargs):
        enable_tensorizer = self._pytorch_model_config.get("enable_tensorizer", None)
        if enable_tensorizer:
            from .tensorizer_utils import save_to_tensorizer

            components = [(name, obj) for name, obj, _, _ in self._get_components()]
            save_to_tensorizer(self.model_path, self._model, components, **kwargs)

    def _get_model_class(self):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM

    def _get_components(self, **kwargs):
        from transformers import AutoTokenizer

        return [
            (
                "tokenizer",
                getattr(self, "_tokenizer", None),
                AutoTokenizer,
                {
                    "use_fast": self._use_fast_tokenizer,
                    "trust_remote_code": kwargs.get("trust_remote_code", True),
                    "revision": kwargs.get("revision"),
                    "code_revision": kwargs.get("code_revision", None),
                },
            )
        ]

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
                    (
                        self._model,
                        self._tokenizer,
                    ) = load_compress_model(
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

        if self._check_tensorizer_integrity():
            self._model, self._tokenizer = self._load_tensorizer(**kwargs)
        else:
            self._model, self._tokenizer = self._load_model(**kwargs)

        self._apply_lora()

        if not is_device_map_auto:
            self._model.to(self._device)

        self._save_tensorizer(**kwargs)

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
        from .utils import generate_stream

        def generator_wrapper(
            prompt: str, generate_config: PytorchGenerateConfig
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

    def build_prefill_attention_mask(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        Build attention mask for prefill phase.
        Padding `0` on the left.
        Note that the parameter `seq_length` is from `input_ids`.
        """
        data = []
        for r in reqs:
            real_len = seq_length - r.padding_len
            x = torch.cat(
                [
                    torch.full((r.padding_len,), 0, dtype=torch.long),
                    torch.ones((real_len,), dtype=torch.long),
                ]
            )
            data.append(x)
            r.extra_kwargs["attention_mask_seq_len"] = real_len
        return torch.stack(data).to(self._device)

    def build_decode_attention_mask(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        Build attention mask for decode phase.
        Note that the `seq_length` parameter is from merged kv_cache.
        So we need pad `0` on the left again.
        """
        data = []
        for r in reqs:
            r.extra_kwargs["attention_mask_seq_len"] += 1
            attention_mask_seq_len = r.extra_kwargs["attention_mask_seq_len"]
            pad_len = seq_length - attention_mask_seq_len
            x = torch.cat(
                [
                    torch.full((pad_len,), 0, dtype=torch.long),
                    torch.ones((attention_mask_seq_len,), dtype=torch.long),
                ]
            )
            data.append(x)
        return torch.stack(data).to(self._device)

    def build_prefill_position_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        Build position ids for prefill phase.
        Padding `0` on the left.
        Note that the parameter `seq_length` is from `input_ids`.
        Record the `max_position_id` on request for the decode phase.
        """
        res = []
        for r in reqs:
            real_seq_len = seq_length - r.padding_len
            res.append(
                torch.cat(
                    [
                        torch.full((r.padding_len,), 0, dtype=torch.long),
                        torch.arange(0, real_seq_len, dtype=torch.long),
                    ]
                )
            )
            r.extra_kwargs["max_position_id"] = real_seq_len - 1
        return torch.stack(res).to(self._device)

    def build_decode_position_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        Build position ids for decode phase.
        For most models, just let the `max_position_id` in previous step += 1 and use the latest `max_position_id`
        """
        data = []
        for r in reqs:
            r.extra_kwargs["max_position_id"] += 1
            data.append([r.extra_kwargs["max_position_id"]])
        position_ids = torch.as_tensor(data, dtype=torch.long, device=self._device)
        return position_ids

    def build_prefill_token_type_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        Build token_type_ids for prefill phase.
        For most models, this is not required.
        """
        return None

    def build_decode_token_type_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        Build token_type_ids for decode phase.
        For most models, this is not required.
        """
        return None

    def build_prefill_inputs(self, prompts: List, req_list: List[InferenceRequest]):
        """
        Get inputs for inference. Models may have their own impl.
        """
        assert isinstance(prompts[0], str)
        inputs = self._tokenizer(prompts, padding=False).input_ids
        context_len = self.get_context_len()
        input_ids = torch.as_tensor(
            pad_prefill_tokens(inputs, context_len, req_list), device=self._device
        )
        return input_ids

    def build_prefill_kwargs(self, prompts: List, req_list: List[InferenceRequest]):
        """
        Get all inputs parameters for prefill phase. Models may have their own impl.
        """
        input_ids = self.build_prefill_inputs(prompts, req_list)
        res = {"input_ids": input_ids}
        batch_size, seq_len = input_ids.shape
        attention_mask = self.build_prefill_attention_mask(
            batch_size, seq_len, req_list
        )
        if attention_mask is not None:
            res["attention_mask"] = attention_mask
        position_ids = self.build_prefill_position_ids(batch_size, seq_len, req_list)
        if position_ids is not None:
            res["position_ids"] = position_ids
        token_type_ids = self.build_prefill_token_type_ids(
            batch_size, seq_len, req_list
        )
        if token_type_ids is not None:
            res["token_type_ids"] = token_type_ids
        return res

    def build_decode_kwargs(
        self,
        prompts: List,
        req_list: List[InferenceRequest],
        batch_size: int,
        seq_len: int,
    ):
        """
        Get all inputs parameters for decode phase. Models may have their own impl.
        """
        res = {"input_ids": torch.as_tensor(prompts, device=self._device)}
        attention_mask = self.build_decode_attention_mask(batch_size, seq_len, req_list)
        if attention_mask is not None:
            res["attention_mask"] = attention_mask
        position_ids = self.build_decode_position_ids(batch_size, seq_len, req_list)
        if position_ids is not None:
            res["position_ids"] = position_ids
        token_type_ids = self.build_decode_token_type_ids(batch_size, seq_len, req_list)
        if token_type_ids is not None:
            res["token_type_ids"] = token_type_ids
        return res

    @staticmethod
    def get_batch_size_and_seq_len_indexes_from_kv() -> Tuple[int, int]:
        """
        From huggingface transformers document, the `pask_key_values` has the shape of
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`.
        However, for some models, the shape may be changed.
        """
        return 0, 2

    def get_dtype(self):
        raise NotImplementedError("Not implemented.")

    @lru_cache
    def get_context_len(self):
        return get_context_length(self._model.config)

    def get_max_num_seqs(self) -> int:
        return self._pytorch_model_config.get("max_num_seqs")  # type: ignore

    def prepare_sanitize_generate_config(self, req: InferenceRequest):
        return self._sanitize_generate_config(req.generate_config)

    def prepare_batch_inference(self, req_list: List[InferenceRequest]):
        # check some parameters
        for r in req_list:
            try:
                if r.sanitized_generate_config is None:
                    r.sanitized_generate_config = self.prepare_sanitize_generate_config(
                        r
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
                        not (
                            isinstance(stop_str, str) or isinstance(stop_str, Iterable)
                        )
                    ):
                        r.stopped = True
                        r.error_msg = "Invalid `stop` field type"
                        continue
            # Catch exception here. If not catch exception, the request would hang.
            except Exception as e:
                logger.exception(f"prepare inference error with {e}")
                r.stopped = True
                r.error_msg = str(e)

    def get_builtin_stop_token_ids(self) -> Tuple:
        from ..utils import get_stop_token_ids_from_config_file

        stop_token_ids = get_stop_token_ids_from_config_file(self.model_path)
        if stop_token_ids is not None:
            return tuple(stop_token_ids)
        else:
            return (
                tuple(self.model_family.stop_token_ids)
                if self.model_family.stop_token_ids
                else tuple()
            )

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
        batch_inference_one_step(
            self, req_list, self.model_uid, self._model, self._tokenizer
        )
        self.handle_batch_inference_results(req_list)


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
        if (not generate_config.get("stop")) and self.model_family.stop is not None:
            generate_config["stop"] = self.model_family.stop.copy()
        if (
            generate_config.get("stop_token_ids", None) is None
            and self.model_family.stop_token_ids is not None
        ):
            generate_config["stop_token_ids"] = self.model_family.stop_token_ids.copy()

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
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        tools = generate_config.pop("tools", []) if generate_config else None
        model_family = self.model_family.model_family or self.model_family.model_name
        full_context_kwargs = {}
        if tools and model_family in QWEN_TOOL_CALL_FAMILY:
            full_context_kwargs["tools"] = tools
        assert self.model_family.chat_template is not None
        full_prompt = self.get_full_context(
            messages,
            self.model_family.chat_template,
            tokenizer=self._tokenizer,
            **full_context_kwargs,
        )

        generate_config = self._sanitize_generate_config(generate_config)

        stream = generate_config.get("stream", False)
        if stream:
            it = self.generate(full_prompt, generate_config)
            assert isinstance(it, Iterator)
            return self._to_chat_completion_chunks(it)
        else:
            c = self.generate(full_prompt, generate_config)
            assert not isinstance(c, Iterator)
            if tools:
                return self._tool_calls_completion(self.model_family, self.model_uid, c)
            return self._to_chat_completion(c)

    def load(self):
        super().load()

    def _get_full_prompt(self, messages: List[Dict], tools):
        assert self.model_family.chat_template is not None
        full_prompt = self.get_full_context(
            messages, self.model_family.chat_template, tokenizer=self._tokenizer
        )
        return full_prompt

    def prepare_batch_inference(self, req_list: List[InferenceRequest]):
        super().prepare_batch_inference(req_list)
        for r in req_list:
            try:
                if not r.stopped and r.is_prefill:
                    r.full_prompt = self._get_full_prompt(r.prompt, None)
            except Exception as e:
                logger.exception(f"prepare inference error with {e}")
                r.stopped = True
                r.error_msg = str(e)

    def handle_batch_inference_results(self, req_list: List[InferenceRequest]):
        for req in req_list:
            if req.error_msg is None and req.completion:
                # The `generate` function can be called for some chat models.
                # So that we cannot convert completion chunk to chat completion chunk.
                if req.call_ability == "generate":
                    results = []
                    for c in req.completion:
                        if c == "<bos_stream>":
                            continue
                        elif c == "<eos_stream>":
                            break
                        else:
                            results.append(c)
                    req.completion = results
                    continue

                if req.stream:
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
                else:
                    req.completion[0] = self._to_chat_completion(req.completion[0])
