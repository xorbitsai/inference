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
import logging
from typing import Dict, List, Set, Tuple, Union

from ...scheduler.request import InferenceRequest
from ..llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from .core import PytorchChatModel, register_non_default_model

logger = logging.getLogger(__name__)


@register_transformer
@register_non_default_model("Gemma3ForCausalLM")
class Gemma3TextChatModel(PytorchChatModel):
    GEMMA3_ARCHITECTURES = {"Gemma3ForCausalLM"}

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV2", model_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if model_spec.model_format not in ["pytorch", "gptq", "awq", "bnb", "fp4"]:
            return (
                False,
                "Gemma3 transformer supports pytorch/gptq/awq/bnb/fp4 formats only",
            )
        if not model_family.has_architecture(*cls.GEMMA3_ARCHITECTURES):
            return (
                False,
                f"Model architectures {model_family.architectures} are not Gemma-3-1B-it",
            )
        return True

    def _load_model(self, **kwargs):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=kwargs["trust_remote_code"],
            revision=kwargs["revision"],
        )
        kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **kwargs,
        )
        self._device = model.device
        return model, tokenizer

    def _get_full_prompt(self, messages: List[Dict], tools, generate_config: dict):
        return messages

    def build_prefill_kwargs(self, prompts: List, req_list: List[InferenceRequest]):
        """
        Note that it is important to prepare `past_key_values` for gemma3 prefill phase
        """
        from transformers import HybridCache

        inputs = self._tokenizer.apply_chat_template(
            prompts,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        ).to(self._device)

        for i, r in enumerate(req_list):
            r.prompt_tokens = inputs["input_ids"][i].tolist()

        batch_size = len(prompts)
        max_cache_len = self.get_context_len()
        kv = HybridCache(
            self._model.config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            dtype=self._model.dtype,
            device=self._device,
        )
        return {**inputs, "past_key_values": kv}

    def merge_kv_cache(self, past_cache, new_cache):
        """
        Note that: DO NOT use the `update` func of `HybridCache`, that is unrelated to KV cache merging.
        """
        import torch
        from transformers import HybridCache

        max_cache_len = new_cache.max_cache_len
        batch_size = past_cache.max_batch_size + new_cache.max_batch_size

        kv_batch = HybridCache(
            self._model.config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            dtype=self._model.dtype,
            device=self._device,
        )

        new_ks = [
            torch.cat([nk, pk], dim=0).contiguous()
            for nk, pk in zip(new_cache.key_cache, past_cache.key_cache)
        ]
        new_vs = [
            torch.cat([nv, pv], dim=0).contiguous()
            for nv, pv in zip(new_cache.value_cache, past_cache.value_cache)
        ]

        kv_batch.key_cache.clear()
        kv_batch.value_cache.clear()
        kv_batch.key_cache.extend(new_ks)
        kv_batch.value_cache.extend(new_vs)

        return kv_batch

    def build_decode_attention_mask(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        In Gemma3's inference script, attention_mask is handled internally for decode phase.
        """
        return None

    def build_decode_position_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        """
        In Gemma3's inference script, position_ids is handled internally for decode phase.
        """
        return None

    def build_reduced_kv_cache(self, cache, skipped_indexes: Set[int]):
        from transformers import HybridCache

        batch_slices = [
            num for num in range(cache.max_batch_size) if num not in skipped_indexes
        ]
        batch_size = len(batch_slices)

        kv_batch = HybridCache(
            self._model.config,
            max_batch_size=batch_size,
            max_cache_len=cache.max_cache_len,
            dtype=self._model.dtype,
            device=self._device,
        )

        ks = cache.key_cache
        vs = cache.value_cache

        new_ks = [_k[batch_slices, ::].contiguous() for _k in ks]
        new_vs = [_v[batch_slices, ::].contiguous() for _v in vs]
        kv_batch.key_cache.clear()
        kv_batch.value_cache.clear()
        kv_batch.key_cache.extend(new_ks)
        kv_batch.value_cache.extend(new_vs)

        return kv_batch
