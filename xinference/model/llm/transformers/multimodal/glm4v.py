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
import typing
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch

from .....core.model import register_batching_multimodal_models
from .....model.utils import select_device
from ....scheduler.request import InferenceRequest
from ...llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from ...utils import _decode_image
from ..core import register_non_default_model
from ..utils import get_max_src_len
from .core import PytorchMultiModalModel

logger = logging.getLogger(__name__)


@register_batching_multimodal_models("glm-4v")
@register_transformer
@register_non_default_model("ChatGLMModel")
class Glm4VModel(PytorchMultiModalModel):
    GLM4V_ARCHITECTURES = {
        "Glm4vForConditionalGeneration",
        "Glm4vMoeForConditionalGeneration",
    }

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV2", model_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if not model_family.has_architecture(*cls.GLM4V_ARCHITECTURES):
            return (
                False,
                f"Model architectures {model_family.architectures} are not GLM-4V",
            )
        if "vision" not in model_family.model_ability:
            return False, "GLM-4V transformer requires vision ability"
        return True

    def decide_device(self):
        device = self._pytorch_model_config.get("device", "auto")
        self._device = select_device(device)

    def load_processor(self):
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def load_multimodal_model(self):
        from transformers import AutoModelForCausalLM

        kwargs = {"device_map": self._device}
        kwargs = self.apply_quantization_config(kwargs)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            **kwargs,
        )
        self._model = model.eval()

    @staticmethod
    def _get_processed_msgs(messages: List[Dict]) -> List[Dict]:
        res = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if isinstance(content, str):
                res.append({"role": role, "content": content})
            else:
                texts = []
                image_urls = []
                for c in content:
                    c_type = c.get("type")
                    if c_type == "text":
                        texts.append(c["text"])
                    else:
                        assert (
                            c_type == "image_url"
                        ), "Please follow the image input of the OpenAI API."
                        image_urls.append(c["image_url"]["url"])
                if len(image_urls) > 1:
                    raise RuntimeError("Only one image per message is supported")
                image_futures = []
                with ThreadPoolExecutor() as executor:
                    for image_url in image_urls:
                        fut = executor.submit(_decode_image, image_url)
                        image_futures.append(fut)
                images = [fut.result() for fut in image_futures]
                assert len(images) <= 1
                text = " ".join(texts)
                if images:
                    res.append({"role": role, "content": text, "image": images[0]})
                else:
                    res.append({"role": role, "content": text})
        return res

    def build_inputs_from_messages(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ):
        msgs = self._get_processed_msgs(messages)
        inputs = self._tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )  # chat mode
        inputs = inputs.to(self._model.device)
        return inputs

    def build_generate_kwargs(
        self,
        generate_config: Dict,
    ) -> Dict[str, Any]:
        return {
            "eos_token_id": [151329, 151336, 151338],
            "do_sample": True,
            "max_length": generate_config.get("max_tokens") or 2048,
            "temperature": generate_config.get("temperature", 0.7),
        }

    def get_stop_strs(self) -> List[str]:
        return ["<|endoftext|>"]

    def build_streaming_iter(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ) -> Tuple[Iterator, int]:
        from transformers import TextIteratorStreamer

        generate_kwargs = self.build_generate_kwargs(generate_config)
        inputs = self.build_inputs_from_messages(messages, generate_config)
        streamer = TextIteratorStreamer(
            tokenizer=self._tokenizer,
            timeout=60,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        kwargs = {
            **inputs,
            **generate_kwargs,
            "streamer": streamer,
        }
        t = Thread(target=self._model.generate, kwargs=kwargs)
        t.start()
        return streamer, len(inputs.input_ids[0])

    def _get_full_prompt(self, messages, tools, generate_config: dict):
        msgs = self._get_processed_msgs(messages)
        inputs = self._tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "images": inputs.images.squeeze(0),
        }

    def prepare_sanitize_generate_config(self, req: InferenceRequest):
        """
        Refer to https://huggingface.co/THUDM/glm-4v-9b/blob/main/generation_config.json
        """
        raw_config = req.inference_kwargs.get("raw_params", {})
        temperature = raw_config.get("temperature", None)
        if temperature is None:
            raw_config["temperature"] = 0.8
        top_p = raw_config.get("top_p", None)
        if top_p is None:
            raw_config["top_p"] = 0.8
        return raw_config

    def build_prefill_inputs(self, prompts: List, req_list: List[InferenceRequest]):
        context_len = self.get_context_len()
        assert isinstance(prompts[0], dict)
        images = []
        max_length = float("-inf")
        for i, feature in enumerate(prompts):
            req = req_list[i]
            if "images" in feature:
                images.append(feature.pop("images", None))
            max_src_len = get_max_src_len(context_len, req)
            input_ids = feature["input_ids"][-max_src_len:]
            req.prompt_tokens = input_ids.tolist()
            feature["input_ids"] = input_ids
            max_length = max(len(input_ids), max_length)

        def pad_to_max_length_internal(feature, max_len, idx):
            padding_length = max_len - len(feature["input_ids"])
            req_list[idx].padding_len = padding_length
            feature["input_ids"] = torch.cat(
                [torch.full((padding_length,), 0), feature["input_ids"]]
            )
            return feature

        features = [
            pad_to_max_length_internal(feature, max_length, i)
            for i, feature in enumerate(prompts)
        ]
        batch = {
            key: torch.stack([feature[key] for feature in features])
            for key in features[0].keys()
        }
        if images:
            batch["images"] = torch.stack(images).to(self._device)
        batch["input_ids"] = batch["input_ids"].to(self._device)
        return batch

    @staticmethod
    def is_empty(images_list: Optional[List[List[torch.Tensor]]]):
        """
        Copied from https://huggingface.co/THUDM/glm-4v-9b/blob/main/modeling_chatglm.py
        """
        if images_list is None or len(images_list) == 0:
            return True
        for image_list in images_list:
            if image_list is not None:
                return False
        return True

    @typing.no_type_check
    def get_full_attention_mask(
        self, attention_mask, input_ids, images, req_list: List[InferenceRequest]
    ):
        """
        Modified according to https://huggingface.co/THUDM/glm-4v-9b/blob/main/modeling_chatglm.py
        """
        image_size: int = self._model.config.vision_config["image_size"]
        patch_size: int = self._model.config.vision_config["patch_size"]
        num_patches = (image_size // patch_size // 2) ** 2
        new_attention_masks = []

        # if not image, use this default id
        eoi_token_pos = 6
        boi_token_pos = 4

        for i in range(len(input_ids)):
            input_id = input_ids[i].tolist()
            req = req_list[i]
            if not self.is_empty(images):
                _boi_token_pos, _eoi_token_pos = input_id.index(
                    self._model.config.boi_token_id
                ), input_id.index(self._model.config.eoi_token_id)
            else:
                _boi_token_pos = boi_token_pos + req.padding_len
                _eoi_token_pos = eoi_token_pos + req.padding_len
            assert eoi_token_pos - boi_token_pos == 2
            new_attention_masks.append(
                torch.cat(
                    (
                        attention_mask[i, : _boi_token_pos + 1],
                        attention_mask.new_ones(num_patches),
                        attention_mask[i, _eoi_token_pos:],
                    )
                )
            )
        attention_mask = torch.stack(new_attention_masks, dim=0).to(self._device)
        return attention_mask

    def build_prefill_kwargs(self, prompts: List, req_list: List[InferenceRequest]):
        batch = self.build_prefill_inputs(prompts, req_list)
        batch_size, seq_len = batch["input_ids"].shape
        attention_mask = self.build_prefill_attention_mask(
            batch_size, seq_len, req_list
        )
        if attention_mask is not None:
            full_attention_mask = self.get_full_attention_mask(
                attention_mask, batch["input_ids"], batch["images"], req_list
            )
            batch["attention_mask"] = full_attention_mask
            for r in req_list:
                r.extra_kwargs["attention_mask_seq_len"] = full_attention_mask.shape[1]
        position_ids = self.build_prefill_position_ids(batch_size, seq_len, req_list)
        if position_ids is not None:
            batch["position_ids"] = position_ids
        return batch

    def build_decode_attention_mask(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        max_seq_len = max(r.extra_kwargs["attention_mask_seq_len"] for r in reqs)

        new_attention_mask = []
        for r in reqs:
            attn_mask_seq_len = r.extra_kwargs["attention_mask_seq_len"]
            pad_len = max_seq_len - attn_mask_seq_len
            new_attention_mask.append(
                torch.cat(
                    [torch.full((pad_len,), 0), torch.ones((attn_mask_seq_len + 1,))]
                )
            )
            r.extra_kwargs["attention_mask_seq_len"] += 1
        return torch.stack(new_attention_mask, dim=0).to(self._device)
