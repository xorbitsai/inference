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
import typing
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Dict, Iterator, List, Optional, Union

import torch

from ....core.scheduler import InferenceRequest
from ....types import ChatCompletion, ChatCompletionChunk, CompletionChunk
from ...utils import select_device
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import _decode_image, generate_chat_completion, generate_completion_chunk
from .core import PytorchChatModel, PytorchGenerateConfig
from .utils import get_max_src_len

logger = logging.getLogger(__name__)


class Glm4VModel(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device = None
        self._tokenizer = None
        self._model = None

    @classmethod
    def match(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        family = model_family.model_family or model_family.model_name
        if "glm-4v" in family.lower():
            return True
        return False

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._pytorch_model_config.get("device", "auto")
        self._device = select_device(device)

        kwargs = {"device_map": self._device}
        quantization = self.quantization

        # referenced from PytorchModel.load
        if quantization != "none":
            if self._device == "cuda" and self._is_linux():
                kwargs["device_map"] = "auto"
                if quantization == "4-bit":
                    kwargs["load_in_4bit"] = True
                elif quantization == "8-bit":
                    kwargs["load_in_8bit"] = True
                else:
                    raise ValueError(
                        f"Quantization {quantization} is not supported in temporary"
                    )
            else:
                if quantization != "8-bit":
                    raise ValueError(
                        f"Only 8-bit quantization is supported if it is not linux system or cuda device"
                    )

        if self._check_tensorizer_integrity():
            self._model, self._tokenizer = self._load_tensorizer()
            return

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            **kwargs,
        )
        self._model = model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._tokenizer = tokenizer
        self._save_tensorizer()

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

    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        from transformers import TextIteratorStreamer

        if not generate_config:
            generate_config = {}

        stream = generate_config.get("stream", False)
        msgs = self._get_processed_msgs(messages)

        inputs = self._tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )  # chat mode
        inputs = inputs.to(self._model.device)

        generate_kwargs = {
            **inputs,
            "eos_token_id": [151329, 151336, 151338],
            "do_sample": True,
            "max_length": generate_config.get("max_tokens", 2048),
            "temperature": generate_config.get("temperature", 0.7),
        }
        stop_str = "<|endoftext|>"

        if stream:
            streamer = TextIteratorStreamer(
                tokenizer=self._tokenizer,
                timeout=60,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generate_kwargs = {
                **generate_kwargs,
                "streamer": streamer,
            }
            t = Thread(target=self._model.generate, kwargs=generate_kwargs)
            t.start()

            it = self.chat_stream(streamer, stop_str)
            return self._to_chat_completion_chunks(it)
        else:
            with torch.no_grad():
                outputs = self._model.generate(**generate_kwargs)
                outputs = outputs[:, inputs["input_ids"].shape[1] :]
                response = self._tokenizer.decode(outputs[0])
                if response.endswith(stop_str):
                    response = response[: -len(stop_str)]
            return generate_chat_completion(self.model_uid, response)

    def chat_stream(self, streamer, stop_str) -> Iterator[CompletionChunk]:
        completion_id = str(uuid.uuid1())
        for new_text in streamer:
            if not new_text.endswith(stop_str):
                yield generate_completion_chunk(
                    chunk_text=new_text,
                    finish_reason=None,
                    chunk_id=completion_id,
                    model_uid=self.model_uid,
                    prompt_tokens=-1,
                    completion_tokens=-1,
                    total_tokens=-1,
                    has_choice=True,
                    has_content=True,
                )

        yield generate_completion_chunk(
            chunk_text=None,
            finish_reason="stop",
            chunk_id=completion_id,
            model_uid=self.model_uid,
            prompt_tokens=-1,
            completion_tokens=-1,
            total_tokens=-1,
            has_choice=True,
            has_content=False,
        )

    def _get_full_prompt(self, messages, tools):
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
