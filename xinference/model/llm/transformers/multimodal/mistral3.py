# Copyright 2022-2025 XProbe Inc.
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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from .....model.utils import select_device
from .....types import PytorchModelConfig
from ...utils import _decode_image
from ...llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from ..core import register_non_default_model
from .core import PytorchMultiModalModel

logger = logging.getLogger(__name__)

@register_transformer
@register_non_default_model("mistral-small-3.2-instruct")
class MistralMultimodalModel(PytorchMultiModalModel):
    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        pytorch_model_config = super()._sanitize_model_config(pytorch_model_config)
        assert pytorch_model_config is not None
        return pytorch_model_config

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV2", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if model_spec.model_format not in ["pytorch", "gptq", "awq"]:
            return False
        llm_family = model_family.model_family or model_family.model_name
        if "mistral-small-3.2-instruct" in llm_family.lower():
            return True
        return False

    def decide_device(self):
        device = self._pytorch_model_config.get("device", "cuda")
        self._device = select_device(device)

    def load_processor(self):
        from transformers import AutoProcessor
        from transformers import AutoTokenizer
        min_pixels = self._pytorch_model_config.get("min_pixels")
        max_pixels = self._pytorch_model_config.get("max_pixels")
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False
        )
        

    def load_multimodal_model(self):
        from transformers import BitsAndBytesConfig
        from transformers import Mistral3ForConditionalGeneration
        kwargs = {"device_map": self._device}
        kwargs = self.apply_bnb_quantization(kwargs)
        
        if '4bit' in self.model_path:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["quantization_config"] = quantization_config
        elif '8bit' in self.model_path:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["quantization_config"] = quantization_config

        self._model = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_path, 
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            **kwargs
        ).eval()
        # if self._device == 'cuda':
        #     self._model.cuda()

                
    @staticmethod
    def _get_processed_msgs(messages: List[Dict]) -> List[Dict]:
        res = []
        texts = []
        images = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if isinstance(content, str):
                res.append({"role": role, "content": [{"type": "text", "text": content}]})
                texts.append(content)
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
                    res.append({"role": role, "content":  [{"type": "image", "image": images[0]}, {"type": "text", "text": text}] })
                    texts.append(text)
                    images.append(images[0])
                else:
                    texts.append(text)
                    res.append({"role": role, "content": [{"type": "text", "text": text}]})
        return res,texts,images
    
    @staticmethod
    def flatten_content(msg):
        if isinstance(msg["content"], list):
            parts = []
            for part in msg["content"]:
                if part["type"] == "image":
                    parts.append("<image>")  # 或者其他占位符
                elif part["type"] == "text":
                    parts.append(part["text"])
            msg["content"] = "".join(parts)
        return msg
    
    def build_inputs_from_messages(
        self, messages: List[Dict], generate_config: Dict
    ):
        rst, text, images = self._get_processed_msgs(messages)
        flattened_messages = [self.flatten_content(m.copy()) for m in rst]
        inputs = self._tokenizer.apply_chat_template(
            conversation=flattened_messages,
            # text=text,
            images=images,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = inputs.to(self._device)
        return inputs

    def build_generate_kwargs(self, generate_config: Dict) -> Dict[str, Any]:
        return dict(
            max_new_tokens=generate_config.get("max_tokens", 1000),
            temperature=generate_config.get("temperature", 1),
            eos_token_id=generate_config.get("eos_token_id", 2),
            do_sample=generate_config.get("do_sample", True),
            bos_token_id=generate_config.get("bos_token_id", 1),
        )

    def build_streaming_iter(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ) -> Tuple[Iterator, int]:
        from threading import Thread
        from transformers import TextIteratorStreamer

        inputs = self.build_inputs_from_messages(messages, generate_config)
        configs = self.build_generate_kwargs(generate_config)

        tokenizer = self._tokenizer
        streamer = TextIteratorStreamer(
            tokenizer, 
            timeout=60.0,
            skip_prompt=True,
            skip_special_tokens=True
        )

        gen_kwargs = {"streamer": streamer, **inputs, **configs}
        t = Thread(target=self._model.generate, kwargs=gen_kwargs)
        t.start()
        return streamer, len(inputs["input_ids"][0])
