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
import base64
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Dict, Iterator, List, Tuple, Union

import requests
import torch

from .....model.utils import select_device
from ...llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from ..core import register_non_default_model
from .core import PytorchMultiModalModel

logger = logging.getLogger(__name__)


@register_transformer
@register_non_default_model("DeepseekV2ForCausalLM")
class DeepSeekVL2ChatModel(PytorchMultiModalModel):
    DEEPSEEK_VL2_ARCHITECTURES = {"DeepseekV2ForCausalLM"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._type = None

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV2", model_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if not model_family.has_architecture(*cls.DEEPSEEK_VL2_ARCHITECTURES):
            return (
                False,
                f"Model architectures {model_family.architectures} are not DeepSeek-VL2",
            )
        if "vision" not in model_family.model_ability:
            return False, "DeepSeek-VL2 transformer requires vision ability"
        return True

    def decide_device(self):
        self._device = self._pytorch_model_config.get("device", "auto")
        self._device = select_device(self._device)
        self._type = torch.bfloat16

    def load_processor(self):
        from .....thirdparty.deepseek_vl2.models import DeepseekVLV2Processor

        # specify the path to the model
        self._processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(  # type: ignore
            self.model_path
        )
        self._tokenizer = self._processor.tokenizer

    def load_multimodal_model(self):
        from transformers import AutoModelForCausalLM

        from .....thirdparty.deepseek_vl2.models import DeepseekVLV2ForCausalLM

        kwargs = self.apply_quantization_config()
        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(  # type: ignore
            self.model_path,
            trust_remote_code=True,
            device_map=self._device,
            torch_dtype=self._type,
            **kwargs,
        )
        self._model = vl_gpt.cuda().eval()

    @staticmethod
    def _message_content_to_deepseek(content) -> Tuple[str, List[str]]:
        def _ensure_url(_url):
            if _url.startswith("data:"):
                logging.info("Parse url by base64 decoder.")
                # https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
                # e.g. f"data:image/jpeg;base64,{base64_image}"
                _type, data = _url.split(";")
                _, ext = _type.split("/")
                data = data[len("base64,") :]
                data = base64.b64decode(data.encode("utf-8"))

                with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
                    f.write(data)
                logging.info("Dump base64 data to %s", f.name)
                return f.name
            else:
                if len(_url) > 2048:
                    raise Exception(f"Image url is too long, {len(_url)} > 2048.")

                return _url

        def _download(_images):
            local_images = []

            # To make requests.get works
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            }
            with ThreadPoolExecutor() as executor:
                for url in images:
                    try:
                        if os.path.exists(url):
                            local_images.append(url)
                            continue
                    except Exception as e:
                        logger.debug("Image is remote: %s, e: %s", url, e)
                        pass
                    # Append a placeholder
                    local_images.append(None)

                    def _fill_placeholder(_url, _index):
                        response = requests.get(url, headers=headers)
                        local_images[_index] = BytesIO(response.content)

                    executor.submit(_fill_placeholder, url, len(local_images) - 1)
            return local_images

        if not isinstance(content, str):
            # TODO(codingl2k1): Optimize _ensure_url

            images = []
            new_content = []
            for c in content:
                c_type = c.get("type")
                if c_type == "image_url":
                    images.append(_ensure_url(c["image_url"]["url"]))
                elif c_type == "text":
                    new_content.append(c["text"])
            if images:
                images = _download(images)
            return "".join(new_content), images
        return content, []

    def get_stop_strs(self) -> List[str]:
        conversation = self._processor.new_chat_template()
        stop_str = conversation.sep2
        return [stop_str]

    def build_generate_kwargs(self, generate_config: Dict):
        max_new_tokens = generate_config.get("max_tokens") or 512
        return {"max_new_tokens": max_new_tokens}

    def build_inputs_from_messages(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ):
        deepseek_messages = []
        for i, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            if role == "user":
                if isinstance(content, str):
                    deepseek_messages.append(
                        {
                            "role": "<|User|>",
                            "content": "<image>\n<|ref|>" + content + "<|/ref|>",
                        }
                    )
                else:
                    content, images = self._message_content_to_deepseek(content)
                    msg: Dict[str, Any] = {
                        "role": "<|User|>",
                        "content": "<image>\n<|ref|>" + content + "<|/ref|>",
                    }
                    if images:
                        msg["images"] = images
                    deepseek_messages.append(msg)
                    deepseek_messages.append({"role": "<|Assistant|>", "content": ""})
            elif role == "assistant":
                deepseek_messages.append({"role": "<|Assistant|>", "content": content})
            else:
                logger.error(
                    f"Unexpected message in messages: role: {role}, message: {message}"
                )

        from .....thirdparty.deepseek_vl2.utils.io import load_pil_images

        # load images and prepare for inputs
        pil_images = load_pil_images(deepseek_messages)
        prepare_inputs = self._processor(
            conversations=deepseek_messages,
            images=pil_images,
            force_batchify=True,
            system_prompt="",
        ).to(self._model.device, self._model.dtype)

        # run image encoder to get the image embeddings
        inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)
        return dict(
            input_ids=prepare_inputs.input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self._tokenizer.eos_token_id,
            bos_token_id=self._tokenizer.bos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

    def build_streaming_iter(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ) -> Tuple[Iterator, int]:
        _inputs = self.build_inputs_from_messages(messages, generate_config)
        configs = self.build_generate_kwargs(generate_config)
        streamer = self._model.language.generate(
            **_inputs,
            **configs,
            do_sample=False,
            use_cache=True,
        )
        return streamer, len(_inputs["input_ids"][0])

    def check_conditions(self, new_text: str) -> Tuple[str, bool]:
        stop_str = self.get_stop_strs()[0]
        if isinstance(new_text, torch.Tensor):
            new_text = self._tokenizer.decode(
                new_text.cpu().tolist(), skip_special_tokens=True
            )

        if new_text.endswith(stop_str):
            new_text = new_text[: -len(stop_str)]
        return new_text, False
