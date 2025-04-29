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
import base64
import logging
import os.path
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import requests
import torch

from ....model.utils import select_device
from ....types import ChatCompletion, ChatCompletionChunk, CompletionChunk
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import generate_chat_completion, generate_completion_chunk
from .core import PytorchChatModel, PytorchGenerateConfig
from .utils import cache_clean

logger = logging.getLogger(__name__)


class DeepSeekVL2ChatModel(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = None
        self._model = None
        self._vl_chat_processor = None
        self._type = None

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        llm_family = model_family.model_family or model_family.model_name
        if "deepseek-vl2" == llm_family.lower():
            return True
        return False

    def load(self):
        from transformers import AutoModelForCausalLM

        from ....thirdparty.deepseek_vl2.models import (
            DeepseekVLV2ForCausalLM,
            DeepseekVLV2Processor,
        )

        self._device = self._pytorch_model_config.get("device", "auto")
        self._device = select_device(self._device)
        self._type = torch.bfloat16
        kwargs = self.apply_bnb_quantization()

        # specify the path to the model
        self._vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(  # type: ignore
            self.model_path
        )
        self._tokenizer = self._vl_chat_processor.tokenizer

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
                new_content.insert(0, "<image_placeholder>")
                images = _download(images)
            return "".join(new_content), images
        return content, []

    @cache_clean
    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        if not generate_config:
            generate_config = {}

        stream = generate_config.get("stream", False)
        stream_options = generate_config.pop("stream_options", None)
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )

        prompt = ""
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
                if i == len(messages) - 1:
                    prompt = "<image>\n<|ref|>" + content + "<|/ref|>"
            elif role == "assistant":
                deepseek_messages.append({"role": "<|Assistant|>", "content": content})
            else:
                logger.error(
                    f"Unexpected message in messages: role: {role}, message: {message}"
                )

        from ....thirdparty.deepseek_vl2.utils.io import load_pil_images

        # load images and prepare for inputs
        pil_images = load_pil_images(deepseek_messages)
        prepare_inputs = self._vl_chat_processor(
            conversations=deepseek_messages,
            images=pil_images,
            force_batchify=True,
            system_prompt="",
        ).to(self._model.device, self._model.dtype)

        # run image encoder to get the image embeddings
        inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)

        max_new_tokens = generate_config.get("max_tokens", 512)
        conversation = self._vl_chat_processor.new_chat_template()
        stop_str = conversation.sep2

        streamer = self._model.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self._tokenizer.eos_token_id,
            bos_token_id=self._tokenizer.bos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        if stream:
            it = self._generate_stream(streamer, stop_str, include_usage, prompt)
            return self._to_chat_completion_chunks(it)
        else:
            return self._generate(streamer, stop_str)

    def _generate(self, streamer, stop_str) -> ChatCompletion:
        generated_text = ""

        for new_text in streamer:
            if isinstance(new_text, torch.Tensor):
                new_text = self._tokenizer.decode(
                    new_text.cpu().tolist(), skip_special_tokens=True
                )

            if new_text.endswith(stop_str):
                new_text = new_text[: -len(stop_str)]

            generated_text += new_text

        return generate_chat_completion(self.model_uid, generated_text)

    def _generate_stream(
        self, streamer, stop_str, include_usage, prompt
    ) -> Iterator[CompletionChunk]:
        completion_id = str(uuid.uuid1())
        prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
        input_ids = self._tokenizer(prompt).input_ids
        prompt_tokens = len(input_ids)
        for i, new_text in enumerate(streamer):
            if new_text.endswith(stop_str):
                new_text = new_text[: -len(stop_str)]
            completion_tokens = i
            total_tokens = prompt_tokens + completion_tokens
            yield generate_completion_chunk(
                chunk_text=new_text,
                finish_reason=None,
                chunk_id=completion_id,
                model_uid=self.model_uid,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                has_choice=True,
                has_content=True,
            )
        yield generate_completion_chunk(
            chunk_text=None,
            finish_reason="stop",
            chunk_id=completion_id,
            model_uid=self.model_uid,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            has_choice=True,
            has_content=False,
        )

        if include_usage:
            yield generate_completion_chunk(
                chunk_text=None,
                finish_reason=None,
                chunk_id=completion_id,
                model_uid=self.model_uid,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                has_choice=False,
                has_content=False,
            )
