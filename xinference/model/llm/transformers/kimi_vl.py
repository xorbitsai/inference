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
import uuid
from typing import Dict, Iterator, List, Optional, Union

import torch
from PIL import Image
from torch.cuda import temperature

from ....model.utils import select_device
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    CompletionChunk,
)
from ..llm_family import LLMFamilyV1, LLMSpecV1, register_transformer
from ..utils import generate_chat_completion, generate_completion_chunk
from .core import PytorchChatModel, PytorchGenerateConfig, register_non_default_model
from .utils import cache_clean

logger = logging.getLogger(__name__)

@register_transformer
@register_non_default_model(
    "Kimi-VL-A3B-Instruct", "Kimi-VL-A3B-Thinking"
)
class KimiVLChatModel(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = None
        self._model = None
        self._device = None
        self._processor = None

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if model_spec.model_format not in ["pytorch", "gptq", "awq"]:
            return False
        llm_family = model_family.model_family or model_family.model_name
        if "kimi-vl-".lower() in llm_family.lower():
            return True
        return False

    def load(self):
        import importlib.util
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._device = self._pytorch_model_config.get("device", "auto")
        self._device = select_device(self._device)

        # 构建模型加载参数
        model_kwargs = {
            "pretrained_model_name_or_path": self.model_path,
            "device_map": self._device,
            "trust_remote_code": True,
            "torch_dtype": "auto"
        }

        flash_attn_installed = importlib.util.find_spec("flash_attn") is not None
        if flash_attn_installed:
            model_kwargs.update({
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2"
            })

        kwargs = self.apply_bnb_quantization()
        model_kwargs.update(kwargs)

        self._model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

    @cache_clean
    def chat(
        self,
        messages: List[ChatCompletionMessage],  # type: ignore
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        messages = self._transform_messages(messages)

        generate_config = generate_config if generate_config else {}

        stream = generate_config.get("stream", False) if generate_config else False

        if stream:
            raise NotImplementedError(
                "Kimi-VL-A3B-Instruct does not support stream generation yet."
            )
            # it = self._generate_stream(messages, generate_config)
            # return self._to_chat_completion_chunks(it)
        else:
            c = self._generate(messages, generate_config)
            return c

    def _generate(
        self, messages: List, config: PytorchGenerateConfig = {}
    ) -> ChatCompletion:
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        image = None
        inputs = self._processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True).to(self._model.device)
        generated_ids = self._model.generate(**inputs, max_new_tokens=config.get("max_tokens", 2048), temperature=config.get("temperature", 0.7))
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        logger.info(f"============输出: {response}")
        return generate_chat_completion(self.model_uid, response)

    def _generate_stream(
        self, messages: List, config: PytorchGenerateConfig = {}
    ) -> Iterator[CompletionChunk]:
        pass

    def _convert_video_tensors_to_pil(self, video_inputs: List) -> List[Image.Image]:
        """Convert video tensors to a list of PIL images"""
        from torchvision import transforms

        to_pil = transforms.ToPILImage()
        pil_images = []

        for video_tensor_4d in video_inputs:
            if isinstance(video_tensor_4d, torch.Tensor):
                # Verify it's a 4D tensor
                if video_tensor_4d.ndim == 4:
                    # Iterate through the first dimension (frames) of 4D tensor
                    for i in range(video_tensor_4d.size(0)):
                        frame_tensor_3d = video_tensor_4d[
                            i
                        ]  # Get 3D frame tensor [C, H, W]
                        # Ensure tensor is on CPU before conversion
                        if frame_tensor_3d.is_cuda:
                            frame_tensor_3d = frame_tensor_3d.cpu()
                        try:
                            pil_image = to_pil(frame_tensor_3d)
                            pil_images.append(pil_image)
                        except Exception as e:
                            logger.error(
                                f"Error converting frame {i} to PIL Image: {e}"
                            )
                            # Can choose to skip this frame or handle error differently
                else:
                    logger.warning(
                        f"Expected 4D tensor in video_inputs, but got {video_tensor_4d.ndim}D. Skipping this tensor."
                    )
            elif isinstance(video_tensor_4d, Image.Image):
                # If fetch_video returns Image list, add directly
                pil_images.append(video_tensor_4d)
            else:
                logger.warning(
                    f"Unexpected type in video_inputs: {type(video_tensor_4d)}. Skipping."
                )

        return pil_images
