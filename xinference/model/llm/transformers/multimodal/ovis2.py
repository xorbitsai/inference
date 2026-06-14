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
from threading import Thread
from typing import Any, Dict, Iterator, List, Tuple, Union

import torch
from PIL import Image

from ...llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from ..core import register_non_default_model
from .core import PytorchMultiModalModel

logger = logging.getLogger(__name__)


@register_transformer
@register_non_default_model("Ovis")
class Ovis2ChatModel(PytorchMultiModalModel):
    OVIS_ARCHITECTURES = {"Ovis"}

    def __init__(self, *args, **kws):
        super().__init__(*args, **kws)
        self._text_tokenizer = None
        self._visual_tokenizer = None

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV2", model_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if model_spec.model_format not in ["pytorch", "gptq", "awq", "bnb", "fp4"]:
            return (
                False,
                "Ovis2 transformer supports pytorch/gptq/awq/bnb/fp4 formats only",
            )
        if not model_family.has_architecture(*cls.OVIS_ARCHITECTURES):
            return (
                False,
                f"Model architectures {model_family.architectures} are not Ovis2",
            )
        if "vision" not in model_family.model_ability:
            return False, "Ovis2 transformer requires vision ability"
        return True

    def decide_device(self):
        pass

    def load_processor(self):
        pass

    def load_multimodal_model(self):
        from transformers import AutoModelForCausalLM

        kwargs = self.apply_quantization_config()
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            multimodal_max_length=32768,
            trust_remote_code=True,
            **kwargs,
        ).cuda()
        self._text_tokenizer = self._model.get_text_tokenizer()
        self._visual_tokenizer = self._model.get_visual_tokenizer()

    @staticmethod
    def _parse_messages_ovis(messages: List[Dict]) -> List[Dict]:
        ovis_msgs = []
        for mess in messages:
            contents = mess["content"]
            role = mess["role"]
            if role == "user":
                role = "human"
            elif role == "assistant":
                role = "gpt"
            elif role == "system":
                role = "system"

            for content in contents:
                if content["type"] == "text":
                    ovis_msgs.append({"from": role, "value": content["text"]})

        return ovis_msgs

    @staticmethod
    def _convert_video_tensors_to_pil(video_inputs: List) -> List[Image.Image]:
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

    def _generate_chat_data(self, messages: List[Dict]):
        from qwen_vl_utils import process_vision_info

        messages_ovis = self._parse_messages_ovis(messages)
        max_partition = None
        prompt = messages_ovis[-1]["value"]

        # Preparation for inference
        image_inputs, video_inputs = process_vision_info(messages)

        image_inputs = image_inputs if image_inputs else []

        if image_inputs and len(image_inputs) > 0:
            if len(image_inputs) == 1:
                max_partition = 9
                prompt = f"<image>\n{prompt}"
            else:
                max_partition = len(image_inputs) + 1
                prompt = (
                    "\n".join(
                        [f"Image {i+1}: <image>" for i in range(len(image_inputs))]
                    )
                    + "\n"
                    + prompt
                )
        elif video_inputs and len(video_inputs) > 0:
            if isinstance(video_inputs[0], torch.Tensor):
                # Convert from list[Tensor] to list[Image]
                pil_images = self._convert_video_tensors_to_pil(video_inputs)

                video_inputs = pil_images  # Update video_inputs to PIL image list

            max_partition = 1
            image_inputs = video_inputs
            prompt = "\n".join(["<image>"] * len(video_inputs)) + "\n" + prompt
        else:
            max_partition = 0
            prompt = prompt

        messages_ovis[-1]["value"] = prompt

        # format conversation
        prompt, input_ids, pixel_values = self._model.preprocess_inputs(
            messages_ovis, image_inputs, max_partition=max_partition
        )

        attention_mask = torch.ne(input_ids, self._text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self._model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self._model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(
                dtype=self._visual_tokenizer.dtype, device=self._visual_tokenizer.device
            )
        pixel_values = [pixel_values]

        return input_ids, attention_mask, pixel_values

    def build_generate_kwargs(
        self,
        generate_config: Dict,
    ) -> Dict[str, Any]:
        return dict(
            max_new_tokens=generate_config.get("max_tokens") or 1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=generate_config.get("temperature", None),
            repetition_penalty=None,
            eos_token_id=self._model.generation_config.eos_token_id,
            pad_token_id=self._text_tokenizer.pad_token_id,
            use_cache=True,
        )

    def build_inputs_from_messages(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ):
        msgs = self._transform_messages(messages)
        input_ids, attention_mask, pixel_values = self._generate_chat_data(msgs)
        _, inputs_embeds, _, attention_mask = self._model.merge_multimodal(
            text_input_ids=input_ids,
            text_attention_masks=attention_mask,
            text_labels=None,
            pixel_values=pixel_values,
            left_padding=True,
        )
        inputs_embeds = inputs_embeds.detach()
        torch.cuda.empty_cache()
        return dict(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

    def build_streaming_iter(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ) -> Tuple[Iterator, int]:
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self._text_tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True
        )
        config = self.build_generate_kwargs(generate_config)
        inputs = self.build_inputs_from_messages(messages, generate_config)
        input_ids = inputs.pop("input_ids")

        gen_kwargs = dict(**inputs, **config, streamer=streamer)

        thread = Thread(target=self._model.llm.generate, kwargs=gen_kwargs)
        thread.start()
        return streamer, len(input_ids[0])
