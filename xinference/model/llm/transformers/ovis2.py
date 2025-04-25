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

from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    CompletionChunk,
)
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import generate_chat_completion, generate_completion_chunk
from .core import PytorchChatModel, PytorchGenerateConfig
from .utils import cache_clean

logger = logging.getLogger(__name__)


class Ovis2ChatModel(PytorchChatModel):
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
        if "ovis2".lower() in llm_family.lower():
            return True
        return False

    def load(self):
        from transformers import AutoModelForCausalLM

        # load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            multimodal_max_length=32768,
            trust_remote_code=True,
        ).cuda()
        self._text_tokenizer = self._model.get_text_tokenizer()
        self._visual_tokenizer = self._model.get_visual_tokenizer()

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
            # raise NotImplementedError("Stream is not supported for Ovis2 model.")
            it = self._generate_stream(messages, generate_config)
            return self._to_chat_completion_chunks(it)
        else:
            c = self._generate(messages, generate_config)
            return c

    def _generate(
        self, messages: List, config: PytorchGenerateConfig = {}
    ) -> ChatCompletion:
        input_ids, attention_mask, pixel_values, gen_kwargs = self._generate_chat_data(
            messages, config
        )

        # generate output
        with torch.inference_mode():
            gen_kwargs.update(
                dict(
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                )
            )

            output_ids = self._model.generate(
                input_ids,
                **gen_kwargs,
            )[0]
            output = self._text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return generate_chat_completion(self.model_uid, output)

    def _generate_stream(
        self, messages: List, config: PytorchGenerateConfig = {}
    ) -> Iterator[CompletionChunk]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        input_ids, attention_mask, pixel_values, gen_kwargs = self._generate_chat_data(
            messages, config
        )

        _, inputs_embeds, _, attention_mask = self._model.merge_multimodal(
            text_input_ids=input_ids,
            text_attention_masks=attention_mask,
            text_labels=None,
            pixel_values=pixel_values,
            left_padding=True,
        )

        streamer = TextIteratorStreamer(
            self._text_tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs.update(
            dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                streamer=streamer,
            )
        )

        inputs_embeds = inputs_embeds.detach()
        torch.cuda.empty_cache()

        thread = Thread(target=self._model.llm.generate, kwargs=gen_kwargs)
        thread.start()

        completion_id = str(uuid.uuid1())

        for new_text in streamer:
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

    def parse_messages_ovis(self, messages: List[Dict]) -> List[Dict]:
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

    def _generate_chat_data(
        self, messages: List[Dict], config: PytorchGenerateConfig = {}
    ):
        from qwen_vl_utils import process_vision_info

        messages_ovis = self.parse_messages_ovis(messages)
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

        gen_kwargs = dict(
            max_new_tokens=config.get("max_tokens", 1024),
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=config.get("temperature", None),
            repetition_penalty=None,
            eos_token_id=self._model.generation_config.eos_token_id,
            pad_token_id=self._text_tokenizer.pad_token_id,
            use_cache=True,
        )

        return input_ids, attention_mask, pixel_values, gen_kwargs

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
