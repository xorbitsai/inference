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
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, Iterator, List, Optional, Tuple, Union

import requests
import torch
from PIL import Image

from ....model.utils import select_device
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionUsage,
)
from ..llm_family import LLMFamilyV1, LLMSpecV1
from .core import PytorchChatModel, PytorchGenerateConfig

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVLChatModel(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = None
        self._model = None

    @classmethod
    def match(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        family = model_family.model_family or model_family.model_name
        if "internvl" in family.lower():
            return True
        return False

    def _get_model_class(self):
        from transformers import AutoModel

        return AutoModel

    def load(self, **kwargs):
        from transformers import AutoModel, AutoTokenizer
        from transformers.generation import GenerationConfig

        if self._check_tensorizer_integrity():
            self._model, self._tokenizer = self._load_tensorizer()
            return

        device = self._pytorch_model_config.get("device", "auto")
        device = select_device(device)
        # for multiple GPU, set back to auto to make multiple devices work
        device = "auto" if device == "cuda" else device

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "device_map": device,
        }

        if "int8" in self.quantization.lower():
            kwargs["load_in_8bit"] = True
        elif 2 == self.model_spec.model_size_in_billions:
            kwargs.pop("device_map")

        self._model = AutoModel.from_pretrained(self.model_path, **kwargs).eval()

        if "int8" not in self.quantization.lower():
            self._model.cuda()

        # Specify hyperparameters for generation
        self._model.generation_config = GenerationConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self._save_tensorizer()

    def _message_content_to_intern(self, content):
        def _load_image(_url):
            if _url.startswith("data:"):
                logging.info("Parse url by base64 decoder.")
                # https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
                # e.g. f"data:image/jpeg;base64,{base64_image}"
                _type, data = _url.split(";")
                _, ext = _type.split("/")
                data = data[len("base64,") :]
                data = base64.b64decode(data.encode("utf-8"))
                return Image.open(BytesIO(data)).convert("RGB")
            else:
                try:
                    response = requests.get(_url)
                except requests.exceptions.MissingSchema:
                    return Image.open(_url).convert("RGB")
                else:
                    return Image.open(BytesIO(response.content)).convert("RGB")

        if not isinstance(content, str):
            texts = []
            image_urls = []
            for c in content:
                c_type = c.get("type")
                if c_type == "text":
                    texts.append(c["text"])
                elif c_type == "image_url":
                    image_urls.append(c["image_url"]["url"])
            image_futures = []
            with ThreadPoolExecutor() as executor:
                for image_url in image_urls:
                    fut = executor.submit(_load_image, image_url)
                    image_futures.append(fut)
            images = [fut.result() for fut in image_futures]
            text = " ".join(texts)
            if len(images) == 0:
                return text, None
            else:
                return text, images
        return content, None

    def _history_content_to_intern(
        self,
        chat_history: List[ChatCompletionMessage],
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
    ):
        def _image_to_piexl_values(images):
            load_images = []
            for image in images:
                if image.startswith("data:"):
                    logging.info("Parse url by base64 decoder.")
                    # https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
                    # e.g. f"data:image/jpeg;base64,{base64_image}"
                    _type, data = image.split(";")
                    _, ext = _type.split("/")
                    data = data[len("base64,") :]
                    data = base64.b64decode(data.encode("utf-8"))
                    img = Image.open(BytesIO(data)).convert("RGB")
                    pixel_value = (
                        self._load_image(img, max_num=6).to(torch.bfloat16).cuda()
                    )
                    load_images.append(pixel_value)
                else:
                    try:
                        response = requests.get(image)
                    except requests.exceptions.MissingSchema:
                        img = Image.open(image).convert("RGB")
                    else:
                        img = Image.open(BytesIO(response.content)).convert("RGB")
                    pixel_value = (
                        self._load_image(img, max_num=6).to(torch.bfloat16).cuda()
                    )
                    load_images.append(pixel_value)
            return torch.cat(tuple(load_images), dim=0)

        history: List[Tuple] = []
        pixel_values = None
        for i in range(0, len(chat_history), 2):
            tmp = []
            images: List[str] = []
            user = chat_history[i]["content"]
            if isinstance(user, List):
                for content in user:
                    c_type = content.get("type")
                    if c_type == "text":
                        tmp.append(content["text"])
                    elif c_type == "image_url" and not history:
                        images.append(content["image_url"]["url"])
                if not history:
                    pixel_values = _image_to_piexl_values(images)
                    image_bs = pixel_values.shape[0]
                    image_tokens = (
                        IMG_START_TOKEN
                        + IMG_CONTEXT_TOKEN * self._model.num_image_token * image_bs
                        + IMG_END_TOKEN
                    )
                    tmp[0] = image_tokens + "\n" + tmp[0]
            else:
                tmp.append(user)
            tmp.append(chat_history[i + 1]["content"])
            history.append(tuple(tmp))
        return history, pixel_values

    def _find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(
        self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
    ):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def _build_transform(self, input_size):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
        return transform

    def _load_image(self, image_file, input_size=448, max_num=6):
        transform = self._build_transform(input_size=input_size)
        images = self._dynamic_preprocess(
            image_file, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def chat(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        if generate_config and generate_config.get("stream"):
            raise Exception(
                f"Chat with model {self.model_family.model_name} does not support stream."
            )
        sanitized_config = {
            "num_beams": 1,
            "max_new_tokens": generate_config.get("max_tokens", 512)
            if generate_config
            else 512,
            "do_sample": False,
        }

        content, image = self._message_content_to_intern(prompt)

        history = None
        if chat_history:
            history, pixel_values = self._history_content_to_intern(chat_history)
        else:
            load_images = []
            for img in image:
                pixel_value = self._load_image(img, max_num=6).to(torch.bfloat16).cuda()
                load_images.append(pixel_value)
            pixel_values = torch.cat(tuple(load_images), dim=0)

        response, history = self._model.chat(
            self._tokenizer,
            pixel_values,
            content,
            sanitized_config,
            history=history,
            return_history=True,
        )
        chunk = Completion(
            id=str(uuid.uuid1()),
            object="text_completion",
            created=int(time.time()),
            model=self.model_uid,
            choices=[
                CompletionChoice(
                    index=0, text=response, finish_reason="stop", logprobs=None
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=-1, completion_tokens=-1, total_tokens=-1
            ),
        )
        return self._to_chat_completion(chunk)
