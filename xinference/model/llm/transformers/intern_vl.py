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
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Optional, Union

import torch

from ....types import ChatCompletion, ChatCompletionChunk
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import (
    _decode_image,
    generate_chat_completion,
    generate_completion_chunk,
    parse_messages,
)
from .core import PytorchChatModel, PytorchGenerateConfig

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _message_content_to_intern(content, image_cnt):
    if not isinstance(content, str):
        texts = []
        image_urls = []
        video_urls = []
        for c in content:
            c_type = c.get("type")
            if c_type == "text":
                texts.append(c["text"])
            elif c_type == "image_url":
                image_urls.append(c["image_url"]["url"])
            elif c_type == "video_url":
                video_urls.append(c["video_url"]["url"])
        if len(video_urls) > 1:
            raise RuntimeError("Only one video per message is supported")
        image_futures = []
        with ThreadPoolExecutor() as executor:
            for image_url in image_urls:
                fut = executor.submit(_decode_image, image_url)
                image_futures.append(fut)
        images = [fut.result() for fut in image_futures]
        videos = []
        for vid_url in video_urls:
            videos.append(_load_video(vid_url, num_segments=8, max_num=1))
        prefix = ""
        for i, _ in enumerate(images):
            prefix += f"Image-{image_cnt + i + 1}: <image>\n\n"

        if len(videos) > 0:
            prefix = "".join(
                [f"Frame{i+1}: <image>\n" for i in range(len(videos[0][1]))]
            )

        text = prefix + " ".join(texts)
        return text, images, videos
    return content, [], []


def _get_prompt_and_chat_history(
    prompt: Union[str, List[Dict]],
    chat_history: Optional[List[Dict]] = None,
):
    # Convert openai history to intern vl history
    images = []
    videos = []
    history = []
    image_cnt = 0
    for h1, h2 in zip(*[iter(chat_history or [])] * 2):
        content1, img, vid = _message_content_to_intern(h1["content"], image_cnt)
        content2, _, _ = _message_content_to_intern(h2["content"], image_cnt)
        history.append([content1, content2])
        images.extend(img)
        image_cnt += len(img)
        videos.extend(vid)

    question, img, vid = _message_content_to_intern(prompt, image_cnt)
    images.extend(img)
    videos.extend(vid)
    return question, history, images, videos


def _build_transform(input_size=448):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
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
    target_aspect_ratio = _find_closest_aspect_ratio(
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


def _load_image(image_file, input_size=448, max_num=12):
    image = image_file.convert("RGB")
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# video multi-round conversation
def _get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    import numpy as np

    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ]
    )
    return frame_indices


def _load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = _build_transform(input_size=input_size)
    frame_indices = _get_index(
        bound, fps, max_frame, first_idx=0, num_segments=num_segments
    )
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = _dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


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
        if "internvl" not in family.lower():
            return False
        if "pytorch" not in model_spec.model_format:
            return False
        return True

    def _get_model_class(self):
        from transformers import AutoModel

        return AutoModel

    # Copy from InternVL page
    # reference: https://huggingface.co/OpenGVLab/InternVL2-8B
    def _split_model(self):
        import math

        device_map = {}
        world_size = torch.cuda.device_count()
        # single gpu
        if world_size == 1:
            return None
        model_size = f"{self.model_spec.model_size_in_billions}B"
        num_layers = {
            "1B": 24,
            "2B": 24,
            "4B": 32,
            "8B": 32,
            "26B": 48,
            "40B": 60,
            "76B": 80,
        }[model_size]
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f"language_model.model.layers.{layer_cnt}"] = i
                layer_cnt += 1
        device_map["vision_model"] = 0
        device_map["mlp1"] = 0
        device_map["language_model.model.tok_embeddings"] = 0
        device_map["language_model.model.embed_tokens"] = 0
        device_map["language_model.output"] = 0
        device_map["language_model.model.norm"] = 0
        device_map["language_model.lm_head"] = 0
        device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
        return device_map

    def load(self, **kwargs):
        from transformers import AutoModel, AutoTokenizer

        if self._check_tensorizer_integrity():
            self._model, self._tokenizer = self._load_tensorizer()
            return

        device = self._split_model()

        kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        if device is not None:
            kwargs["device_map"] = device

        if "8-bit" in self.quantization.lower():
            kwargs["load_in_8bit"] = True
        elif "4-bit" in self.quantization.lower():
            kwargs["load_in_4bit"] = True

        self._model = AutoModel.from_pretrained(self.model_path, **kwargs).eval()

        if device is None and "none" in self.quantization.lower():
            self._model.cuda()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,
        )

    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        from ....thirdparty.internvl.conversation import get_conv_template

        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

        generation_config = {
            "max_new_tokens": generate_config.get("max_tokens", 1024)
            if generate_config
            else 1024,
            "do_sample": False,
        }

        stream = (
            generate_config.get("stream", False)
            if isinstance(generate_config, dict)
            else False
        )
        stream_options = (
            generate_config.get("stream_options", None)
            if isinstance(generate_config, dict)
            else False
        )
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )

        prompt, _, chat_history = parse_messages(messages)
        content, history, images, videos = _get_prompt_and_chat_history(
            prompt, chat_history
        )

        num_patches_list = []
        if len(images) == 1:
            content = content.replace("Image-1: <image>\n\n", "<image>\n")
            history = [
                [item[0].replace("Image-1: <image>\n\n", "<image>\n"), item[1]]
                for item in history
            ]
            pixel_values = _load_image(images[-1], max_num=12).to(torch.bfloat16).cuda()
            num_patches_list = (
                [pixel_values.shape[0]] if pixel_values is not None else []
            )
        elif len(images) > 1:
            pixel_values = [
                _load_image(img, max_num=12).to(torch.bfloat16).cuda() for img in images
            ]
            num_patches_list = [values.size(0) for values in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = None

        if len(videos) > 0:
            pixel_values = videos[0][0]
            num_patches_list = videos[0][1]

        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = self._tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self._model.img_context_token_id = img_context_token_id

        template = get_conv_template(self._model.template)
        template.system_message = self._model.system_message
        eos_token_id = self._tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], content)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self._model.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = self._tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].cuda()
        attention_mask = model_inputs["attention_mask"].cuda()
        generation_config["eos_token_id"] = eos_token_id
        generate_kwargs = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        generate_kwargs.update(generation_config)

        if stream:
            chunk = self._generate_stream(generate_kwargs, input_ids, include_usage)
            return self._to_chat_completion_chunks(chunk)
        else:
            return self._generate(generate_kwargs, input_ids, template)

    def _generate(self, generate_kwargs, input_ids, template) -> ChatCompletion:
        prompt_tokens = len(input_ids[0])
        generation_output = self._model.generate(**generate_kwargs)
        completion_tokens = len(generation_output[0])
        response = self._tokenizer.batch_decode(
            generation_output, skip_special_tokens=True
        )[0]
        response = response.split(template.sep)[0].strip()
        return generate_chat_completion(
            self.model_uid,
            response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _generate_stream(self, generate_kwargs, input_ids, include_usage):
        from threading import Thread

        from transformers import TextIteratorStreamer

        # Initialize the streamer
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10
        )
        # Define the generation configuration
        generate_kwargs["streamer"] = streamer
        # Start the model chat in a separate thread
        thread = Thread(
            target=self._model.generate,
            kwargs=generate_kwargs,
        )
        thread.start()

        completion_id = str(uuid.uuid1())
        prompt_tokens = len(input_ids[0])
        total_tokens, completion_tokens = 0, 0
        # Loop through the streamer to get the new text as it is generated
        for i, new_text in enumerate(streamer):
            if new_text == self._model.conv_template.sep:
                break
            completion_tokens = max(completion_tokens, len(streamer.token_cache))
            total_tokens = prompt_tokens + completion_tokens
            yield generate_completion_chunk(
                chunk_text=new_text,
                finish_reason=None,
                chunk_id=completion_id,
                model_uid=self.model_uid,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
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
