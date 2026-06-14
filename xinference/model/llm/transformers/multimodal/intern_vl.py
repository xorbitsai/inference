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
import math
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch

from ...llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from ...utils import _decode_image, parse_messages
from ..core import register_non_default_model
from .core import PytorchMultiModalModel

logger = logging.getLogger(__name__)


@register_transformer
@register_non_default_model("InternVLChatModel")
class InternVLChatModel(PytorchMultiModalModel):
    INTERN_VL_ARCHITECTURES = {"InternVLChatModel"}

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV2", model_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if not model_family.has_architecture(*cls.INTERN_VL_ARCHITECTURES):
            return (
                False,
                f"Model architectures {model_family.architectures} are not InternVL3",
            )
        if "vision" not in model_family.model_ability:
            return False, "InternVL transformer requires vision ability"
        return True

    def decide_device(self):
        from transformers import AutoConfig

        device_map = {}
        world_size = torch.cuda.device_count()
        # single gpu
        if world_size == 1:
            self._device = device_map
            return
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        num_layers = config.llm_config.num_hidden_layers

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
        self._device = device_map

    def load_processor(self):
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False
        )

    def load_multimodal_model(self):
        from transformers import AutoModel

        kwargs: Dict[str, Any] = {  # type: ignore
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if self._device:
            kwargs["device_map"] = self._device
        kwargs = self.apply_quantization_config(kwargs)

        self._model = AutoModel.from_pretrained(self.model_path, **kwargs).eval()

        if not self._device and "none" in self.quantization.lower():
            self._model.cuda()

    def _build_transform(self, input_size=448):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
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

    # video multi-round conversation
    @staticmethod
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
        self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
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

    def _load_video(
        self, video_path, bound=None, input_size=448, max_num=1, num_segments=32
    ):
        from decord import VideoReader, cpu
        from PIL import Image

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = self._build_transform(input_size=input_size)
        frame_indices = self._get_index(
            bound, fps, max_frame, first_idx=0, num_segments=num_segments
        )
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            img = self._dynamic_preprocess(
                img, image_size=input_size, use_thumbnail=True, max_num=max_num
            )
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def _message_content_to_intern(self, content, image_cnt):
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
                videos.append(self._load_video(vid_url, num_segments=8, max_num=1))
            prefix = ""
            for i, _ in enumerate(images):
                prefix += f"Image-{image_cnt + i + 1}: <image>\n\n"

            if len(videos) > 0:
                prefix = "".join(
                    [f"Frame{i + 1}: <image>\n" for i in range(len(videos[0][1]))]
                )

            text = prefix + " ".join(texts)
            return text, images, videos
        return content, [], []

    def _get_prompt_and_chat_history(
        self,
        prompt: Union[str, List[Dict]],
        chat_history: Optional[List[Dict]] = None,
    ):
        # Convert openai history to intern vl history
        images = []
        videos = []
        history = []
        image_cnt = 0
        for h1, h2 in zip(*[iter(chat_history or [])] * 2):
            content1, img, vid = self._message_content_to_intern(
                h1["content"], image_cnt
            )
            content2, _, _ = self._message_content_to_intern(h2["content"], image_cnt)
            history.append([content1, content2])
            images.extend(img)
            image_cnt += len(img)
            videos.extend(vid)

        question, img, vid = self._message_content_to_intern(prompt, image_cnt)
        images.extend(img)
        videos.extend(vid)
        return question, history, images, videos

    def _load_image(self, image_file, input_size=448, max_num=12):
        image = image_file.convert("RGB")
        transform = self._build_transform(input_size=input_size)
        images = self._dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def build_inputs_from_messages(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ):
        from .....thirdparty.internvl.conversation import get_conv_template

        prompt, _, chat_history = parse_messages(messages)
        content, history, images, videos = self._get_prompt_and_chat_history(
            prompt, chat_history
        )
        num_patches_list = []
        if len(images) == 1:
            content = content.replace("Image-1: <image>\n\n", "<image>\n")
            history = [
                [item[0].replace("Image-1: <image>\n\n", "<image>\n"), item[1]]
                for item in history
            ]
            pixel_values = (
                self._load_image(images[-1], max_num=12).to(torch.bfloat16).cuda()
            )
            num_patches_list = (
                [pixel_values.shape[0]] if pixel_values is not None else []
            )
        elif len(images) > 1:
            pixel_values = [
                self._load_image(img, max_num=12).to(torch.bfloat16).cuda()
                for img in images
            ]
            num_patches_list = [values.size(0) for values in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = None

        if len(videos) > 0:
            pixel_values = videos[0][0]
            num_patches_list = videos[0][1]

        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

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

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "eos_token_id": eos_token_id,
        }

    def build_generate_kwargs(
        self,
        generate_config: Dict,
    ) -> Dict[str, Any]:
        return {
            "max_new_tokens": generate_config.get("max_tokens") or 1024,
            "do_sample": False,
            "temperature": generate_config.get("temperature", None),
        }

    def build_streaming_iter(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ) -> Tuple[Iterator, int]:
        from transformers import TextIteratorStreamer

        # Initialize the streamer
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10
        )

        configs = self.build_generate_kwargs(generate_config)
        inputs = self.build_inputs_from_messages(messages, generate_config)
        generate_kwargs = {**inputs, **configs, "streamer": streamer}
        thread = Thread(
            target=self._model.generate,
            kwargs=generate_kwargs,
        )
        thread.start()
        return streamer, len(inputs["input_ids"][0])

    def check_conditions(self, new_text: str) -> Tuple[str, bool]:
        if new_text == self._model.conv_template.sep:
            return "", True
        return new_text, False
