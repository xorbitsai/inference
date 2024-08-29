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
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch

from ....model.utils import select_device
from ....types import ChatCompletion, ChatCompletionChunk, CompletionChunk
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import (
    _decode_image,
    generate_chat_completion,
    generate_completion_chunk,
    parse_messages,
)
from .core import PytorchChatModel, PytorchGenerateConfig

logger = logging.getLogger(__name__)


LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


def recur_move_to(item, tgt, criterion_func):
    """
    This function is copied from https://github.com/THUDM/CogVLM2/blob/main/basic_demo/cli_demo_batch_inference.py
    """
    if criterion_func(item):
        device_copy = item.to(tgt)
        return device_copy
    elif isinstance(item, list):
        return [recur_move_to(v, tgt, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple([recur_move_to(v, tgt, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: recur_move_to(v, tgt, criterion_func) for k, v in item.items()}
    else:
        return item


class CogVLM2VideoModel(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._torch_type = None
        self._device = None
        self._tokenizer = None
        self._model = None

    @classmethod
    def match(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        family = model_family.model_family or model_family.model_name
        if "cogvlm2" in family.lower() and "video" in family.lower():
            return True
        return False

    def load(self, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig

        device = self._pytorch_model_config.get("device", "auto")
        self._device = select_device(device)
        self._torch_type = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        if self._check_tensorizer_integrity():
            self._model, self._tokenizer = self._load_tensorizer()
            return

        if "8-bit" in self.quantization.lower():
            kwargs["load_in_8bit"] = True
        elif "4-bit" in self.quantization.lower():
            kwargs["load_in_4bit"] = True

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self._torch_type,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            **kwargs
        ).eval()

        # Specify hyperparameters for generation
        self._model.generation_config = GenerationConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self._save_tensorizer()

    def _load_video(self, video_path):
        import numpy as np
        from decord import VideoReader, bridge, cpu

        bridge.set_bridge("torch")
        num_frames = 24

        decord_vr = VideoReader(video_path, ctx=cpu(0))
        frame_id_list = None
        total_frames = len(decord_vr)
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break
        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)
        return video_data

    def _message_content_to_cogvlm2(self, content):
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
            video = None
            with ThreadPoolExecutor() as executor:
                for image_url in image_urls:
                    fut = executor.submit(_decode_image, image_url)
                    image_futures.append(fut)
            images = [fut.result() for fut in image_futures]
            for v in video_urls:
                video = self._load_video(v)
            text = " ".join(texts)
            return text, images, video
        return content, [], None

    def _history_content_to_cogvlm2(self, system_prompt: str, chat_history: List[Dict]):
        query = system_prompt
        history: List[Tuple] = []
        pixel_values = None
        video_urls: List[str] = []
        for i in range(0, len(chat_history), 2):
            user = chat_history[i]["content"]
            if isinstance(user, List):
                for content in user:
                    c_type = content.get("type")
                    if c_type == "text":
                        user = content["text"]
                    elif c_type == "image_url" and not pixel_values:
                        pixel_values = _decode_image(content["image_url"]["url"])
                    elif c_type == "video_url":
                        video_urls.append(content["video_url"]["url"])
            assistant = chat_history[i + 1]["content"]
            history.append((user, assistant))
            query = assistant  # type: ignore
        if len(video_urls) > 1:
            raise RuntimeError("Only one video per message is supported")
        video = None
        for v in video_urls:
            video = self._load_video(v)
        return query, history, [pixel_values], video

    def get_query_and_history(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[Dict]] = None,
    ):
        content, image, video = self._message_content_to_cogvlm2(prompt)

        history = []
        history_image = None
        history_video = None
        if chat_history:
            (
                query,
                history,
                history_image,
                history_video,
            ) = self._history_content_to_cogvlm2(
                system_prompt, chat_history  # type: ignore
            )

        if image and history_image:
            history = []
            query = content
        else:
            image = image if image else history_image
            query = content

        if video is not None and history_video is not None:
            history = []
            query = content
        else:
            video = video if video is not None else history_video
            query = content

        return query, image, video, history

    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        system_prompt = ""
        if messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
        stream = generate_config.get("stream", False) if generate_config else False

        sanitized_config = {
            "pad_token_id": 128002,
            "max_new_tokens": generate_config.get("max_tokens", 512)
            if generate_config
            else 512,
        }

        prompt, _, chat_history = parse_messages(messages)
        query, image, video, history = self.get_query_and_history(
            prompt, system_prompt=system_prompt, chat_history=chat_history
        )

        if video is not None:
            image = [video]

        input_by_model = self._model.build_conversation_input_ids(
            self._tokenizer,
            query=query,
            history=history,
            images=image,
            template_version="chat",
        )

        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self._device),
            "token_type_ids": input_by_model["token_type_ids"]
            .unsqueeze(0)
            .to(self._device),
            "attention_mask": input_by_model["attention_mask"]
            .unsqueeze(0)
            .to(self._device),
            "images": [
                [input_by_model["images"][0].to(self._device).to(self._torch_type)]
            ]
            if image is not None
            else None,
        }

        if stream:
            it = self._streaming_chat_response(inputs, sanitized_config)
            return self._to_chat_completion_chunks(it)
        else:
            with torch.no_grad():
                outputs = self._model.generate(**inputs, **sanitized_config)
                outputs = outputs[:, inputs["input_ids"].shape[1] :]
                response = self._tokenizer.decode(outputs[0])
                response = response.split("<|end_of_text|>")[0]

            return generate_chat_completion(self.model_uid, response)

    def _streaming_chat_response(
        self, inputs: Dict, config: Dict
    ) -> Iterator[CompletionChunk]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs["token_type_ids"],
            "images": inputs["images"],
            "max_new_tokens": config["max_new_tokens"],
            "pad_token_id": config["pad_token_id"],
            "streamer": streamer,
        }

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
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
