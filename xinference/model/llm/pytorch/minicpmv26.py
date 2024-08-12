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
from typing import Dict, Iterator, List, Optional, Union

import requests
import torch
from decord import VideoReader, cpu
from PIL import Image

from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
)
from ...utils import select_device
from ..llm_family import LLMFamilyV1, LLMSpecV1
from .core import PytorchChatModel, PytorchGenerateConfig

logger = logging.getLogger(__name__)


class MiniCPMV26Model(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device = None
        self._tokenizer = None
        self._model = None

    @classmethod
    def match(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        family = model_family.model_family or model_family.model_name
        if "MiniCPM-V-2.6".lower() in family.lower():
            return True
        return False

    def _get_model_class(self):
        from transformers import AutoModel

        return AutoModel

    def load(self, **kwargs):
        from transformers import AutoModel, AutoTokenizer
        from transformers.generation import GenerationConfig

        device = self._pytorch_model_config.get("device", "auto")
        self._device = select_device(device)
        self._device = (
            "auto"
            if self._device == "cuda" and self.quantization is None
            else self._device
        )

        if "int4" in self.model_path and device == "mps":
            logger.error(
                "Error: running int4 model with bitsandbytes on Mac is not supported right now."
            )
            exit()

        if self._check_tensorizer_integrity():
            self._model, self._tokenizer = self._load_tensorizer()
            return

        if "int4" in self.model_path:
            model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        else:
            model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=self._device,
            )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._model = model.eval()
        self._tokenizer = tokenizer

        # Specify hyperparameters for generation
        self._model.generation_config = GenerationConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self._save_tensorizer()

    def _message_content_to_chat(self, content):
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

        MAX_NUM_FRAMES = 64

        def encode_video(video_path):
            def uniform_sample(l, n):
                gap = len(l) / n
                idxs = [int(i * gap + gap / 2) for i in range(n)]
                return [l[i] for i in idxs]

            vr = VideoReader(video_path, ctx=cpu(0))
            sample_fps = round(vr.get_avg_fps() / 1)  # FPS
            frame_idx = [i for i in range(0, len(vr), sample_fps)]
            if len(frame_idx) > MAX_NUM_FRAMES:
                frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
            frames = vr.get_batch(frame_idx).asnumpy()
            frames = [Image.fromarray(v.astype("uint8")) for v in frames]
            print("num frames:", len(frames))
            return frames

        def _load_video(_url):
            frames = None
            if _url.startswith("data:"):
                raise RuntimeError("Only video url format is supported")
            else:
                frames = encode_video(_url)
            return frames

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
            image_futures = []
            with ThreadPoolExecutor() as executor:
                for image_url in image_urls:
                    fut = executor.submit(_load_image, image_url)
                    image_futures.append(fut)
            images = [fut.result() for fut in image_futures]
            frames = []
            if len(video_urls) > 1:
                raise RuntimeError("Only one video per message is supported")
            for v in video_urls:
                frames = _load_video(v)
            text = " ".join(texts)
            return text, images, frames
        return content, [], []

    def chat(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        stream = generate_config.get("stream", False) if generate_config else False
        videoExisted = False

        content, images_chat, video_frames = self._message_content_to_chat(prompt)
        if len(video_frames) > 0:
            videoExisted = True
            images_chat = video_frames

        msgs = []
        query_to_response: List[Dict] = []
        for h in chat_history or []:
            images_history = []
            role = h["role"]
            content_h, images_tmp, video_frames_h = self._message_content_to_chat(
                h["content"]
            )
            if images_tmp != []:
                images_history = images_tmp
            if len(video_frames_h) > 0:
                videoExisted = True
                images_history = video_frames_h
            if len(query_to_response) == 0 and role == "user":
                query_to_response.append(
                    {"role": "user", "content": images_history + [content_h]}
                )
            if len(query_to_response) == 1 and role == "assistant":
                query_to_response.append(
                    {"role": "assistant", "content": images_history + [content_h]}
                )
            if len(query_to_response) == 2:
                msgs.extend(query_to_response)
                query_to_response = []
        msgs.append({"role": "user", "content": images_chat + [content]})

        # Set decode params for video
        params = {}
        if videoExisted:
            params = {
                "use_image_id": False,
                "max_slice_nums": 1
            }

        chat = self._model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self._tokenizer,
            sampling=True,
            **generate_config,
            **params,
        )
        if stream:
            it = self.chat_stream(chat)
            return self._to_chat_completion_chunks(it)
        else:
            c = Completion(
                id=str(uuid.uuid1()),
                object="text_completion",
                created=int(time.time()),
                model=self.model_uid,
                choices=[
                    CompletionChoice(
                        index=0, text=chat, finish_reason="stop", logprobs=None
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=-1, completion_tokens=-1, total_tokens=-1
                ),
            )
            return self._to_chat_completion(c)

    def chat_stream(self, chat) -> Iterator[CompletionChunk]:
        completion_id = str(uuid.uuid1())
        for new_text in chat:
            completion_choice = CompletionChoice(
                text=new_text, index=0, logprobs=None, finish_reason=None
            )
            chunk = CompletionChunk(
                id=completion_id,
                object="text_completion",
                created=int(time.time()),
                model=self.model_uid,
                choices=[completion_choice],
            )
            completion_usage = CompletionUsage(
                prompt_tokens=-1,
                completion_tokens=-1,
                total_tokens=-1,
            )
            chunk["usage"] = completion_usage
            yield chunk

        completion_choice = CompletionChoice(
            text="", index=0, logprobs=None, finish_reason="stop"
        )
        chunk = CompletionChunk(
            id=completion_id,
            object="text_completion",
            created=int(time.time()),
            model=self.model_uid,
            choices=[completion_choice],
        )
        completion_usage = CompletionUsage(
            prompt_tokens=-1,
            completion_tokens=-1,
            total_tokens=-1,
        )
        chunk["usage"] = completion_usage
        yield chunk
