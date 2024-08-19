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
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch

from ....core.scheduler import InferenceRequest
from ....model.utils import select_device
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
)
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import _decode_image
from .core import PytorchChatModel, PytorchGenerateConfig
from .utils import get_max_src_len

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

    def _history_content_to_cogvlm2(
        self, system_prompt: str, chat_history: List[ChatCompletionMessage]
    ):
        query = system_prompt
        history: List[Tuple] = []
        pixel_values = None
        video_urls = []
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
        chat_history: Optional[List[ChatCompletionMessage]] = None,
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
            history = ([],)
            query = content
        else:
            video = video if video is not None else history_video
            query = content

        return query, image, video, history

    def chat(
        self,
        prompt: Union[str, List[Dict]],
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        system_prompt = system_prompt if system_prompt else ""
        stream = generate_config.get("stream", False) if generate_config else False

        sanitized_config = {
            "pad_token_id": 128002,
            "max_new_tokens": generate_config.get("max_tokens", 512)
            if generate_config
            else 512,
        }

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
            chunk = CompletionChunk(
                id=completion_id,
                object="text_completion",
                created=int(time.time()),
                model=self.model_uid,
                choices=[
                    CompletionChoice(
                        index=0, text=new_text, finish_reason=None, logprobs=None
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=-1, completion_tokens=-1, total_tokens=-1
                ),
            )
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
            usage=CompletionUsage(
                prompt_tokens=-1, completion_tokens=-1, total_tokens=-1
            ),
        )
        yield chunk

    @staticmethod
    def build_position_ids(x, attention_mask=None):
        """
        Copied from https://huggingface.co/THUDM/cogvlm2-llama3-chinese-chat-19B-int4/blob/main/modeling_cogvlm.py
        """
        # Fix: 参考官方开源代码
        if attention_mask is not None:
            tmp = x.clone()
            tmp[~(attention_mask.bool())] = -1
        else:
            tmp = x.clone()
        # image boi eoi token as LANGUAGE_TOKEN_TYPE
        is_boi_eoi = torch.zeros_like(x, dtype=torch.bool)
        is_boi_eoi[:, 1:] |= (tmp[:, 1:] == VISION_TOKEN_TYPE) & (
            tmp[:, :-1] == LANGUAGE_TOKEN_TYPE
        )
        is_boi_eoi[:, 0] |= tmp[:, 0] == VISION_TOKEN_TYPE
        is_boi_eoi[:, :-1] |= (tmp[:, :-1] == VISION_TOKEN_TYPE) & (
            tmp[:, 1:] == LANGUAGE_TOKEN_TYPE
        )
        is_boi_eoi[:, -1] |= tmp[:, -1] == VISION_TOKEN_TYPE
        tmp[is_boi_eoi] = LANGUAGE_TOKEN_TYPE
        # final position ids
        y = torch.zeros_like(x, dtype=torch.long)
        y[:, 1:] = (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE) | (
            (tmp[:, 1:] == VISION_TOKEN_TYPE) & (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE)
        )
        y = y.cumsum(dim=-1)
        return y

    def get_dtype(self):
        return self._torch_type

    def _get_full_prompt(self, prompt, system_prompt, chat_history, tools):
        query, image, video, history = self.get_query_and_history(
            prompt, system_prompt=system_prompt, chat_history=chat_history
        )

        if video:
            image = [video]

        input_by_model: dict = self._model.build_conversation_input_ids(  # type: ignore
            self._tokenizer,
            query=query,
            history=history,
            images=image,
            template_version="chat",
        )
        return {
            "input_ids": input_by_model["input_ids"],  # seq_len
            "token_type_ids": input_by_model["token_type_ids"],  # seq_len
            "attention_mask": input_by_model["attention_mask"],  # seq_len
            "images": input_by_model["images"],
        }

    def prepare_sanitize_generate_config(self, req: InferenceRequest):
        """
        See https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B/blob/main/generation_config.json
        """
        raw_config = req.inference_kwargs.get("raw_params", {})
        temperature = raw_config.get("temperature", None)
        if temperature is None:
            raw_config["temperature"] = 0.6
        top_p = raw_config.get("top_p", None)
        if top_p is None:
            raw_config["top_p"] = 0.9
        return raw_config

    def build_prefill_kwargs(self, prompts: List, req_list: List[InferenceRequest]):
        context_len = self.get_context_len()
        assert isinstance(prompts[0], dict)
        images = []
        max_length = float("-inf")
        for i, feature in enumerate(prompts):
            req = req_list[i]
            if "images" in feature:
                images.append(feature.pop("images", None))
            max_src_len = get_max_src_len(context_len, req)
            input_ids = feature["input_ids"][-max_src_len:]
            req.prompt_tokens = input_ids.tolist()
            feature["input_ids"] = input_ids
            feature["token_type_ids"] = feature["token_type_ids"][-max_src_len:]
            feature["attention_mask"] = feature["attention_mask"][-max_src_len:]
            req.extra_kwargs["attention_mask_seq_len"] = feature[
                "attention_mask"
            ].shape[0]
            max_length = max(len(input_ids), max_length)

        def pad_to_max_length_internal(feature, max_len, idx):
            padding_length = max_len - len(feature["input_ids"])
            req_list[idx].padding_len = padding_length
            feature["input_ids"] = torch.cat(
                [torch.full((padding_length,), 0), feature["input_ids"]]
            )
            feature["token_type_ids"] = torch.cat(
                [
                    torch.zeros(padding_length, dtype=torch.long),
                    feature["token_type_ids"],
                ]
            )
            feature["attention_mask"] = torch.cat(
                [
                    torch.zeros(padding_length, dtype=torch.long),
                    feature["attention_mask"],
                ]
            )
            return feature

        features = [
            pad_to_max_length_internal(feature, max_length, i)
            for i, feature in enumerate(prompts)
        ]
        batch = {
            key: torch.stack([feature[key] for feature in features])
            for key in features[0].keys()
        }

        position_ids = self.build_position_ids(batch["token_type_ids"])
        batch["position_ids"] = position_ids

        for i in range(len(prompts)):
            req = req_list[i]
            req.extra_kwargs["max_position_id"] = position_ids[i : i + 1, -1].item()

        if images:
            batch["images"] = images

        batch = recur_move_to(
            batch, self._device, lambda x: isinstance(x, torch.Tensor)
        )
        dtype = self.get_dtype()
        if dtype:
            batch = recur_move_to(
                batch,
                dtype,
                lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x),
            )
        return batch

    def build_decode_token_type_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        token_type_ids = torch.full(
            (batch_size, 1), fill_value=1, dtype=torch.long, device=self._device
        )
        return token_type_ids

    def build_decode_position_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        tmp = []
        for r in reqs:
            r.extra_kwargs["max_position_id"] += 1
            tmp.append(r.extra_kwargs["max_position_id"])
        position_ids = torch.as_tensor(
            tmp, device=self._device, dtype=torch.long
        ).unsqueeze(1)
        return position_ids
