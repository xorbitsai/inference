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
from PIL import Image

from ....core.scheduler import InferenceRequest
from ....types import ChatCompletion, ChatCompletionChunk, CompletionChunk
from ...utils import select_device
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import (
    _decode_image,
    generate_chat_completion,
    generate_completion_chunk,
    parse_messages,
)
from .core import PytorchChatModel, PytorchGenerateConfig

logger = logging.getLogger(__name__)


class MiniCPMV26Model(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device = None
        self._tokenizer = None
        self._model = None
        self._processor = None

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
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
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
        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._device = self._model.device
        self._save_tensorizer()

    def _message_content_to_chat(self, content):
        MAX_NUM_FRAMES = 64

        def encode_video(video_path):
            from decord import VideoReader, cpu

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
            logger.info(
                f"Num frames: {len(frames)} when decoding video for {self.model_uid}"
            )
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
                    fut = executor.submit(_decode_image, image_url)
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

    def _convert_to_specific_style(self, messages: List[Dict]) -> Tuple:
        video_existed = False
        prompt, _, chat_history = parse_messages(messages)

        content, images_chat, video_frames = self._message_content_to_chat(prompt)
        if len(video_frames) > 0:
            video_existed = True
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
                video_existed = True
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
        return msgs, video_existed

    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        stream = generate_config.get("stream", False) if generate_config else False
        msgs, video_existed = self._convert_to_specific_style(messages)

        # Set decode params for video
        params = {}
        if video_existed:
            params = {"use_image_id": False, "max_slice_nums": 1}

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
            return generate_chat_completion(self.model_uid, chat)

    def chat_stream(self, chat) -> Iterator[CompletionChunk]:
        completion_id = str(uuid.uuid1())
        for new_text in chat:
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

    def prepare_sanitize_generate_config(self, req: InferenceRequest):
        """
        Refer to https://huggingface.co/openbmb/MiniCPM-V-2_6/blob/main/modeling_minicpmv.py
        """
        raw_config = req.inference_kwargs.get("raw_params", {})
        temperature = raw_config.get("temperature", None)
        if temperature is None:
            raw_config["temperature"] = 0.7
        top_p = raw_config.get("top_p", None)
        if top_p is None:
            raw_config["top_p"] = 0.8
        top_k = raw_config.get("top_k", None)
        if top_k is None:
            raw_config["top_k"] = 100
        repetition_penalty = raw_config.get("repetition_penalty", None)
        if repetition_penalty is None:
            raw_config["repetition_penalty"] = 1.05
        return raw_config

    def _handle_input_ids_and_images(self, msgs: List[Dict]) -> Dict:
        """
        Copied from https://huggingface.co/openbmb/MiniCPM-V-2_6/blob/main/modeling_minicpmv.py#L315
        """
        from copy import deepcopy

        copy_msgs = deepcopy(msgs)

        images = []
        for i, msg in enumerate(copy_msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["user", "assistant"]
            if i == 0:
                assert role == "user", "The role of first msg should be user"
            if isinstance(content, str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg["content"] = "\n".join(cur_msgs)

        return {
            "prompt": self._processor.tokenizer.apply_chat_template(
                copy_msgs, tokenize=False, add_generation_prompt=True
            ),
            "input_image": images,
        }

    def _get_full_prompt(self, messages: List[Dict], tools):
        msgs, video_existed = self._convert_to_specific_style(messages)
        if video_existed:
            raise RuntimeError(
                f"Continuous batching does not support video inputs for this model: {self.model_uid}"
            )
        return self._handle_input_ids_and_images(msgs)

    def build_prefill_kwargs(self, prompts: List, req_list: List[InferenceRequest]):
        prompts_lists = [x["prompt"] for x in prompts]
        input_images_lists = [x["input_image"] for x in prompts]
        inputs = self._processor(
            prompts_lists,
            input_images_lists,
            max_slice_nums=None,
            use_image_id=None,
            return_tensors="pt",
            max_length=8192,
        ).to(self._model.device)
        inputs.pop("image_sizes")

        masked_input_ids = inputs["input_ids"] * inputs["attention_mask"]
        for i in range(masked_input_ids.shape[0]):
            non_zero_values = masked_input_ids[i][masked_input_ids[i] != 0].tolist()
            req_list[i].prompt_tokens = non_zero_values
            req_list[i].extra_kwargs["attention_mask_seq_len"] = len(non_zero_values)
            req_list[i].padding_len = masked_input_ids.shape[1] - len(non_zero_values)

        model_inputs = {
            "input_ids": inputs["input_ids"],
            "image_bound": inputs["image_bound"],
            "pixel_values": inputs["pixel_values"],
            "tgt_sizes": inputs["tgt_sizes"],
        }
        model_inputs["inputs_embeds"], _ = self._model.get_vllm_embedding(model_inputs)

        return {
            "inputs_embeds": model_inputs["inputs_embeds"],
            "attention_mask": inputs["attention_mask"],
        }

    def build_decode_position_ids(
        self, batch_size: int, seq_length: int, reqs: List[InferenceRequest]
    ):
        return None

    def batch_inference(self, req_list: List[InferenceRequest]):
        """
        This method is rewritten
        because the specific inference process is performed by `self._model.llm`,
        not `self._model` itself
        """
        from .utils import batch_inference_one_step

        self.prepare_batch_inference(req_list)
        batch_inference_one_step(
            self, req_list, self.model_uid, self._model.llm, self._tokenizer
        )
        self.handle_batch_inference_results(req_list)
