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
from io import BytesIO
from typing import Dict, Iterator, List, Optional, Union
from urllib.request import urlopen

import numpy as np

from ....model.utils import select_device
from ....types import ChatCompletion, ChatCompletionChunk, CompletionChunk
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import generate_chat_completion, generate_completion_chunk
from .core import PytorchChatModel, PytorchGenerateConfig

logger = logging.getLogger(__name__)


class Qwen2AudioChatModel(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._processor = None
        self._model = None
        self._device = None

    @classmethod
    def match(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        llm_family = model_family.model_family or model_family.model_name
        if "qwen2-audio".lower() in llm_family.lower():
            return True
        return False

    def load(self):
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        device = self._pytorch_model_config.get("device", "auto")
        device = select_device(device)
        self._device = device
        # for multiple GPU, set back to auto to make multiple devices work
        device = "auto" if device == "cuda" else device

        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            device_map=device,
            # trust_remote_code=True,
            code_revision=self.model_spec.model_revision,
        )
        self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map=device,
            # trust_remote_code=True,
            revision=self.model_spec.model_revision,
        )

    def _transform_messages(
        self,
        messages: List[Dict],
    ):
        import librosa

        text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        audios: List[np.ndarray] = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, List):
                for item in content:  # type: ignore
                    if item.get("type") == "audio" and "audio_url" in item:
                        audio = librosa.load(
                            BytesIO(urlopen(item["audio_url"]).read()),
                            sr=self._processor.feature_extractor.sampling_rate,
                        )[0]
                        audios.append(audio)

        return text, audios

    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        text, audios = self._transform_messages(messages)
        inputs = self._processor(
            text=text, audios=audios, return_tensors="pt", padding=True
        )
        inputs.input_ids = inputs.input_ids.to(self._device)
        generate_config = generate_config if generate_config else {}
        stream = generate_config.get("stream", False) if generate_config else False

        if stream:
            it = self._generate_stream(inputs, generate_config)
            return self._to_chat_completion_chunks(it)
        else:
            c = self._generate(inputs, generate_config)
            return c

    def _generate(self, inputs, config: PytorchGenerateConfig = {}) -> ChatCompletion:
        generate_ids = self._model.generate(
            **inputs,
            max_length=config.get("max_tokens", 512),
        )
        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
        response = self._processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return generate_chat_completion(self.model_uid, response)

    def _generate_stream(
        self, inputs, config: PytorchGenerateConfig = {}
    ) -> Iterator[CompletionChunk]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        tokenizer = self._processor.tokenizer
        streamer = TextIteratorStreamer(
            tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {
            "max_new_tokens": config.get("max_tokens", 512),
            "streamer": streamer,
            **inputs,
        }

        thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
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
