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
from io import BytesIO
from threading import Thread
from typing import Any, Dict, Iterator, List, Tuple, Union
from urllib.request import urlopen

import numpy as np

from .....model.utils import select_device
from ...llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from ..core import register_non_default_model
from .core import PytorchMultiModalModel

logger = logging.getLogger(__name__)


@register_transformer
@register_non_default_model("Qwen2AudioForConditionalGeneration")
class Qwen2AudioChatModel(PytorchMultiModalModel):
    QWEN2_AUDIO_ARCHITECTURES = {"Qwen2AudioForConditionalGeneration"}

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV2", model_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if not model_family.has_architecture(*cls.QWEN2_AUDIO_ARCHITECTURES):
            return (
                False,
                f"Model architectures {model_family.architectures} are not Qwen2-Audio",
            )
        if "audio" not in model_family.model_ability:
            return False, "Qwen2-Audio transformer requires audio ability"
        return True

    def decide_device(self):
        device = self._pytorch_model_config.get("device", "auto")
        self._device = select_device(device)

    def load_processor(self):
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            device_map="auto" if self._device == "cuda" else self._device,
            # trust_remote_code=True,
            code_revision=self.model_spec.model_revision,
        )

        self._tokenizer = self._processor.tokenizer

    def load_multimodal_model(self):
        from transformers import Qwen2AudioForConditionalGeneration

        kwargs = self.apply_quantization_config()
        self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map="auto" if self._device == "cuda" else self._device,
            # trust_remote_code=True,
            revision=self.model_spec.model_revision,
            **kwargs,
        )

    def _transform_messages(
        self,
        messages: List[dict],  # type: ignore
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
                            BytesIO(urlopen(item["audio_url"]["url"]).read()),
                            sr=self._processor.feature_extractor.sampling_rate,
                        )[0]
                        audios.append(audio)

        return text, audios

    def build_inputs_from_messages(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ):
        text, audios = self._transform_messages(messages)
        inputs = self._processor(
            text=text, audios=audios, return_tensors="pt", padding=True
        )
        # Make sure that the inputs and the model are on the same device.
        inputs.data = {k: v.to(self._device) for k, v in inputs.data.items()}
        inputs.input_ids = inputs.input_ids.to(self._device)
        return inputs

    def build_generate_kwargs(
        self,
        generate_config: Dict,
    ) -> Dict[str, Any]:
        return dict(max_length=generate_config.get("max_tokens") or 512)

    def build_streaming_iter(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ) -> Tuple[Iterator, int]:
        from transformers import TextIteratorStreamer

        inputs = self.build_inputs_from_messages(messages, generate_config)
        config = self.build_generate_kwargs(generate_config)

        tokenizer = self._processor.tokenizer
        streamer = TextIteratorStreamer(
            tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {"streamer": streamer, **inputs, **config}
        thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        return streamer, len(inputs.input_ids[0])
