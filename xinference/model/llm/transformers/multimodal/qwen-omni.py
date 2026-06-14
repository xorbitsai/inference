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
import base64
import io
import logging
import time
import uuid
from threading import Thread
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch

from .....types import (
    ChatCompletion,
    ChatCompletionAudio,
    ChatCompletionChoice,
    CompletionUsage,
)
from ....utils import is_flash_attn_available, select_device
from ...llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from ..core import PytorchGenerateConfig, register_non_default_model
from .core import PytorchMultiModalModel

logger = logging.getLogger(__name__)


@register_transformer
@register_non_default_model(
    "Qwen2_5OmniModel",
    "Qwen3OmniMoeForConditionalGeneration",
)
class QwenOmniChatModel(PytorchMultiModalModel):
    QWEN_OMNI_ARCHITECTURES = {
        "Qwen2_5OmniModel",
        "Qwen3OmniMoeForConditionalGeneration",
    }
    DEFAULT_SYSTEM_PROMPT = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 2.5 or 3
        if self.model_family.has_architecture("Qwen2_5OmniModel"):
            self._omni_version = "2.5"
        else:
            self._omni_version = "3"

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV2", model_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if model_spec.model_format not in ["pytorch", "gptq", "awq", "bnb", "fp4"]:
            return (
                False,
                "Qwen Omni transformer supports pytorch/gptq/awq/bnb/fp4 formats only",
            )
        if not model_family.has_architecture(*cls.QWEN_OMNI_ARCHITECTURES):
            return (
                False,
                f"Model architectures {model_family.architectures} are not Qwen Omni",
            )
        abilities = model_family.model_ability
        if (
            "omni" not in abilities
            and "vision" not in abilities
            and "audio" not in abilities
        ):
            return (
                False,
                "Qwen Omni transformer requires omni, vision, or audio ability",
            )
        return True

    def decide_device(self):
        device = self._pytorch_model_config.get("device", "auto")
        device = select_device(device)
        self._device = device

    def load_processor(self):
        if self._omni_version == "2.5":
            from transformers import Qwen2_5OmniProcessor as QwenOminiProcessor
        else:
            from transformers import Qwen3OmniMoeProcessor as QwenOminiProcessor

        self._processor = QwenOminiProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._tokenizer = self._processor.tokenizer

    def load_multimodal_model(self):
        if self._omni_version == "2.5":
            from transformers import (
                Qwen2_5OmniForConditionalGeneration as QwenOmniForConditionalGeneration,
            )
        else:
            from transformers import (
                Qwen3OmniMoeForConditionalGeneration as QwenOmniForConditionalGeneration,
            )

        # for multiple GPU, set back to auto to make multiple devices work
        device = "auto" if self._device == "cuda" else self._device
        kwargs = {}
        enable_flash_attn = self._pytorch_model_config.get(
            "enable_flash_attn", is_flash_attn_available()
        )
        if enable_flash_attn:
            kwargs["attn_implementation"] = "flash_attention_2"
        kwargs = self.apply_quantization_config(kwargs)
        logger.debug("Loading model with extra kwargs: %s", kwargs)

        self._model = QwenOmniForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True,
            **kwargs,
        )

    def _transform_messages(
        self,
        messages: List[dict],  # type: ignore
    ):
        messages = super()._transform_messages(messages)
        if messages[0]["role"] != "system":
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.DEFAULT_SYSTEM_PROMPT}],  # type: ignore
                },
            )
        else:
            logger.debug("Force to set system prompt")
            messages[0]["content"] = [{"type": "text", "text": self.DEFAULT_SYSTEM_PROMPT}]  # type: ignore
        return messages

    def build_inputs_from_messages(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ):
        from qwen_omni_utils import process_mm_info

        use_audio_in_video = generate_config.get("use_audio_in_video", True)

        messages = self._transform_messages(messages)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=use_audio_in_video
        )
        logger.debug(
            "Text, audio, image, video: %s, %s, %s, %s", text, audios, images, videos
        )
        inputs = self._processor(
            text=text,
            images=images,
            audio=audios,
            videos=videos,
            padding=True,
            return_tensors="pt",
            use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(self._device)
        return inputs

    def build_generate_kwargs(
        self,
        generate_config: Dict,
    ) -> Dict[str, Any]:
        voice = generate_config.get("voice", "Chelsie")
        return {
            "max_new_tokens": generate_config.get("max_tokens") or 512,
            "temperature": generate_config.get("temperature", 1),
            "speaker": voice,
        }

    def build_streaming_iter(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ) -> Tuple[Iterator, int]:
        from transformers import TextIteratorStreamer

        tokenizer = self._tokenizer
        streamer = TextIteratorStreamer(
            tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )

        config = self.build_generate_kwargs(generate_config)
        inputs = self.build_inputs_from_messages(messages, generate_config)
        gen_kwargs = dict(**inputs, **config, streamer=streamer)
        thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        return streamer, len(inputs.input_ids[0])

    def generate_non_streaming(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> ChatCompletion:
        """
        Special case for qwen2.5-omni, since it has audio output
        """
        import soundfile as sf

        generate_config = generate_config if generate_config else {}  # type: ignore
        config = self.build_generate_kwargs(generate_config)  # type: ignore
        inputs = self.build_inputs_from_messages(messages, generate_config)  # type: ignore
        use_audio_in_video = generate_config.get("use_audio_in_video", True)
        gen_kwargs = dict(**inputs, **config, use_audio_in_video=use_audio_in_video)
        # === Run model.generate() (handle both (ids, audio) and ids-only cases) ===
        result = self._model.generate(**gen_kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            # Qwen2.5-Omni returns (generated_ids, audio)
            generated_ids, audio = result
        else:
            # Qwen3-Omni returns only generated_ids
            generated_ids, audio = result, None
        if hasattr(generated_ids, "sequences"):
            generated_ids = generated_ids.sequences

        # === Handle text decoding ===
        input_len = inputs.input_ids.shape[1]
        # Ensure we have a consistent 2D structure
        # Normalize to list[list[int]]
        if isinstance(generated_ids, torch.Tensor):
            generated_ids = generated_ids.tolist()
        elif isinstance(generated_ids, list) and all(
            isinstance(x, int) for x in generated_ids
        ):
            # Single sequence as flat list of ints
            generated_ids = [generated_ids]
        elif isinstance(generated_ids, list) and all(
            isinstance(x, list) for x in generated_ids
        ):
            pass  # already correct
        else:
            raise TypeError(f"Unexpected generated_ids type: {type(generated_ids)}")

        # Remove prompt tokens
        generated_ids_trimmed = [out_ids[input_len:] for out_ids in generated_ids]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        wav_io = io.BytesIO()
        sf.write(
            wav_io,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
            format="WAV",
        )
        wav_bytes = wav_io.getvalue()
        audio_content = base64.b64encode(wav_bytes).decode()

        return ChatCompletion(
            id="chat" + str(uuid.uuid1()),
            object="chat.completion",
            created=int(time.time()),
            model=self.model_uid,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message={
                        "role": "assistant",
                        "content": output_text,
                        "audio": ChatCompletionAudio(
                            id="audio" + str(uuid.uuid1()),
                            data=audio_content,
                            expires_at=int(time.time()),
                            transcript="",
                        ),
                    },
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=-1, completion_tokens=-1, total_tokens=-1
            ),
        )
