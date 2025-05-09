# Copyright 2022-2025 XProbe Inc.
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
import importlib.util
import io
import logging
import sys
import time
import uuid
from typing import Dict, Iterator, List, Optional, Union

from ....model.utils import select_device
from ....types import (
    ChatCompletion,
    ChatCompletionAudio,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionMessage,
    CompletionChunk,
    CompletionUsage,
)
from ..llm_family import LLMFamilyV1, LLMSpecV1, register_transformer
from ..utils import generate_completion_chunk
from .core import PytorchChatModel, PytorchGenerateConfig, register_non_default_model
from .utils import cache_clean

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


@register_transformer
@register_non_default_model("qwen2.5-omni")
class Qwen2_5OmniChatModel(PytorchChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._tokenizer = None
        self._model = None
        self._device = None
        self._processor = None

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if model_spec.model_format not in ["pytorch", "gptq", "awq"]:
            return False
        llm_family = model_family.model_family or model_family.model_name
        if "qwen2.5-omni".lower() in llm_family.lower():
            return True
        return False

    def load(self):
        logger.debug(
            "Try to load model, current python: %s, sys path: %s",
            sys.executable,
            sys.path,
        )

        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )

        device = self._pytorch_model_config.get("device", "auto")
        device = select_device(device)
        self._device = device
        # for multiple GPU, set back to auto to make multiple devices work
        device = "auto" if device == "cuda" else device
        flash_attn_installed = importlib.util.find_spec("flash_attn") is not None
        kwargs = (
            {}
            if not flash_attn_installed
            else {"attn_implementation": "flash_attention_2"}
        )
        kwargs = self.apply_bnb_quantization(kwargs)
        logger.debug("Loading model with extra kwargs: %s", kwargs)

        self._processor = Qwen2_5OmniProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._tokenizer = self._processor.tokenizer
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True,
            **kwargs,
        )

    @cache_clean
    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        messages = self._transform_messages(messages)

        generate_config = generate_config if generate_config else {}

        stream = generate_config.get("stream", False) if generate_config else False

        if stream:
            it = self._generate_stream(messages, generate_config)
            return self._to_chat_completion_chunks(it)
        else:
            c = self._generate(messages, generate_config)
            return c

    def _transform_messages(
        self,
        messages: Union[List[ChatCompletionMessage], List[dict]],
    ):
        messages = super()._transform_messages(messages)
        if messages[0]["role"] != "system":
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}],  # type: ignore
                },
            )
        else:
            logger.debug("Force to set system prompt")
            messages[0]["content"] = [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}]  # type: ignore
        return messages

    def _generate(
        self, messages: List, config: PytorchGenerateConfig = {}
    ) -> ChatCompletion:
        import soundfile as sf
        from qwen_omni_utils import process_mm_info

        use_audio_in_video = config.get("use_audio_in_video", True)
        voice = config.get("voice", "Chelsie")

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

        # Inference: Generation of the output
        generated_ids, audio = self._model.generate(
            **inputs,
            speaker=voice,
            max_new_tokens=config.get("max_tokens", 512),
            temperature=config.get("temperature", 1),
            use_audio_in_video=use_audio_in_video,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
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

    def _generate_stream(
        self, messages: List, config: PytorchGenerateConfig = {}
    ) -> Iterator[CompletionChunk]:
        from threading import Thread

        from qwen_omni_utils import process_mm_info
        from transformers import TextIteratorStreamer

        use_audio_in_video = config.get("use_audio_in_video", True)
        voice = config.get("voice", "Chelsie")

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

        tokenizer = self._tokenizer
        streamer = TextIteratorStreamer(
            tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )

        # TODO(xuye): Cannot find a way to streaming output,
        # will implement it when it's supported

        gen_kwargs = {
            "max_new_tokens": config.get("max_tokens", 512),
            "temperature": config.get("temperature", 1),
            "streamer": streamer,
            "speaker": voice,
            **inputs,
        }
        error = None

        def model_generate():
            try:
                return self._model.generate(**gen_kwargs)
            except Exception:
                nonlocal error
                error = sys.exc_info()
                streamer.end()
                raise

        thread = Thread(target=model_generate)
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

        if error:
            _, err, tb = error  # type: ignore
            raise err.with_traceback(tb)

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
