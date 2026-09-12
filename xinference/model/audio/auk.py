# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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
import os
import sys
import tempfile
import threading
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)

_QWEN_ENCODER_ID = "Qwen/Qwen2.5-Omni-3B"
_BASE_CHECKPOINT = "auk_base.safetensors"
_FLASH_CHECKPOINT = "auk_flash.safetensors"


def _load_auk_infer():
    thirdparty_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../thirdparty")
    )
    if thirdparty_dir not in sys.path:
        sys.path.insert(0, thirdparty_dir)

    from auk.infer.infer_auk import AukInfer

    return AukInfer


def _audio_to_bytes(response_format: str, sample_rate: int, audio) -> bytes:
    from .utils import audio_to_bytes

    return audio_to_bytes(response_format, sample_rate, audio)


def _estimate_gen_seconds(text: str, speed: float) -> float:
    # UTF-8 byte length gives Chinese and Latin text a comparable first-pass duration.
    return min(30.0, max(1.0, len(text.encode("utf-8")) / 18.0 / speed))


class AukModel:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "AudioModelFamilyV2",
        device: Optional[str] = None,
        **kwargs,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._device = device
        self._qwen_path = kwargs.pop("qwen_path", None)
        self._dtype = kwargs.pop("dtype", "bf16")
        self._kwargs = kwargs
        self._model = None
        self._inference_lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_inference_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._inference_lock = threading.Lock()

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def _resolve_qwen_path(self) -> str:
        if self._qwen_path:
            return str(self._qwen_path)

        bundled_path = os.path.join(self._model_path, "Qwen2.5-Omni-3B")
        if os.path.isdir(bundled_path):
            return bundled_path

        logger.info(
            "AuK Qwen2.5-Omni-3B encoder is not bundled with %s; "
            "downloading it from Hugging Face. Set qwen_path for a local snapshot.",
            self._model_spec.model_name,
        )
        return _QWEN_ENCODER_ID

    def load(self):
        config_path = os.path.join(self._model_path, "config.yaml")
        checkpoint = (
            _FLASH_CHECKPOINT
            if self._model_spec.model_name == "AuK-Flash"
            else _BASE_CHECKPOINT
        )
        checkpoint_path = os.path.join(self._model_path, checkpoint)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"AuK config not found: {config_path}")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"AuK checkpoint not found: {checkpoint_path}")

        AukInfer = _load_auk_infer()
        logger.info("Loading %s model...", self._model_spec.model_name)
        self._model = AukInfer(
            config_path,
            checkpoint_path,
            device=self._device,
            dtype=self._dtype,
            qwen_path=self._resolve_qwen_path(),
        )

    @staticmethod
    def _pop_instruction(kwargs: dict[str, Any]) -> Optional[str]:
        instruction = None
        for name in ("instruction", "instruct", "instruct_text"):
            value = kwargs.pop(name, None)
            if instruction is None and value is not None:
                instruction = str(value).strip()
        return instruction or None

    @staticmethod
    def _build_instruction(
        text: str, instruction: Optional[str], has_reference_audio: bool
    ) -> str:
        if instruction is None:
            if has_reference_audio:
                return f'Say the following with the same voice: "{text}"'
            return f'Generate natural speech for the following text: "{text}"'
        if has_reference_audio:
            return f'{instruction}\nSay the following with the same voice: "{text}"'
        return f'{instruction}\nGenerate the following speech: "{text}"'

    def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        if stream:
            raise ValueError("AuK does not support streaming generation.")
        if not input or not input.strip():
            raise ValueError("input must be a non-empty string")
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            raise ValueError(f"speed must be greater than 0, got {speed!r}") from None
        if speed <= 0:
            raise ValueError(f"speed must be greater than 0, got {speed!r}")
        if voice:
            logger.warning(
                "AuK does not use preset OpenAI voices; ignoring voice=%r.", voice
            )

        options = self._kwargs.copy()
        options.update(kwargs)
        prompt_speech = options.pop("prompt_speech", None)
        # AuK conditions on reference audio directly, so its transcript is optional.
        options.pop("prompt_text", None)
        instruction = self._pop_instruction(options)
        gen_seconds = options.pop("gen_seconds", None)
        nfe = int(options.pop("nfe", 32))
        cfg_strength = float(options.pop("cfg_strength", 2.0))
        sway_sampling_coef = options.pop("sway_sampling_coef", -1.0)
        t_grid = options.pop("t_grid", None)
        seed = options.pop("seed", None)
        if options:
            logger.warning("Ignoring unsupported AuK speech kwargs: %s", options)
        if gen_seconds is None:
            gen_seconds = _estimate_gen_seconds(input, speed)
        else:
            gen_seconds = float(gen_seconds)
            if gen_seconds <= 0:
                raise ValueError("gen_seconds must be greater than 0")
        if nfe <= 0:
            raise ValueError("nfe must be greater than 0")
        if seed is not None:
            seed = int(seed)

        assert self._model is not None
        prompt_path = None
        try:
            content = [
                {
                    "type": "text",
                    "text": self._build_instruction(
                        input, instruction, bool(prompt_speech)
                    ),
                }
            ]
            if prompt_speech:
                with tempfile.NamedTemporaryFile(
                    prefix="auk_ref_", suffix=".wav", delete=False
                ) as prompt_file:
                    prompt_path = prompt_file.name
                    prompt_file.write(prompt_speech)
                content.append({"type": "audio", "audio": prompt_path})

            messages = [{"role": "user", "content": content}]
            with self._inference_lock:
                audio, sample_rate = self._model.generate(
                    messages,
                    gen_seconds=gen_seconds,
                    nfe=nfe,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    t_grid=t_grid,
                    seed=seed,
                )
            return _audio_to_bytes(response_format, int(sample_rate), audio)
        finally:
            if prompt_path is not None:
                try:
                    os.unlink(prompt_path)
                except OSError:
                    pass
