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
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ...core.exceptions import InvalidAudioInputError

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)

_CONTROLS = {"dialect", "emotion", "style", "speed", "pitch", "volume", "ip"}
_COMMON_OPTIONS = {"sigma", "temperature", "max_decode_steps"}
_TINY_OPTIONS = _COMMON_OPTIONS | {
    "cfg_scale",
    "flow_steps",
    "min_decode_steps",
    "stop_threshold",
    "min_generation_frames",
    "max_generation_frames",
    "log_every_n_steps",
}
_BASE_OPTIONS = _COMMON_OPTIONS | {
    "cfg",
    "max_length",
    "trim_silence",
    "raw_text",
    "instruction_schema",
}


def _load_ming_runtime(tiny: bool):
    if tiny:
        from ...thirdparty.ming_tts.ming_tiny import MingTinyTTS

        return MingTinyTTS
    from ...thirdparty.ming_tts.ming_tts import MingTTS

    return MingTTS


class MingTTSModel:
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
        self._tiny = model_spec.model_name == "Ming-3.0-tts-tiny"
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

    def load(self):
        import torch

        if not Path(self._model_path).is_dir():
            raise FileNotFoundError(
                f"Ming TTS model directory not found: {self._model_path}"
            )
        runtime = _load_ming_runtime(self._tiny)
        device = self._device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if self._tiny:
            if torch.device(device).type != "cuda":
                raise ValueError("Ming-3.0-tts-tiny requires a CUDA device (fla-core)")
            self._model = runtime.from_pretrained(self._model_path, device=device)
        else:
            # FST normalization requires optional pynini and large grammar assets.
            # Request-scoped files must not leave cached prompt tensors behind.
            self._model = runtime(
                self._model_path,
                device=device,
                text_normalization=False,
                cache_prompts=False,
            )

    def speech(
        self,
        input: str,
        voice: Optional[str] = None,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ) -> bytes:
        from ...thirdparty.ming_tts.ming_tiny.prompt_contract import (
            normalize_target_language,
        )
        from ...thirdparty.ming_tts.ming_tiny.task_protocol import normalize_task
        from .utils import apply_audio_seed, audio_to_bytes

        if stream:
            raise ValueError("Ming-3.0-tts does not support streaming generation.")
        if not isinstance(input, str) or not input.strip():
            raise ValueError("input must be a non-empty string")
        if self._model is None:
            raise RuntimeError("Ming TTS model is not loaded")
        if speed != 1.0:
            logger.warning(
                "Ming TTS ignores numeric speed; use task='basic' with "
                "controls={'speed': ..., 'pitch': ..., 'volume': ...}."
            )

        options = self._kwargs.copy()
        options.update(kwargs)
        reference = options.pop("prompt_speech", None)
        prompt_text = options.pop("prompt_text", None)
        instruction = options.pop("instruction", options.pop("instruct", None))
        default_task = "seed" if reference is not None else "ip"
        if instruction is not None:
            default_task = "freeform"
        task = normalize_task(options.pop("task", default_task))
        if task == "podcast" and not self._tiny:
            raise ValueError("Podcast generation requires Ming-3.0-tts-tiny")
        language = options.pop(
            "language", options.pop("target_language", options.pop("language_id", None))
        )
        if language is None and task in {"seed", "paralinguistic"}:
            language = "Chinese"
        language = normalize_target_language(language)
        controls = options.pop("controls", {})
        if not isinstance(controls, dict) or set(controls) - _CONTROLS:
            raise ValueError(f"controls must contain only {sorted(_CONTROLS)}")
        controls = controls.copy()
        for key in _CONTROLS - {"speed"}:
            if key in options:
                controls[key] = options.pop(key)
        if task == "ip":
            controls.setdefault("ip", voice or "灵小柔")

        allowed_options = _TINY_OPTIONS if self._tiny else _BASE_OPTIONS
        generation = {
            key: options.pop(key) for key in allowed_options if key in options
        }
        seed_options = {"seed": options.pop("seed", None)}
        if options:
            logger.warning("Ignoring unsupported Ming TTS speech kwargs: %s", options)

        with tempfile.TemporaryDirectory(prefix="ming_tts_") as directory:
            prompt_wav = None
            if reference is not None:
                references = (
                    reference if isinstance(reference, (list, tuple)) else [reference]
                )
                if not references or any(
                    not isinstance(item, (bytes, bytearray)) or not item
                    for item in references
                ):
                    raise InvalidAudioInputError(
                        "prompt_speech must contain non-empty audio bytes"
                    )
                paths = []
                for index, audio in enumerate(references):
                    path = Path(directory) / f"reference_{index}.wav"
                    path.write_bytes(audio)
                    paths.append(str(path))
                prompt_wav = paths if isinstance(reference, (list, tuple)) else paths[0]
            if task in {"seed", "multilingual", "paralinguistic"} and (
                not prompt_wav
                or not isinstance(prompt_text, str)
                or not prompt_text.strip()
            ):
                raise InvalidAudioInputError(
                    f"Ming TTS task {task!r} requires prompt_speech and prompt_text"
                )
            with self._inference_lock:
                seed = apply_audio_seed(seed_options)
                if self._tiny:
                    result = self._model.generate(
                        task=task,
                        text=input,
                        prompt_wav=prompt_wav,
                        prompt_text=prompt_text,
                        target_language=language,
                        instruction=instruction,
                        seed=seed,
                        **controls,
                        **generation,
                    )
                    waveform = result.waveform
                else:
                    upstream_task = {
                        "seed": "TTS",
                        "paralinguistic": "TTS",
                    }.get(task, task.upper())
                    waveform = self._model.synthesize(
                        task=upstream_task,
                        text=input,
                        prompt_wav_path=prompt_wav,
                        prompt_text=prompt_text,
                        language_id=language,
                        instruction=instruction,
                        controls=controls,
                        use_spk_emb=prompt_wav is not None,
                        use_zero_spk_emb=prompt_wav is None,
                        **generation,
                    )
                return audio_to_bytes(
                    response_format,
                    int(self._model.sample_rate),
                    waveform.detach().cpu(),
                )
