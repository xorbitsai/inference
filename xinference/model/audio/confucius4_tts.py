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

import importlib
import json
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from unittest.mock import patch

from ...core.exceptions import InvalidAudioInputError

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)

_MODEL_ID = "netease-youdao/Confucius4-TTS"
_LANGUAGES = {
    "zh",
    "en",
    "ja",
    "ko",
    "de",
    "fr",
    "es",
    "id",
    "it",
    "th",
    "pt",
    "ru",
    "ms",
    "vi",
}
_GENERATE_OPTIONS = {
    "raw",
    "temperature",
    "top_p",
    "top_k",
    "num_beams",
    "repetition_penalty",
    "max_length",
    "n_timesteps",
    "inference_cfg_rate",
    "max_text_tokens_per_segment",
    "cross_fade_duration",
    "edge_fade_duration",
    "edge_pad_duration",
    "verbose",
}
_LOAD_LOCK = threading.Lock()


class Confucius4TTSModel:
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
        self._model = None

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def load(self):
        import torch

        model_dir = Path(self._model_path)
        config_path = Path(__file__).with_name("confucius4_tts_config.json")
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)

        paths = config["paths"]
        paths["tokenizer_path"] = str(model_dir)
        paths["w2v_stat"] = str(model_dir / "wav2vec2bert_stats.pt")
        campplus_checkpoint = None
        if getattr(self._model_spec, "model_hub", None) == "modelscope":
            from modelscope.hub.file_download import model_file_download

            def download_auxiliary(repo_id: str, filenames: tuple[str, ...]) -> str:
                for filename in filenames:
                    downloaded = model_file_download(
                        repo_id, filename, revision="master"
                    )
                return str(Path(downloaded).parent)

            paths["w2v_bert_path"] = download_auxiliary(
                "facebook/w2v-bert-2.0",
                ("config.json", "preprocessor_config.json", "model.safetensors"),
            )
            campplus_checkpoint = model_file_download(
                "iic/speech_campplus_sv_zh-cn_16k-common",
                paths["style_encoder"]["checkpoint"],
                revision="master",
            )
            paths["vocoder_path"] = download_auxiliary(
                "nv-community/bigvgan_v2_22khz_80band_256x",
                ("config.json", "bigvgan_generator.pt"),
            )
        for name in ("t2s_checkpoint", "s2a_checkpoint"):
            checkpoint = model_dir / paths[name]
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Confucius4-TTS checkpoint not found: {checkpoint}"
                )
        if not Path(paths["w2v_stat"]).is_file():
            raise FileNotFoundError(
                f"Confucius4-TTS statistics not found: {paths['w2v_stat']}"
            )

        thirdparty_dir = str(Path(__file__).resolve().parents[2] / "thirdparty")
        if thirdparty_dir in sys.path:
            sys.path.remove(thirdparty_dir)
        sys.path.insert(0, thirdparty_dir)
        upstream = importlib.import_module("confuciustts.cli.inference")
        original_download = upstream.hf_hub_download

        def download_checkpoint(repo_id, filename, **kwargs):
            if repo_id == _MODEL_ID:
                return str(model_dir / filename)
            if repo_id == "funasr/campplus" and campplus_checkpoint:
                return campplus_checkpoint
            return original_download(repo_id, filename, **kwargs)

        fd, temporary_config = tempfile.mkstemp(suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as config_file:
                # JSON is valid YAML and can be read by the upstream loader.
                json.dump(config, config_file)
            # The upstream constructor always calls hf_hub_download for its main
            # weights. Resolve those calls to Xinference's selected hub snapshot.
            with (
                _LOAD_LOCK,
                patch.object(upstream, "hf_hub_download", download_checkpoint),
            ):
                self._model = upstream.ConfuciusTTS(
                    config_path=temporary_config,
                    device=self._device
                    or ("cuda" if torch.cuda.is_available() else "cpu"),
                )
        finally:
            os.unlink(temporary_config)

    def speech(
        self,
        input: str,
        voice: Optional[str] = None,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ) -> bytes:
        from .utils import apply_audio_seed, audio_to_bytes

        if stream:
            raise ValueError("Confucius4-TTS does not support streaming generation.")
        if not isinstance(input, str) or not input.strip():
            raise ValueError("input must be a non-empty string")
        if self._model is None:
            raise RuntimeError("Confucius4-TTS model is not loaded")
        prompt_speech = kwargs.pop("prompt_speech", None)
        if not isinstance(prompt_speech, (bytes, bytearray)) or not prompt_speech:
            raise InvalidAudioInputError(
                "Confucius4-TTS requires reference audio bytes in prompt_speech"
            )
        language = str(kwargs.pop("language", kwargs.pop("lang", "zh"))).lower()
        if language not in _LANGUAGES:
            raise ValueError(f"Unsupported Confucius4-TTS language: {language}")
        kwargs.pop("prompt_text", None)  # The reference transcript is unnecessary.
        apply_audio_seed(kwargs)
        if speed != 1.0:
            logger.warning("Confucius4-TTS does not support speed; ignoring it.")
        if voice:
            logger.warning(
                "Confucius4-TTS does not use named voices; ignoring voice=%r.", voice
            )
        generate_options = {
            key: kwargs.pop(key) for key in _GENERATE_OPTIONS if key in kwargs
        }
        if kwargs:
            logger.warning(
                "Ignoring unsupported Confucius4-TTS speech kwargs: %s", kwargs
            )

        prompt_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as prompt_file:
                prompt_path = prompt_file.name
                prompt_file.write(prompt_speech)
            audio = self._model.generate(
                text=input, lang=language, prompt_wav=prompt_path, **generate_options
            )
            return audio_to_bytes(
                response_format, int(self._model.sample_rate), audio.detach().cpu()
            )
        finally:
            if prompt_path is not None:
                os.unlink(prompt_path)
