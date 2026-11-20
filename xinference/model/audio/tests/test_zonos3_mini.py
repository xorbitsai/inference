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


# CI: linux

import io
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from .... import constants
from ....core.exceptions import InvalidAudioInputError, InvalidModelRequestError
from ...cache_manager import CacheManager
from .. import engine
from ..core import create_audio_model_instance, match_audio
from ..engine_family import AUDIO_ENGINES
from ..zonos3_mini import Zonos3MiniModel

MODEL = "ZONOS3-Mini"


@pytest.mark.parametrize(
    "hub, revision", [("huggingface", "main"), ("modelscope", "master")]
)
@pytest.mark.parametrize(
    "backend, model_id, model_class",
    [
        (
            "transformers",
            "Zyphra/ZONOS3-Mini-hf",
            engine.TransformersZonos3MiniAudioModel,
        ),
        ("SGLang-Omni", "Zyphra/ZONOS3-Mini", engine.SGLangOmniZonos3MiniAudioModel),
        ("GGML", "Zyphra/ZONOS3-Mini", engine.GGMLZonos3MiniAudioModel),
    ],
)
def test_catalog_and_dispatch(
    monkeypatch, tmp_path, hub, revision, backend, model_id, model_class
):
    monkeypatch.setattr(engine, "has_cuda_device", lambda: True)
    monkeypatch.setattr(engine.platform, "system", lambda: "Linux")
    spec = match_audio(MODEL, hub, model_engine=backend)
    assert (spec.model_hub, spec.model_revision, spec.model_id) == (
        hub,
        revision,
        model_id,
    )
    assert len(spec.model_lang) == 43
    assert CacheManager(spec).get_cache_dir().endswith("-modelscope") == (
        hub == "modelscope"
    )
    monkeypatch.setattr(CacheManager, "cache", lambda _: str(tmp_path))
    model = create_audio_model_instance(
        "zonos3", MODEL, download_hub=hub, model_engine=backend.lower()
    )
    assert isinstance(model, model_class)
    assert model.model_family.model_engine == backend


def test_default_engine():
    assert next(iter(AUDIO_ENGINES[MODEL])) == "transformers"


def test_transformers_explicit_model_path_requires_operator_trust(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(constants, "XINFERENCE_TRUST_REMOTE_CODE", False)
    with pytest.raises(ValueError, match="XINFERENCE_TRUST_REMOTE_CODE=1"):
        create_audio_model_instance("zonos3", MODEL, model_path=str(tmp_path))
    assert match_audio(MODEL, "huggingface", model_engine="transformers").is_builtin


def test_sglang_fails_before_download_without_cuda(monkeypatch):
    monkeypatch.setattr(engine, "has_cuda_device", lambda: False)
    cache = Mock()
    monkeypatch.setattr(CacheManager, "cache", cache)
    with pytest.raises(ValueError, match="requires Linux with NVIDIA CUDA"):
        create_audio_model_instance("zonos3", MODEL, model_engine="SGLang-Omni")
    cache.assert_not_called()


def test_transformers_speech_and_clone(tmp_path):
    sf = pytest.importorskip("soundfile")
    import torch

    spec = match_audio(MODEL, "huggingface", model_engine="transformers")
    model = engine.TransformersZonos3MiniAudioModel("zonos3", str(tmp_path), spec)
    generate = Mock(
        return_value=SimpleNamespace(
            waveform=torch.ones(441, dtype=torch.float32) / 2, sampling_rate=44100
        )
    )
    model._model = SimpleNamespace(generate=generate)
    wav = model.speech(
        "你好", "meilin", "wav", language="zh-CN", top_k=12, max_new_tokens=150
    )
    generate.assert_called_once_with(
        "你好", voice="Meilin", language="cmn", topk=12, max_new_frames=150
    )
    info = sf.info(io.BytesIO(wav))
    assert (info.frames, info.channels, info.samplerate) == (441, 1, 44100)

    model.speech("Hello", None, "wav", prompt_speech=wav, seed=17)
    options = generate.call_args.kwargs
    assert options["speaker_audio"].shape == (441,)
    assert options["speaker_sample_rate"] == 44100
    assert options["seed"] == 17
    assert "voice" not in options


def test_stream_close_releases_reference_and_engine_generator(tmp_path):
    sf = pytest.importorskip("soundfile")
    prompt = io.BytesIO()
    sf.write(prompt, np.zeros((441, 2)), 44100, format="WAV")

    class StreamingModel(Zonos3MiniModel):
        supports_streaming = True
        reference = None
        closed = False

        def _generate_stream(self, text, voice, reference_path, language, options):
            self.reference = reference_path
            assert sf.info(reference_path).channels == 1
            assert voice == "Savannah"
            try:
                yield np.ones(3, dtype=np.float32)
                yield np.zeros(5, dtype=np.float32)
            finally:
                self.closed = True

    spec = match_audio(MODEL, "huggingface", model_engine="GGML")
    model = StreamingModel("zonos3", str(tmp_path), spec)
    stream = model.speech(
        "Hello", "alloy", "pcm", stream=True, prompt_speech=prompt.getvalue()
    )
    assert next(stream) == b"\xff\x7f" * 3
    assert model.reference.is_file()
    stream.close()
    assert model.closed
    assert not model.reference.exists()


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"stream": True}, InvalidModelRequestError),
        ({"prompt_text": "unused"}, InvalidModelRequestError),
        ({"top_p": float("nan")}, InvalidModelRequestError),
        ({"temperature": 10**400}, InvalidModelRequestError),
        ({"prompt_speech": b"invalid"}, InvalidAudioInputError),
    ],
)
def test_validate_speech_before_generation(monkeypatch, tmp_path, kwargs, error):
    pytest.importorskip("soundfile")
    spec = match_audio(MODEL, "huggingface", model_engine="transformers")
    model = Zonos3MiniModel("zonos3", str(tmp_path), spec)
    generate = Mock()
    monkeypatch.setattr(model, "_generate", generate)
    with pytest.raises(error):
        model.speech("Hello", None, **kwargs)
    generate.assert_not_called()
