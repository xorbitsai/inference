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

import io
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ..minimax_music3_mlx import MLXMiniMaxMusic3Model


def make_model():
    return MLXMiniMaxMusic3Model(
        "uid", "/local/music", SimpleNamespace(model_ability=["text2music"])
    )


@pytest.mark.parametrize(
    "system, processor, expected",
    [
        ("linux", "x86_64", False),
        ("linux", "arm", False),
        ("win32", "AMD64", False),
        ("darwin", "i386", False),
        ("darwin", "arm", True),
    ],
)
def test_music_mlx_catalog_platform_filter(monkeypatch, system, processor, expected):
    import platform

    from .. import load_model_family_from_json

    models = {}
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", system)
        patch.setattr(platform, "processor", lambda: processor)
        load_model_family_from_json("model_spec.json", models)
    mlx_specs = [spec for spec in models["MiniMax-Music3"] if spec.engine == "MLX"]
    assert {spec.model_hub for spec in mlx_specs} == (
        {"huggingface", "modelscope"} if expected else set()
    )


@pytest.mark.parametrize("response_format", ["wav", "flac", "mp3", "ogg"])
def test_music_stereo_mapping_and_thread_affinity(monkeypatch, response_format):
    sf = pytest.importorskip("soundfile", minversion="0.13.1")
    calls = []
    left = np.linspace(-0.25, 0.25, 4410, dtype=np.float32)
    audio = np.column_stack((left, -left))

    class FakeModel:
        def generate(self, *, text, lyrics, duration, steps, seed):
            calls.append(threading.get_ident())
            assert (text, lyrics, duration, steps, seed) == (
                "Warm piano",
                "[verse]\nHello",
                1.0,
                8,
                7,
            )
            yield SimpleNamespace(audio=audio, sample_rate=44100)

    def load(path, **kwargs):
        calls.append(threading.get_ident())
        assert path == "/local/music" and kwargs == {}
        return FakeModel()

    music = ModuleType("mlx_audio.music")
    music.load = load
    monkeypatch.setitem(sys.modules, "mlx_audio.music", music)
    model = make_model()
    model.load()
    output = model.speech(
        "[verse]\nHello",
        instruct="Warm piano",
        duration=1,
        steps=8,
        seed=7,
        response_format=response_format,
    )
    info = sf.info(io.BytesIO(output))
    assert info.channels == 2 and info.samplerate == 44100
    assert info.format == response_format.upper()
    if response_format == "wav":
        decoded, _ = sf.read(io.BytesIO(output), dtype="float32")
        np.testing.assert_array_equal(decoded, audio)
    assert len(calls) == 2 and calls[0] == calls[1] != threading.get_ident()


def test_music_preserves_two_sample_stereo_and_concatenates_results():
    sf = pytest.importorskip("soundfile")
    audio = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    model = make_model()
    model._model = SimpleNamespace(
        generate=lambda **kw: iter(
            [
                SimpleNamespace(audio=audio[:1], sample_rate=44100),
                SimpleNamespace(audio=audio[1:], sample_rate=44100),
            ]
        )
    )
    output = model.speech("[instrumental]", instruct="Piano")
    decoded, rate = sf.read(io.BytesIO(output), dtype="float32")
    assert rate == 44100
    np.testing.assert_array_equal(decoded, audio)


@pytest.mark.parametrize(
    "override",
    [
        {"input": ""},
        {"instruct": " "},
        {"seed": -1},
        {"seed": True},
        {"duration": 361},
        {"duration": float("nan")},
        {"steps": 0},
        {"steps": 31},
        {"steps": True},
        {"steps": 1.5},
        {"stream": True},
        {"voice": "alloy"},
        {"speed": 2},
        {"prompt_speech": b"ref"},
        {"response_format": "pcm"},
    ],
)
def test_music_rejects_invalid_requests_before_generation(override):
    model = make_model()
    model._model = object()
    request = {"input": "[instrumental]", "instruct": "Piano"}
    request.update(override)
    with pytest.raises(ValueError):
        model.speech(**request)


@pytest.mark.parametrize(
    "audio, rate",
    [
        (np.zeros(4), 44100),
        (np.zeros((4, 1)), 44100),
        (np.full((4, 2), np.nan), 44100),
        (np.zeros((4, 2)), 24000),
        (np.zeros((0, 2)), 44100),
    ],
)
def test_music_rejects_invalid_output(audio, rate):
    model = make_model()
    model._model = SimpleNamespace(
        generate=lambda **kw: iter([SimpleNamespace(audio=audio, sample_rate=rate)])
    )
    with pytest.raises(RuntimeError):
        model.speech("[instrumental]", instruct="Piano")


def test_music_rejects_empty_output():
    model = make_model()
    model._model = SimpleNamespace(generate=lambda **kw: iter([]))
    with pytest.raises(RuntimeError, match="no generated audio"):
        model.speech("[instrumental]", instruct="Piano")
