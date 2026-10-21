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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ....thirdparty.ming_tts import tts_tasks
from ....thirdparty.ming_tts.ming_tiny.task_protocol import resolve_task_request
from .. import ming_tts, utils
from ..core import create_audio_model_instance, match_audio
from ..engine import PyTorchMingTTSAudioModel
from ..ming_tts import MingTTSModel

MODEL_NAMES = ["Ming-3.0-tts-tiny", "Ming-3.0-tts-0.5B"]


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize(
    "hub,revision", [("huggingface", "main"), ("modelscope", "master")]
)
def test_catalog_and_dispatch(name, hub, revision, tmp_path):
    spec = match_audio(name, hub)
    assert spec.model_hub == hub
    assert spec.model_id == f"inclusionAI/{name}"
    assert spec.model_revision == revision
    assert PyTorchMingTTSAudioModel.match(spec)
    model = create_audio_model_instance(
        "ming", name, hub, model_path=str(tmp_path), enable_virtual_env=False
    )
    assert isinstance(model, MingTTSModel)
    assert model._model_spec.model_hub == hub
    config = json.loads(
        (Path(__file__).parents[1] / "models" / f"{name}.json").read_text()
    )[0]
    packages = config["virtualenv"]["packages"]
    assert "transformers==4.46.3" in packages
    assert not any("git+" in package for package in packages)
    if name.endswith("tiny"):
        assert "fla-core==0.4.2" in packages
        assert "x-transformers==2.7.4" in packages
    else:
        assert "diffusers==0.33.0" in packages
        assert "x-transformers==2.9.2" in packages


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
def test_load_selected_local_snapshot(monkeypatch, tmp_path, name, hub):
    calls = {}

    class Runtime:
        def __init__(self, path, **kwargs):
            calls.update(path=path, options=kwargs)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls(path, **kwargs)

    def load_runtime(tiny):
        assert tiny == name.endswith("tiny")
        return Runtime

    monkeypatch.setattr(ming_tts, "_load_ming_runtime", load_runtime)
    device = "cuda:1" if name.endswith("tiny") else "cpu"
    model = MingTTSModel("ming", str(tmp_path), match_audio(name, hub), device=device)
    model.load()
    assert calls["path"] == str(tmp_path)
    assert calls["options"]["device"] == device
    if not name.endswith("tiny"):
        assert calls["options"]["cache_prompts"] is False
        assert calls["options"]["text_normalization"] is False


def test_tiny_requires_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(ming_tts, "_load_ming_runtime", lambda tiny: object())
    model = MingTTSModel(
        "ming", str(tmp_path), match_audio(MODEL_NAMES[0], "huggingface"), device="cpu"
    )
    with pytest.raises(ValueError, match="CUDA"):
        model.load()


@pytest.fixture(params=MODEL_NAMES)
def loaded_model(request, monkeypatch, tmp_path):
    model = MingTTSModel(
        "ming", str(tmp_path), match_audio(request.param, "huggingface")
    )
    calls = {}
    waveform = torch.tensor([[0.1, -0.2]], dtype=torch.float32)

    class Runtime:
        sample_rate = 24000

        def generate(self, **kwargs):
            resolve_task_request(
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key
                    in {
                        "task",
                        "text",
                        "prompt_wav",
                        "prompt_text",
                        "target_language",
                        "instruction",
                        "dialect",
                        "emotion",
                        "style",
                        "speed",
                        "pitch",
                        "volume",
                        "ip",
                    }
                }
            )
            return self.record(kwargs, "prompt_wav", SimpleNamespace(waveform=waveform))

        def synthesize(self, **kwargs):
            tts_tasks.build_task_generation_kwargs(**kwargs)
            return self.record(kwargs, "prompt_wav_path", waveform)

        def record(self, kwargs, path_key, result):
            calls["generate"] = kwargs
            paths = kwargs.get(path_key)
            if paths is not None:
                paths = paths if isinstance(paths, list) else [paths]
                calls["paths"] = paths
                calls["references"] = [Path(path).read_bytes() for path in paths]
            if calls.get("fail"):
                raise RuntimeError("synthesis failed")
            return result

    def encode(response_format, sample_rate, audio):
        calls["encode"] = response_format, sample_rate, audio.tolist()
        return b"encoded"

    monkeypatch.setattr(utils, "audio_to_bytes", encode)
    model._model = Runtime()
    return model, calls


def test_builtin_voice(loaded_model):
    model, calls = loaded_model
    assert model.speech("你好", response_format="wav") == b"encoded"
    generated = calls["generate"]
    assert generated["task"].lower() == "ip"
    assert (generated.get("controls") or generated)["ip"] == "灵小柔"
    assert calls["encode"][:2] == ("wav", 24000)
    assert calls["encode"][2][0] == pytest.approx([0.1, -0.2])
    model.speech("Hello", voice="US_James")
    generated = calls["generate"]
    assert (generated.get("controls") or generated)["ip"] == "US_James"


def test_voice_cloning(loaded_model):
    model, calls = loaded_model
    options = {"cfg_scale" if model._tiny else "cfg": 3.0}
    model.speech(
        "Hello",
        prompt_speech=b"reference",
        prompt_text="Reference text",
        language="en",
        temperature=0.5,
        seed=42,
        **options,
    )
    generated = calls["generate"]
    assert generated["task"] == ("seed" if model._tiny else "TTS")
    assert generated["prompt_text"] == "Reference text"
    assert generated["temperature"] == 0.5
    assert generated.get("target_language", generated.get("language_id")) == "English"
    assert calls["references"] == [b"reference"]
    assert all(not Path(path).exists() for path in calls["paths"])
    if model._tiny:
        assert generated["seed"] == 42
    else:
        assert generated["use_spk_emb"] is True
        assert generated["use_zero_spk_emb"] is False


@pytest.mark.parametrize(
    "task,options",
    [
        ("freeform", {"instruction": "Speak softly"}),
        ("voice_design", {"controls": {"style": "A warm voice"}}),
        ("emotion", {"prompt_speech": b"reference", "emotion": "高兴"}),
        ("dialect", {"prompt_speech": b"reference", "dialect": "川渝话"}),
        (
            "basic",
            {
                "prompt_speech": b"reference",
                "controls": {
                    "speed": "慢速",
                    "pitch": "中",
                    "volume": "中",
                },
            },
        ),
        (
            "multilingual",
            {
                "prompt_speech": b"reference",
                "prompt_text": "参考文本",
                "language": "ja",
            },
        ),
        ("paralinguistic", {"prompt_speech": b"reference", "prompt_text": "参考文本"}),
    ],
)
def test_native_tasks(loaded_model, task, options):
    model, calls = loaded_model
    assert model.speech("你好[Laughter]", task=task, **options) == b"encoded"
    generated = calls["generate"]
    expected = "tts" if not model._tiny and task == "paralinguistic" else task
    assert generated["task"].lower() == expected
    assert all(not Path(path).exists() for path in calls.get("paths", []))


def test_tiny_podcast(loaded_model):
    model, calls = loaded_model
    options = {
        "task": "podcast",
        "prompt_speech": [b"first", b"second"],
        "prompt_text": ["第一位", "第二位"],
    }
    if not model._tiny:
        with pytest.raises(ValueError, match="tiny"):
            model.speech("speaker_1: 你好\nspeaker_2: 世界", **options)
        return
    assert model.speech("speaker_1: 你好\nspeaker_2: 世界", **options) == b"encoded"
    assert calls["references"] == [b"first", b"second"]
    assert all(not Path(path).exists() for path in calls["paths"])


def test_reference_cleanup_on_failure(loaded_model):
    model, calls = loaded_model
    calls["fail"] = True
    with pytest.raises(RuntimeError, match="synthesis failed"):
        model.speech("Hello", prompt_speech=b"reference", prompt_text="Reference")
    assert all(not Path(path).exists() for path in calls["paths"])


@pytest.mark.parametrize(
    "options,match",
    [
        ({"stream": True}, "streaming"),
        ({"task": "tts"}, "prompt_speech and prompt_text"),
        ({"prompt_speech": b"reference"}, "prompt_speech and prompt_text"),
        ({"prompt_speech": b""}, "audio bytes"),
        ({"prompt_speech": "/server/file.wav"}, "audio bytes"),
        ({"controls": {"unknown": 1}}, "controls"),
        ({"task": "invalid"}, "task"),
    ],
)
def test_invalid_requests(loaded_model, options, match):
    model, calls = loaded_model
    with pytest.raises(ValueError, match=match):
        model.speech("Hello", **options)
    assert "generate" not in calls


def test_empty_input(loaded_model):
    model, _ = loaded_model
    with pytest.raises(ValueError, match="non-empty"):
        model.speech("  ")
