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

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from .. import auk as auk_module
from .. import load_model_family_from_json
from ..auk import AukModel, _estimate_gen_seconds
from ..core import create_audio_model_instance


@pytest.fixture
def model_spec():
    return SimpleNamespace(
        model_name="AuK",
        model_family="AuK",
        model_ability=[
            "text2audio",
            "text2audio_voice_design",
            "text2audio_voice_cloning",
        ],
        engine=None,
    )


def test_load_uses_model_checkpoint_and_local_qwen_snapshot(
    monkeypatch, tmp_path, model_spec
):
    captured = {}

    class FakeAukInfer:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    (tmp_path / "config.yaml").write_text("model: {}")
    (tmp_path / "auk_base.safetensors").touch()
    (tmp_path / "Qwen2.5-Omni-3B").mkdir()
    monkeypatch.setattr(auk_module, "_load_auk_infer", lambda: FakeAukInfer)

    model = AukModel("auk", str(tmp_path), model_spec, device="cuda:1", dtype="fp16")
    model.load()

    assert captured["args"] == (
        str(tmp_path / "config.yaml"),
        str(tmp_path / "auk_base.safetensors"),
    )
    assert captured["kwargs"] == {
        "device": "cuda:1",
        "dtype": "fp16",
        "qwen_path": str(tmp_path / "Qwen2.5-Omni-3B"),
    }


def test_flash_loads_flash_checkpoint(monkeypatch, tmp_path, model_spec):
    captured = {}
    model_spec.model_name = "AuK-Flash"

    class FakeAukInfer:
        def __init__(self, *args, **kwargs):
            captured["args"] = args

    (tmp_path / "config.yaml").write_text("model: {}")
    (tmp_path / "auk_flash.safetensors").touch()
    monkeypatch.setattr(auk_module, "_load_auk_infer", lambda: FakeAukInfer)

    model = AukModel("auk-flash", str(tmp_path), model_spec)
    model.load()

    assert captured["args"][1] == str(tmp_path / "auk_flash.safetensors")


def test_speech_generates_zero_shot_audio_and_removes_temp_prompt(
    monkeypatch, model_spec
):
    captured = {}

    class FakeModel:
        def generate(self, messages, **kwargs):
            prompt_path = messages[0]["content"][1]["audio"]
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            captured["prompt_path"] = prompt_path
            assert Path(prompt_path).read_bytes() == b"reference-audio"
            return "audio", 24000

    model = AukModel("auk", "/models/auk", model_spec)
    model._model = FakeModel()
    monkeypatch.setattr(
        auk_module,
        "_audio_to_bytes",
        lambda response_format, sample_rate, audio: b"encoded-audio",
    )

    result = model.speech(
        input="Hello from AuK.",
        voice="",
        response_format="wav",
        prompt_speech=b"reference-audio",
        instruction="Speak warmly.",
        gen_seconds=2.5,
        nfe=12,
        cfg_strength=1.5,
        seed=7,
    )

    assert result == b"encoded-audio"
    expected_instruction = (
        'Speak warmly.\nSay the following with the same voice: "Hello from AuK."'
    )
    assert captured["messages"][0]["content"][0]["text"] == expected_instruction
    assert captured["kwargs"] == {
        "gen_seconds": 2.5,
        "nfe": 12,
        "cfg_strength": 1.5,
        "sway_sampling_coef": -1.0,
        "t_grid": None,
        "seed": 7,
    }
    assert not os.path.exists(captured["prompt_path"])


def test_speech_without_prompt_uses_instruction_tts(model_spec, monkeypatch):
    captured = {}

    class FakeModel:
        def generate(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "audio", 24000

    model = AukModel("auk", "/models/auk", model_spec)
    model._model = FakeModel()
    monkeypatch.setattr(auk_module, "_audio_to_bytes", lambda *args: b"encoded")

    assert model.speech("你好，AuK。", voice="") == b"encoded"
    assert captured["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": 'Generate natural speech for the following text: "你好，AuK。"',
                }
            ],
        }
    ]
    assert captured["kwargs"]["gen_seconds"] == _estimate_gen_seconds(
        "你好，AuK。", 1.0
    )


@pytest.mark.parametrize("speed", [0, -1, "bad"])
def test_speech_rejects_invalid_speed(model_spec, speed):
    model = AukModel("auk", "/models/auk", model_spec)
    with pytest.raises(ValueError, match="speed must be greater than 0"):
        model.speech("hello", voice="", speed=speed)


def test_speech_rejects_streaming(model_spec):
    model = AukModel("auk", "/models/auk", model_spec)
    with pytest.raises(ValueError, match="does not support streaming"):
        model.speech("hello", voice="", stream=True)


def test_builtin_catalog_has_huggingface_and_modelscope_sources():
    models = {}
    load_model_family_from_json("model_spec.json", models)

    expected = {
        "AuK": ("tencent/AuK", "Tencent-Hunyuan/AuK"),
        "AuK-Flash": ("tencent/AuK-Flash", "Tencent-Hunyuan/AuK-Flash"),
    }
    for model_name, (huggingface_id, modelscope_id) in expected.items():
        specs = models[model_name]
        assert {
            spec.model_hub: (spec.model_id, spec.model_revision) for spec in specs
        } == {
            "huggingface": (huggingface_id, "main"),
            "modelscope": (modelscope_id, "master"),
        }
        assert all(
            set(spec.model_ability)
            == {
                "text2audio",
                "text2audio_voice_design",
                "text2audio_voice_cloning",
            }
            for spec in specs
        )
        assert all(
            "qwen-omni-utils>=0.0.9" in spec.virtualenv.packages for spec in specs
        )
        assert all(spec.virtualenv.no_build_isolation for spec in specs)


@pytest.mark.parametrize("model_name", ["AuK", "AuK-Flash"])
def test_create_audio_model_instance_dispatches_auk(
    monkeypatch, model_spec, model_name
):
    from .. import core

    model_spec.model_name = model_name
    monkeypatch.setattr(core, "match_audio", lambda *args, **kwargs: model_spec)

    model = create_audio_model_instance("auk", model_name, model_path="/models/auk")

    assert isinstance(model, AukModel)
