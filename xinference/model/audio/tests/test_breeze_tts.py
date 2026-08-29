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

import pickle
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from types import SimpleNamespace

import numpy as np
import pytest

from .. import breeze_tts as breeze_tts_module
from .. import load_model_family_from_json
from ..breeze_tts import BreezeTTS2Model, _validate_cfg_scale
from ..core import create_audio_model_instance


@pytest.fixture
def model_spec():
    return SimpleNamespace(
        model_name="Breeze-TTS-2",
        model_family="Breeze-TTS-2",
        model_ability=[
            "text2audio",
            "text2audio_voice_design",
            "text2audio_voice_cloning",
        ],
        engine=None,
    )


class _FakeRuntime:
    sample_rate = 24000

    def __init__(self, chunks=None):
        self._chunks = chunks or [
            SimpleNamespace(
                audio=np.array([0.1, 0.2], dtype=np.float32), sample_rate=24000
            ),
            SimpleNamespace(audio=np.array([0.3], dtype=np.float32), sample_rate=24000),
        ]
        self.request_ids = []

    def iter_audio_chunks(self, inputs, *, request_id):
        self.request_ids.append(request_id)
        yield from self._chunks


def _install_fake_loaded_model(model, captured):
    model._model = object()
    model._tokenizer = object()
    model._audio_tokenizer = object()
    model._runtime = _FakeRuntime()
    model._set_all_seeds = lambda seed: captured.setdefault("seeds", []).append(seed)
    model._get_template = lambda name: name

    def prepare_inputs(
        tokenizer,
        audio_tokenizer,
        runtime_model,
        requests,
        template,
        **kwargs,
    ):
        captured["request"] = dict(requests[0])
        captured["template"] = template
        captured["prepare_kwargs"] = kwargs
        reference_path = requests[0].get("ref_audio_path")
        if reference_path:
            captured["reference_path"] = reference_path
            captured["reference_audio"] = Path(reference_path).read_bytes()
        return {"prepared": True}

    model._prepare_inputs = prepare_inputs


@pytest.mark.parametrize("value", [1, 4.0, "2.5"])
def test_validate_cfg_scale(value):
    assert _validate_cfg_scale(value) == pytest.approx(float(value))


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), "bad"])
def test_validate_cfg_scale_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="cfg_scale must be greater than 0"):
        _validate_cfg_scale(value)


def test_load_builds_official_streaming_runtime(monkeypatch, model_spec):
    import torch

    captured = {}
    loaded_model = object()
    tokenizer = object()
    audio_tokenizer = object()

    def load_runtime(model_path, *, device, attn_implementation):
        captured["load"] = (model_path, device, attn_implementation)
        return tokenizer, loaded_model, audio_tokenizer

    def update_generation_config(model):
        captured["updated_model"] = model

    class FastStreamingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FastRuntime:
        def __init__(self, model, audio_codec, config, *, tokenizer):
            captured["runtime"] = (model, audio_codec, config, tokenizer)
            self.fast_enabled = config.kwargs["fast_all"] is True
            self.codec_chunk_frames = 2

        def warmup_from_profile(self, profile):
            captured["warmup_profile"] = profile
            return {"total_elapsed_ms": 12.5}

    @dataclass(frozen=True)
    class WarmupProfile:
        codec_chunk_frames: int

    def load_warmup_profile(path):
        captured["warmup_path"] = path
        return WarmupProfile(codec_chunk_frames=1)

    components = (
        load_runtime,
        lambda device: device or "cuda:0",
        lambda seed: None,
        update_generation_config,
        lambda name: name,
        lambda *args, **kwargs: {},
        FastRuntime,
        FastStreamingConfig,
        load_warmup_profile,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        breeze_tts_module, "_load_breeze_runtime_components", lambda: components
    )

    model = BreezeTTS2Model(
        "breeze",
        "/models/breeze",
        model_spec,
        device="cuda:2",
        max_new_tokens=900,
        fast_all=True,
    )
    model.load()

    assert captured["load"] == (Path("/models/breeze"), "cuda:2", "eager")
    assert captured["updated_model"] is loaded_model
    assert captured["runtime"][0:2] == (loaded_model, audio_tokenizer)
    assert captured["runtime"][3] is tokenizer
    assert captured["runtime"][2].kwargs["max_new_tokens"] == 900
    assert captured["runtime"][2].kwargs["max_seq_len"] == 2048
    assert captured["runtime"][2].kwargs["fast_all"] is True
    assert captured["warmup_path"].name == "fast.json"
    assert captured["warmup_profile"].codec_chunk_frames == 2


def test_load_rejects_non_cuda_runtime(monkeypatch, model_spec):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = BreezeTTS2Model("breeze", "/models/breeze", model_spec)
    with pytest.raises(RuntimeError, match="requires a CUDA-capable GPU"):
        model.load()


def test_model_recreates_inference_lock_after_serialization(model_spec):
    model = BreezeTTS2Model("breeze", "/models/breeze", model_spec)

    restored = pickle.loads(pickle.dumps(model))

    assert restored._inference_lock is not None
    assert restored._inference_lock.acquire(blocking=False)
    restored._inference_lock.release()


def test_voice_design_non_streaming(monkeypatch, model_spec):
    captured = {}
    model = BreezeTTS2Model("breeze", "/models/breeze", model_spec)
    _install_fake_loaded_model(model, captured)

    def audio_to_bytes(response_format, sample_rate, audio):
        captured["encoded"] = (response_format, sample_rate, audio.numpy())
        return b"encoded-audio"

    monkeypatch.setattr(breeze_tts_module, "_audio_to_bytes", audio_to_bytes)

    result = model.speech(
        input="Hello from Breeze.",
        voice="",
        response_format="wav",
        instruct="A warm and calm voice.",
        cfg_scale=4,
        seed=7,
    )

    assert result == b"encoded-audio"
    assert captured["template"] == "tts_instruction"
    assert captured["request"]["instruction"] == "A warm and calm voice."
    assert captured["prepare_kwargs"]["guidance_scale"] == 4
    assert captured["seeds"] == [7]
    assert captured["encoded"][0:2] == ("wav", 24000)
    np.testing.assert_allclose(
        captured["encoded"][2], np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    )


def test_voice_clone_encodes_and_cleans_reference(monkeypatch, model_spec):
    captured = {}
    model = BreezeTTS2Model("breeze", "/models/breeze", model_spec)
    _install_fake_loaded_model(model, captured)
    monkeypatch.setattr(
        breeze_tts_module,
        "_audio_to_bytes",
        lambda response_format, sample_rate, audio: b"clone",
    )

    result = model.speech(
        input="This is cloned speech.",
        voice="",
        prompt_speech=b"reference-audio",
        prompt_text="Reference transcript.",
        instruction="Speak slowly.",
    )

    assert result == b"clone"
    assert captured["template"] == "ref_edit_tata"
    assert captured["reference_audio"] == b"reference-audio"
    assert captured["request"]["ref_text"] == "Reference transcript."
    assert captured["request"]["instruction"] == "Speak slowly."
    assert not Path(captured["reference_path"]).exists()


def test_voice_clone_requires_reference_transcript(model_spec):
    model = BreezeTTS2Model("breeze", "/models/breeze", model_spec)
    model._runtime = object()

    with pytest.raises(ValueError, match="`prompt_text`"):
        model.speech(
            input="Missing transcript.",
            voice="",
            prompt_speech=b"reference-audio",
        )


def test_streaming_defers_seed_until_iteration(monkeypatch, model_spec):
    captured = {}
    model = BreezeTTS2Model("breeze", "/models/breeze", model_spec)
    _install_fake_loaded_model(model, captured)

    def stream_generator(response_format, sample_rate, chunks):
        captured["stream"] = (response_format, sample_rate, list(chunks))
        yield b"first"
        yield b"second"

    monkeypatch.setattr(breeze_tts_module, "_audio_stream_generator", stream_generator)

    output = model.speech(
        input="Stream this.",
        voice="",
        response_format="pcm",
        stream=True,
        seed=9,
    )
    assert captured.get("seeds") is None
    assert list(output) == [b"first", b"second"]
    assert captured["seeds"] == [9]
    assert captured["stream"][0:2] == ("pcm", 24000)


def test_streaming_serializes_runtime_until_generator_finishes(monkeypatch, model_spec):
    captured = {}
    model = BreezeTTS2Model("breeze", "/models/breeze", model_spec)
    _install_fake_loaded_model(model, captured)

    non_stream_entered = Event()
    worker_started = Event()
    worker_done = Event()

    class MixedRuntime:
        sample_rate = 24000

        def __init__(self):
            self._call_lock = Lock()
            self._call_count = 0

        def iter_audio_chunks(self, inputs, *, request_id):
            with self._call_lock:
                self._call_count += 1
                call_number = self._call_count
            if call_number == 2:
                non_stream_entered.set()
            yield SimpleNamespace(
                audio=np.array([float(call_number)], dtype=np.float32),
                sample_rate=self.sample_rate,
            )
            if call_number == 1:
                yield SimpleNamespace(
                    audio=np.array([1.5], dtype=np.float32),
                    sample_rate=self.sample_rate,
                )

    model._runtime = MixedRuntime()

    def stream_generator(response_format, sample_rate, chunks):
        for index, _chunk in enumerate(chunks):
            yield f"stream-{index}".encode()

    monkeypatch.setattr(breeze_tts_module, "_audio_stream_generator", stream_generator)
    monkeypatch.setattr(
        breeze_tts_module,
        "_audio_to_bytes",
        lambda response_format, sample_rate, audio: b"non-stream",
    )

    stream = model.speech(input="Stream first.", voice="", stream=True)
    assert next(stream) == b"stream-0"

    worker_result = {}

    def generate_non_streaming():
        worker_started.set()
        try:
            worker_result["audio"] = model.speech(
                input="Generate second.", voice="", stream=False
            )
        except Exception as exc:  # pragma: no cover - surfaced below
            worker_result["error"] = exc
        finally:
            worker_done.set()

    worker = Thread(target=generate_non_streaming, daemon=True)
    worker.start()
    assert worker_started.wait(timeout=1)
    try:
        assert not non_stream_entered.wait(timeout=0.2)
        assert not worker_done.is_set()
    finally:
        assert list(stream) == [b"stream-1"]

    assert worker_done.wait(timeout=2)
    worker.join(timeout=1)
    assert "error" not in worker_result
    assert worker_result["audio"] == b"non-stream"
    assert non_stream_entered.is_set()


def test_builtin_catalog_has_huggingface_and_modelscope_sources():
    models = {}
    load_model_family_from_json("model_spec.json", models)
    specs = models["Breeze-TTS-2"]

    assert {spec.model_hub: (spec.model_id, spec.model_revision) for spec in specs} == {
        "huggingface": ("BreezeBlue/Breeze-TTS-2", "main"),
        "modelscope": ("BreezeBlue/Breeze-TTS-2", "master"),
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
    assert all("#system_torchcodec#" in spec.virtualenv.packages for spec in specs)
    assert all("flash-attn==2.8.3" not in spec.virtualenv.packages for spec in specs)
    assert all(spec.virtualenv.no_build_isolation for spec in specs)


def test_flash_attention_dependency_is_added_only_when_requested():
    models = {}
    load_model_family_from_json("model_spec.json", models)

    for spec in models["Breeze-TTS-2"]:
        registered_packages = spec.virtualenv.packages.copy()
        eager_model = BreezeTTS2Model("eager", "/models/breeze", spec)
        flash_model = BreezeTTS2Model(
            "flash",
            "/models/breeze",
            spec,
            attn_implementation="flash_attention_2",
        )

        assert eager_model.model_family.virtualenv.packages == registered_packages
        assert "flash-attn==2.8.3" in flash_model.model_family.virtualenv.packages
        assert spec.virtualenv.packages == registered_packages


def test_create_audio_model_instance_dispatches_breeze(monkeypatch, model_spec):
    from .. import core

    monkeypatch.setattr(core, "match_audio", lambda *args, **kwargs: model_spec)
    model = create_audio_model_instance(
        "breeze", "Breeze-TTS-2", model_path="/models/breeze"
    )
    assert isinstance(model, BreezeTTS2Model)
