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

import ast
import importlib.util
import io
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile
import torch
from xoscar.serialization import deserialize, serialize

from ....core.exceptions import InvalidModelRequestError
from ....core.utils import filter_virtualenv_packages_by_markers
from ...utils import (
    get_engine_params_by_name,
    get_engine_params_by_name_with_virtual_env,
)
from .. import dots_tts as dots_tts_module
from .. import utils
from ..core import create_audio_model_instance, match_audio
from ..dots_tts import DotsTTSEditModel
from ..engine import PyTorchDotsTTSEditAudioModel
from ..engine_family import (
    AUDIO_ENGINES,
    generate_engine_config_by_model_name,
    get_supported_engines_for_model,
)

MODEL = "dots.tts.edit-v1.1"


@pytest.mark.parametrize("engine", ["PyTorch", "pytorch"])
def test_soundfile_install_requirement_survives_engine_filtering(engine):
    spec = match_audio(MODEL, "modelscope")
    packages = filter_virtualenv_packages_by_markers(
        spec.virtualenv.packages, model_engine=engine, cuda_version=None
    )
    assert "soundfile>=0.13.1" in packages


@pytest.mark.parametrize("enable_virtual_env", [False, True])
def test_engine_discovery_handles_missing_host_soundfile(
    monkeypatch, enable_virtual_env
):
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "soundfile" else find_spec(name),
    )
    query = (
        get_engine_params_by_name_with_virtual_env
        if enable_virtual_env
        else get_engine_params_by_name
    )
    result = query("audio", MODEL, enable_virtual_env=enable_virtual_env)
    if enable_virtual_env:
        assert isinstance(result["PyTorch"], list)
        assert result["PyTorch"][0]["virtualenv_required"] is True
        assert "soundfile" in result["PyTorch"][0]["virtualenv_reason"]
    else:
        assert result["PyTorch"] == "Library 'soundfile' is not installed"


class FakeRuntime:
    sample_rate = 48000

    def __init__(self):
        self.requests = []
        self.paths = []
        self.closed = False
        self.fail = False

    def _record(self, task, options):
        self.requests.append((task, options))
        path = options.get("prompt_audio_path") or options.get("source_audio_path")
        if path:
            self.paths.append(Path(path))
            assert Path(path).read_bytes() == b"reference"
        if self.fail:
            raise ValueError("Invalid tagged edit instruction")

    def generate(self, **options):
        self._record("tts", options)
        return {"audio": torch.rand(1, 2400) * 0.2, "sample_rate": self.sample_rate}

    def generate_edit(self, **options):
        result = self.generate(**options)
        self.requests[-1] = ("edit", options)
        return result

    def generate_stream(self, **options):
        try:
            self._record("stream", options)
            for _ in range(3):
                yield torch.rand(1, 800) * 0.2
        finally:
            self.closed = True


@pytest.fixture
def model(tmp_path):
    spec = SimpleNamespace(model_ability=["text2audio"], default_model_config=None)
    model = DotsTTSEditModel("dots", str(tmp_path), spec, device="cpu")
    model._runtime = FakeRuntime()
    model._generation_lock = threading.Lock()
    model._stream_executor = ThreadPoolExecutor(max_workers=1)
    yield model
    model.stop()


@pytest.mark.parametrize(
    "hub,revision", [("huggingface", "main"), ("modelscope", "master")]
)
def test_catalog_and_dispatch(monkeypatch, tmp_path, hub, revision):
    monkeypatch.setitem(AUDIO_ENGINES, MODEL, {})
    spec = match_audio(MODEL, hub)
    generate_engine_config_by_model_name(spec)
    assert set(AUDIO_ENGINES[MODEL]) == {"PyTorch"}
    assert spec.model_id == "dots-studio/dots.tts.edit"
    assert spec.model_revision == revision
    assert spec.model_lang == ["en", "zh"]
    assert "text2audio_voice_cloning" in spec.model_ability
    assert "text2audio_emotion_control" in spec.model_ability
    assert not any(
        "dots.tts" in package or "http" in package
        for package in spec.virtualenv.packages
    )
    assert {
        "loguru",
        "langcodes[data]",
        "einops",
        "librosa>=0.11.0",
        "pydantic>=2.12.5,<3",
        "safetensors>=0.8.0rc0",
        "torchdiffeq",
        "lingua-language-detector",
        "WeTextProcessing",
    }.issubset(spec.virtualenv.packages)
    assert not any("gradio" in package for package in spec.virtualenv.packages)
    assert PyTorchDotsTTSEditAudioModel.match(spec)
    assert set(get_supported_engines_for_model([spec])) == {"PyTorch"}
    instance = create_audio_model_instance(
        "dots",
        MODEL,
        hub,
        model_path=str(tmp_path),
        model_engine="pytorch",
        enable_virtual_env=False,
    )
    assert isinstance(instance, PyTorchDotsTTSEditAudioModel)
    assert instance.model_family.model_engine == "PyTorch"
    assert isinstance(
        create_audio_model_instance(
            "dots",
            MODEL,
            hub,
            model_path=str(tmp_path),
            enable_virtual_env=False,
        ),
        PyTorchDotsTTSEditAudioModel,
    )
    with pytest.raises(ValueError, match="cannot be run on engine"):
        create_audio_model_instance(
            "dots",
            MODEL,
            hub,
            model_path=str(tmp_path),
            model_engine="SGLang-Omni",
            enable_virtual_env=False,
        )


@pytest.fixture
def runtime_module(monkeypatch):
    calls = []
    runtime = SimpleNamespace(
        from_pretrained=lambda path, **options: calls.append((path, options))
        or FakeRuntime()
    )
    monkeypatch.setattr(dots_tts_module, "_load_dots_tts_runtime", lambda: runtime)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "version_info", (3, 12, 0))
    return calls


def test_vendored_runtime_has_license_and_closed_relative_imports():
    root = Path(dots_tts_module.__file__).resolve().parents[2] / "thirdparty/dots_tts"
    assert "Apache License" in (root / "LICENSE").read_text()
    assert (
        "4cf7d6b13f42066f97784e7d7cf2a9ed5f956997" in (root / "README.md").read_text()
    )
    assert (root / "edit_runtime.py").is_file()
    assert not (root / "cli.py").exists()
    prefix = "xinference.thirdparty.dots_tts"
    for path in root.rglob("*.py"):
        parts = path.relative_to(root).with_suffix("").parts
        module = prefix + (
            "." + ".".join(parts[:-1])
            if path.name == "__init__.py" and len(parts) > 1
            else ""
        )
        if path.name != "__init__.py":
            module = prefix + "." + ".".join(parts)
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("dots_tts") for alias in node.names
                ), path
            if not isinstance(node, ast.ImportFrom):
                continue
            assert not (node.module or "").startswith("dots_tts"), path
            if node.level:
                target = importlib.util.resolve_name(
                    "." * node.level + (node.module or ""), package
                )
                assert target == prefix or target.startswith(prefix + "."), path
                relative = target.removeprefix(prefix).lstrip(".").replace(".", "/")
                imported = root / relative
                assert (
                    imported.with_suffix(".py").is_file()
                    or (imported / "__init__.py").is_file()
                ), (path, target)


def test_unloaded_model_can_cross_actor_boundary(tmp_path):
    model = create_audio_model_instance(
        "dots",
        MODEL,
        "modelscope",
        model_path=str(tmp_path),
        model_engine="PyTorch",
        enable_virtual_env=False,
    )
    header, buffers = serialize(model)
    restored = deserialize(header, buffers)
    assert isinstance(restored, PyTorchDotsTTSEditAudioModel)
    assert restored._runtime is None
    assert restored._generation_lock is None
    assert restored._stream_executor is None
    assert restored.model_family.model_hub == "modelscope"


def test_load_uses_cached_path_and_runtime_options(model, runtime_module):
    model._kwargs.update(
        optimize=True, warmup_on_optimize=False, max_generate_length=200
    )
    model.load()
    path, options = runtime_module[0]
    assert path == model._model_path
    assert options == {
        "precision": "float32",
        "optimize": True,
        "warmup_on_optimize": False,
        "max_generate_length": 200,
    }
    assert model._device == "cpu"
    assert not model._generation_lock.locked()


@pytest.mark.parametrize(
    "options, message",
    [
        ({"precision": "bfloat16"}, "float32"),
        ({"unknown_option": True}, "Unsupported"),
    ],
)
def test_load_rejects_invalid_options(model, runtime_module, options, message):
    model._kwargs.update(options)
    with pytest.raises(ValueError, match=message):
        model.load()
    assert not runtime_module


def test_load_rejects_unsupported_python(model, monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 13, 0))
    with pytest.raises(RuntimeError, match="Python 3.10 through 3.12"):
        model.load()


@pytest.mark.parametrize("fmt", ["wav", "mp3", "flac", "ogg", "pcm"])
def test_real_audio_encoding(model, fmt):
    output = model.speech("Hello world.", response_format=fmt)
    options = (
        {"format": "RAW", "subtype": "PCM_16", "samplerate": 48000, "channels": 1}
        if fmt == "pcm"
        else {}
    )
    audio, rate = soundfile.read(io.BytesIO(output), **options)
    assert rate == 48000
    assert audio.shape == (2400,)
    assert np.max(np.abs(audio)) > 0


def test_voice_cloning_and_sampler_options(model):
    model.speech(
        "Hello",
        prompt_speech=b"reference",
        prompt_text="Reference transcript",
        language="EN",
        num_steps=20,
        guidance_scale=1.2,
        speaker_scale=1.5,
        ode_method="midpoint",
        normalize_text=True,
        use_xvector=False,
        template_name="tts",
        seed=42,
    )
    task, options = model._runtime.requests[0]
    assert task == "tts"
    assert options["prompt_text"] == "Reference transcript"
    assert options["num_steps"] == 20
    assert options["ode_method"] == "midpoint"
    assert options["language"] == "EN"
    assert options["normalize_text"] is True
    assert options["use_xvector"] is False
    assert "seed" not in options
    assert not model._runtime.paths[0].exists()


def test_xvector_only_cloning(model):
    model.speech("Hello", prompt_speech=b"reference")
    assert model._runtime.requests[0][1]["prompt_text"] is None


@pytest.mark.parametrize(
    "selector",
    [
        {"task": "edit"},
        {"template_name": "edit"},
        {"instruction": "tagged"},
        {"instruct": "tagged"},
    ],
)
def test_edit_preserves_instruction_and_transcript_derivation(model, selector):
    tagged = 'Hello <sub targ="small">brave</sub> world.'
    options = {k: tagged if v == "tagged" else v for k, v in selector.items()}
    model.speech(tagged, prompt_speech=b"reference", use_xvector="auto", **options)
    task, options = model._runtime.requests[0]
    assert task == "edit"
    assert options["instruction"] == tagged
    assert options["source_text"] is None
    assert options["target_text"] is None
    assert options["use_xvector"] == "auto"
    assert not model._runtime.paths[0].exists()


def test_edit_transcript_overrides(model):
    model.speech(
        "tagged instruction",
        task="edit",
        prompt_speech=b"reference",
        prompt_text="Original text",
        target_text="Edited text",
    )
    assert model._runtime.requests[0][1]["source_text"] == "Original text"
    assert model._runtime.requests[0][1]["target_text"] == "Edited text"


@pytest.mark.parametrize(
    "task,path_key", [("tts", "prompt_audio_path"), ("edit", "source_audio_path")]
)
def test_local_audio_path_is_preserved(model, tmp_path, task, path_key):
    path = tmp_path / "source.wav"
    path.write_bytes(b"reference")
    model.speech("Hello", task=task, **{path_key: str(path)})
    assert model._runtime.requests[0][1][path_key] == str(path)
    assert path.exists()


def test_pcm_stream_is_incremental_and_closes_upstream(model):
    stream = model.speech(
        "Hello", stream=True, response_format="pcm", prompt_speech=b"reference"
    )
    assert model._runtime.requests == []
    first = next(stream)
    assert len(first) == 800 * 2
    path = model._runtime.paths[0]
    assert path.exists()
    assert not model._runtime.closed
    with pytest.raises(RuntimeError, match="active generation"):
        model.speech("Overlapping request")
    stream.close()
    assert model._runtime.closed
    assert not path.exists()
    assert not model._generation_lock.locked()
    assert model.speech("Next request")


def test_compressed_stream_uses_shared_encoder(model, monkeypatch):
    def encode(fmt, rate, chunks, transform):
        assert fmt == "mp3"
        assert rate == 48000
        for chunk in chunks:
            assert transform(chunk).shape == (800, 1)
            yield b"encoded chunk"

    monkeypatch.setattr(utils, "audio_stream_generator", encode)
    assert list(model.speech("Hello", stream=True)) == [b"encoded chunk"] * 3
    assert model._runtime.closed


def test_stream_can_resume_and_close_on_different_threads(model):
    stream = model.speech("Hello", stream=True, response_format="pcm", seed=42)
    state = torch.random.get_rng_state().clone()

    def read():
        chunk = next(stream)
        return chunk, torch.is_inference_mode_enabled()

    with ThreadPoolExecutor(max_workers=1) as executor:
        chunk, inference_mode = executor.submit(read).result()
        assert chunk
        assert not inference_mode
        assert next(stream)
        assert not torch.is_inference_mode_enabled()
        executor.submit(stream.close).result()
    assert torch.equal(state, torch.random.get_rng_state())
    assert model._runtime.closed
    assert not model._generation_lock.locked()


def test_stream_keeps_upstream_autocast_on_one_thread(model, monkeypatch):
    threads = []
    closed = []

    def generate_stream(**options):
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                for _ in range(3):
                    assert torch.is_autocast_enabled("cpu")
                    threads.append(threading.get_ident())
                    yield torch.ones(1, 800) * 0.2
        finally:
            closed.append(threading.get_ident())

    monkeypatch.setattr(model._runtime, "generate_stream", generate_stream)
    stream = model.speech("Hello", stream=True, response_format="pcm")
    with ThreadPoolExecutor(max_workers=1) as caller:
        assert caller.submit(next, stream).result()
        assert next(stream)
        caller.submit(stream.close).result()
    assert len(threads) == 2
    assert len(set(threads + closed)) == 1
    assert threads[0] != threading.get_ident()
    assert not torch.is_autocast_enabled("cpu")


def test_failure_cleans_temporary_audio(model):
    model._runtime.fail = True
    with pytest.raises(
        InvalidModelRequestError, match="Invalid tagged edit instruction"
    ):
        model.speech("Hello", task="edit", prompt_speech=b"reference")
    assert not model._runtime.paths[0].exists()
    assert not model._generation_lock.locked()


def test_seed_is_repeatable_and_restores_rng(model):
    state = torch.random.get_rng_state().clone()
    first = model.speech("Hello", seed=42)
    assert torch.equal(state, torch.random.get_rng_state())
    assert first == model.speech("Hello", seed=42)
    assert first != model.speech("Hello", seed=43)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"input": ""}, "non-empty"),
        ({"speed": 2.0}, "speed=1.0"),
        ({"response_format": "unknown"}, "response_format"),
        ({"stream": "yes"}, "boolean"),
        ({"task": "other"}, "task"),
        ({"task": "edit"}, "requires prompt_speech"),
        ({"task": "edit", "stream": True}, "does not support streaming"),
        ({"task": "tts", "instruction": "edit"}, "require task"),
        ({"task": "edit", "template_name": "tts"}, "template_name"),
        ({"prompt_speech": b""}, "non-empty audio"),
        ({"prompt_speech": "path.wav"}, "audio bytes"),
        (
            {"prompt_speech": b"audio", "prompt_audio_path": "a.wav"},
            "cannot be combined",
        ),
        ({"prompt_audio_path": "nonexistent.wav"}, "existing audio file"),
        ({"prompt_text": "Reference"}, "requires reference audio"),
        ({"instruction": "one", "instruct": "two"}, "must agree"),
        (
            {
                "task": "edit",
                "prompt_speech": b"audio",
                "source_text": "one",
                "prompt_text": "two",
            },
            "must agree",
        ),
        ({"num_steps": 0}, "positive integer"),
        ({"num_steps": True}, "positive integer"),
        ({"guidance_scale": float("nan")}, "finite"),
        ({"speaker_scale": -1}, "non-negative"),
        ({"use_xvector": "auto"}, "boolean"),
        ({"seed": True}, "Seed"),
        ({"unknown": True}, "Unsupported"),
    ],
)
def test_invalid_requests_fail_before_generation(model, options, message):
    options = {"input": "Hello", **options}
    with pytest.raises(InvalidModelRequestError, match=message):
        model.speech(**options)
    assert model._runtime.requests == []
