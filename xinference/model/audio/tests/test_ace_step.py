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
import os
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import ANY, Mock

import numpy as np
import pytest

from .. import ace_step
from ..ace_step import AceStepModel


@pytest.fixture
def model_spec():
    return SimpleNamespace(
        default_model_config={},
        model_ability=["text_to_audio"],
        model_hub="huggingface",
    )


@pytest.fixture
def loaded_model(tmp_path, model_spec):
    model = AceStepModel(
        "ace-step-test",
        str(tmp_path),
        model_spec,
        device="cpu",
    )
    model._model = object()
    model._generation_params_cls = Mock(name="GenerationParams")
    model._generation_config_cls = Mock(name="GenerationConfig")
    model._generate_music = Mock(name="generate_music")
    return model


def _install_fake_acestep_runtime(monkeypatch, handler_cls) -> None:
    package = ModuleType("acestep")
    package.__path__ = []

    handler_module = ModuleType("acestep.handler")
    setattr(handler_module, "AceStepHandler", handler_cls)

    inference_module = ModuleType("acestep.inference")
    setattr(inference_module, "GenerationConfig", Mock(name="GenerationConfig"))
    setattr(inference_module, "GenerationParams", Mock(name="GenerationParams"))
    setattr(inference_module, "generate_music", Mock(name="generate_music"))

    monkeypatch.setitem(sys.modules, "acestep", package)
    monkeypatch.setitem(sys.modules, "acestep.handler", handler_module)
    monkeypatch.setitem(sys.modules, "acestep.inference", inference_module)
    monkeypatch.setattr(ace_step, "is_ace_step_python_supported", lambda: True)
    monkeypatch.setattr(ace_step, "_ensure_vendored_source_paths", lambda: None)


@pytest.mark.parametrize(
    ("config", "device", "message"),
    [
        (
            {"unsupported_option": True},
            "cpu",
            "Unsupported ACE-Step 1.5 load option(s): unsupported_option",
        ),
        (
            {"config_path": "other-model"},
            "cpu",
            "`config_path=acestep-v15-turbo`",
        ),
        (
            {"lm_model_path": "other-lm"},
            "cpu",
            "`lm_model_path=acestep-5Hz-lm-1.7B` or no LM",
        ),
        (
            {"vae_checkpoint": "other-vae"},
            "cpu",
            "`vae_checkpoint=official`",
        ),
        (
            {"lm_backend": "invalid"},
            "cpu",
            "`lm_backend` must be one of",
        ),
        (
            {"lm_offload_to_cpu": "yes"},
            "cpu",
            "`lm_offload_to_cpu` must be a boolean",
        ),
        ({}, "tpu", "does not support device 'tpu'"),
    ],
)
def test_load_validates_options_before_creating_workspace(
    monkeypatch, tmp_path, model_spec, config, device, message
):
    class UnusedHandler:
        pass

    _install_fake_acestep_runtime(monkeypatch, UnusedHandler)
    model_path = tmp_path / "model"
    model_path.mkdir()
    model = AceStepModel(
        "ace-step-test",
        str(model_path),
        model_spec,
        device=device,
        **config,
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        model.load()

    assert not list(tmp_path.glob("xinference-ace-step-runtime-*"))


def test_load_cleans_workspace_and_environment_after_initialization_failure(
    monkeypatch, tmp_path, model_spec
):
    class FailingHandler:
        project_root = None
        checkpoint_dir = None

        def initialize_service(self, **kwargs):
            type(self).project_root = kwargs["project_root"]
            type(self).checkpoint_dir = os.environ["ACESTEP_CHECKPOINTS_DIR"]
            return "broken initialization", False

    _install_fake_acestep_runtime(monkeypatch, FailingHandler)
    monkeypatch.setenv("ACESTEP_CHECKPOINTS_DIR", "original-checkpoints")
    monkeypatch.setenv("ACESTEP_PROJECT_ROOT", "original-project")
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.py").write_text("MODEL_CONFIG = {}")
    model = AceStepModel(
        "ace-step-test",
        str(model_path),
        model_spec,
        device="cpu",
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to initialize ACE-Step 1.5: broken initialization",
    ):
        model.load()

    assert FailingHandler.project_root is not None
    assert FailingHandler.checkpoint_dir is not None
    assert not Path(FailingHandler.project_root).exists()
    assert not Path(FailingHandler.checkpoint_dir).exists()
    assert model._runtime_workspace is None
    assert os.environ["ACESTEP_CHECKPOINTS_DIR"] == "original-checkpoints"
    assert os.environ["ACESTEP_PROJECT_ROOT"] == "original-project"


def test_prepare_runtime_checkpoint_isolates_mutable_model_files(tmp_path, model_spec):
    model_path = tmp_path / "model"
    model_path.mkdir()
    source_config = model_path / "config.py"
    source_config.write_text("ORIGINAL = True")
    (model_path / "weights.bin").write_bytes(b"weights")
    model = AceStepModel("ace-step-test", str(model_path), model_spec, device="cpu")

    workspace, checkpoint_dir = model._prepare_runtime_checkpoint()
    workspace_path = Path(workspace.name)
    checkpoint_path = Path(checkpoint_dir)
    try:
        assert checkpoint_path.parent == workspace_path
        assert checkpoint_path != model_path
        assert (checkpoint_path / "weights.bin").read_bytes() == b"weights"

        runtime_config = checkpoint_path / "config.py"
        assert runtime_config.read_text() == "ORIGINAL = True"
        runtime_config.write_text("ORIGINAL = False")
        assert source_config.read_text() == "ORIGINAL = True"
    finally:
        workspace.cleanup()

    assert not workspace_path.exists()


@pytest.mark.parametrize(
    ("speech_kwargs", "message"),
    [
        (
            {"input": "", "instruct": "piano"},
            "requires non-empty lyrics in `input`",
        ),
        (
            {"input": "lyrics", "instruct": None},
            "requires a non-empty music description in `instruct`",
        ),
        (
            {"input": "lyrics", "instruct": "piano", "voice": "alloy"},
            "only accepts `voice` as null",
        ),
        (
            {"input": "lyrics", "instruct": "piano", "response_format": "m4a"},
            "supports these response formats",
        ),
        (
            {"input": "lyrics", "instruct": "piano", "speed": 1.5},
            "only supports `speed=1.0`",
        ),
        (
            {"input": "lyrics", "instruct": "piano", "stream": True},
            "only supports non-streaming generation",
        ),
        (
            {"input": "lyrics", "instruct": "piano", "seed": True},
            "`seed` must be -1 or a non-negative integer",
        ),
        (
            {"input": "lyrics", "instruct": "piano", "duration": 9},
            "`duration` must be -1 or a number from 10 to 600 seconds",
        ),
        (
            {"input": "lyrics", "instruct": "piano", "unsupported": True},
            "does not support speech parameter(s): unsupported",
        ),
        (
            {"input": "lyrics", "instruct": "piano", "thinking": "yes"},
            "`thinking` must be a boolean",
        ),
        (
            {"input": "lyrics", "instruct": "piano", "thinking": True},
            "LM generation options require launching with",
        ),
    ],
)
def test_speech_validates_requests(loaded_model, speech_kwargs, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        loaded_model.speech(**speech_kwargs)

    loaded_model._generate_music.assert_not_called()


def test_speech_returns_generated_audio(loaded_model):
    generated_paths = []

    def generate_music(*args, save_dir, **kwargs):
        audio_path = Path(save_dir) / "output.wav"
        audio_path.write_bytes(b"generated audio")
        generated_paths.append(audio_path)
        return SimpleNamespace(
            success=True,
            error=None,
            status_message="success",
            audios=[{"path": str(audio_path)}],
        )

    params = object()
    generation_config = object()
    loaded_model._generation_params_cls.return_value = params
    loaded_model._generation_config_cls.return_value = generation_config
    loaded_model._generate_music.side_effect = generate_music

    result = loaded_model.speech(
        "lyrics",
        instruct="  warm piano  ",
        response_format="wav",
        seed=7,
        duration=30,
        bpm=120,
    )

    assert result == b"generated audio"
    loaded_model._generation_params_cls.assert_called_once_with(
        bpm=120,
        task_type="text2music",
        caption="warm piano",
        lyrics="lyrics",
        duration=30.0,
        seed=7,
        thinking=False,
        use_cot_caption=False,
        use_cot_language=False,
        use_cot_metas=False,
    )
    loaded_model._generation_config_cls.assert_called_once_with(
        batch_size=1,
        use_random_seed=False,
        seeds=[7],
        audio_format="wav",
    )
    loaded_model._generate_music.assert_called_once_with(
        loaded_model._model,
        None,
        params,
        generation_config,
        save_dir=ANY,
    )
    assert len(generated_paths) == 1
    assert not generated_paths[0].exists()


def test_speech_transcodes_ogg_from_native_wav(loaded_model):
    generated_paths = []

    def generate_music(*args, save_dir, **kwargs):
        audio_path = Path(save_dir) / "output.wav"
        audio_path.write_bytes(b"native wav")
        generated_paths.append(audio_path)
        return SimpleNamespace(
            success=True,
            error=None,
            status_message="success",
            audios=[{"path": str(audio_path)}],
        )

    loaded_model._generate_music.side_effect = generate_music
    loaded_model._wav_file_to_ogg = Mock(return_value=b"encoded ogg")

    result = loaded_model.speech(
        "lyrics",
        instruct="piano",
        response_format="ogg",
        seed=7,
        duration=30,
    )

    assert result == b"encoded ogg"
    loaded_model._generation_config_cls.assert_called_once_with(
        batch_size=1,
        use_random_seed=False,
        seeds=[7],
        audio_format="wav",
    )
    assert len(generated_paths) == 1
    loaded_model._wav_file_to_ogg.assert_called_once_with(str(generated_paths[0]))
    assert not generated_paths[0].exists()


def test_wav_file_to_ogg_encodes_vorbis(tmp_path):
    soundfile = pytest.importorskip("soundfile", minversion="0.13.1")
    wav_path = tmp_path / "native.wav"
    soundfile.write(
        wav_path,
        np.zeros((2400, 2), dtype=np.float32),
        24000,
        format="WAV",
    )

    encoded = AceStepModel._wav_file_to_ogg(str(wav_path))
    info = soundfile.info(io.BytesIO(encoded))

    assert encoded.startswith(b"OggS")
    assert info.format == "OGG"
    assert info.subtype == "VORBIS"
    assert info.samplerate == 24000
    assert info.channels == 2


@pytest.mark.parametrize(
    ("generation_result", "message"),
    [
        (
            SimpleNamespace(
                success=False,
                error="backend failure",
                status_message="failed",
                audios=[],
            ),
            "generation failed: backend failure",
        ),
        (
            SimpleNamespace(
                success=True,
                error=None,
                status_message="success",
                audios=[],
            ),
            "returned no saved audio output",
        ),
    ],
)
def test_speech_rejects_failed_or_missing_output(
    loaded_model, generation_result, message
):
    loaded_model._generate_music.return_value = generation_result

    with pytest.raises(RuntimeError, match=re.escape(message)):
        loaded_model.speech("lyrics", instruct="piano")
