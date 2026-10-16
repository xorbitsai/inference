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
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from ... import utils as model_utils
from .. import yue2 as yue2_module
from ..yue2 import YuE2Model


@pytest.fixture
def model_spec():
    return SimpleNamespace(
        default_model_config={
            "backend": "torch",
            "memory_budget_gib": 24,
            "quantization": "none",
            "offload_ar": False,
            "vae_core_frames": None,
            "verify_hashes": True,
        },
        model_ability=["text2music"],
        model_hub="huggingface",
        vae_model_id="m-a-p/YuE2-Vae",
        vae_model_revision="main",
    )


def _install_fake_yue2_runtime(monkeypatch):
    class Device:
        def __init__(self, name):
            self.name = name
            self.type = name.split(":", 1)[0]

        def __str__(self):
            return self.name

    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
    )
    torch_module.version = SimpleNamespace(hip=None)
    torch_module.device = Device

    yue2_module = ModuleType("yue2")
    pipeline = Mock(name="YuE2Pipeline")
    yue2_module.YuE2Pipeline = pipeline

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "yue2", yue2_module)
    return pipeline


def test_load_uses_cached_model_and_matching_vae_source(monkeypatch, model_spec):
    pipeline = _install_fake_yue2_runtime(monkeypatch)
    model = YuE2Model("yue2-test", "/models/YuE2-3B", model_spec, device="cuda:1")
    download_vae = Mock(return_value="/models/YuE2-Vae")
    monkeypatch.setattr(model, "_download_vae", download_vae)

    model.load()

    assert sys.path[0] == yue2_module._YUE2_VENDOR_ROOT
    download_vae.assert_called_once_with()
    pipeline.assert_called_once_with(
        "/models/YuE2-3B",
        "/models/YuE2-Vae",
        device="cuda:1",
        progress=False,
        backend="torch",
        memory_budget_gib=24,
        quantization="none",
        offload_ar=False,
        vae_core_frames=None,
        verify_hashes=True,
    )
    assert model._model is pipeline.return_value


def test_download_vae_uses_modelscope_revision(monkeypatch, model_spec):
    modelscope = ModuleType("modelscope")
    hub = ModuleType("modelscope.hub")
    snapshot_module = ModuleType("modelscope.hub.snapshot_download")
    snapshot_download = Mock()
    snapshot_module.snapshot_download = snapshot_download
    modelscope.hub = hub
    hub.snapshot_download = snapshot_module
    monkeypatch.setitem(sys.modules, "modelscope", modelscope)
    monkeypatch.setitem(sys.modules, "modelscope.hub", hub)
    monkeypatch.setitem(
        sys.modules, "modelscope.hub.snapshot_download", snapshot_module
    )

    model_spec.model_hub = "modelscope"
    model_spec.vae_model_revision = "master"
    model = YuE2Model("yue2-test", "/unused", model_spec)
    download = Mock(return_value="/models/YuE2-Vae")
    monkeypatch.setattr(model_utils, "retry_snapshot_download", download)

    assert model._download_vae() == "/models/YuE2-Vae"
    download.assert_called_once_with(
        snapshot_download,
        "m-a-p/YuE2-Vae",
        None,
        "m-a-p/YuE2-Vae",
        revision="master",
    )


@pytest.mark.parametrize(
    ("speech_kwargs", "message"),
    [
        ({"input": "", "instruct": "piano"}, "requires non-empty lyrics"),
        ({"input": "lyrics", "instruct": None}, "non-empty music description"),
        (
            {"input": "lyrics", "instruct": "piano", "voice": "alloy"},
            "only accepts `voice`",
        ),
        (
            {"input": "lyrics", "instruct": "piano", "response_format": "aac"},
            "supports these response formats",
        ),
        (
            {"input": "lyrics", "instruct": "piano", "speed": 1.5},
            "only supports `speed=1.0`",
        ),
        (
            {"input": "lyrics", "instruct": "piano", "stream": True},
            "only supports non-streaming",
        ),
        (
            {"input": "lyrics", "instruct": "piano", "seed": -1},
            "integer in \\[0, 2\\*\\*63\\)",
        ),
    ],
)
def test_speech_validates_requests(model_spec, speech_kwargs, message):
    model = YuE2Model("yue2-test", "/unused", model_spec)
    model._model = Mock()

    with pytest.raises(ValueError, match=message):
        model.speech(**speech_kwargs)

    model._model.assert_not_called()


def test_speech_maps_request_to_yue2_pipeline(model_spec):
    model = YuE2Model("yue2-test", "/unused", model_spec)
    model._model = Mock(
        return_value=SimpleNamespace(
            audio=np.zeros((16, 2), dtype=np.float32), sample_rate=48000
        )
    )
    model._audio_to_bytes = Mock(return_value=b"encoded audio")

    result = model.speech(
        "[Verse]\\nlyrics",
        instruct="  warm piano pop  ",
        response_format="wav",
        seed=7,
        duration=60,
        cot="melody",
        cfg_scale=1.2,
    )

    assert result == b"encoded audio"
    model._model.assert_called_once_with(
        style="warm piano pop",
        lyrics="[Verse]\\nlyrics",
        seed=7,
        cot="melody",
        cfg_scale=1.2,
    )
    model._audio_to_bytes.assert_called_once_with(
        model._model.return_value.audio,
        48000,
        "wav",
    )


def test_audio_to_bytes_preserves_yue2_stereo_sample_rate():
    soundfile = pytest.importorskip("soundfile", minversion="0.13.1")

    encoded = YuE2Model._audio_to_bytes(
        np.zeros((480, 2), dtype=np.float32), 48000, "flac"
    )
    info = soundfile.info(io.BytesIO(encoded))

    assert info.format == "FLAC"
    assert info.samplerate == 48000
    assert info.channels == 2


def test_stop_closes_pipeline(model_spec):
    model = YuE2Model("yue2-test", "/unused", model_spec)
    pipeline = Mock()
    model._model = pipeline

    model.stop()

    pipeline.close.assert_called_once_with()
    assert model._model is None
