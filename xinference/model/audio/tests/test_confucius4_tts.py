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
import sys
from importlib.machinery import PathFinder
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml  # type: ignore[import-untyped]

from .. import BUILTIN_AUDIO_MODELS
from .. import confucius4_tts as confucius_module
from .. import load_model_family_from_json
from ..confucius4_tts import Confucius4TTSModel
from ..core import create_audio_model_instance, match_audio


def _import_confuciustts_module(module_name):
    thirdparty_dir = str(Path(__file__).resolve().parents[3] / "thirdparty")
    if thirdparty_dir not in sys.path:
        sys.path.insert(0, thirdparty_dir)
    import importlib

    return importlib.import_module(module_name)


@pytest.fixture
def model_spec():
    return SimpleNamespace(
        model_name="Confucius4-TTS",
        model_family="Confucius4-TTS",
        model_ability=["text2audio", "text2audio_voice_cloning"],
    )


def test_catalog_registers_both_hubs(monkeypatch, tmp_path):
    catalog = Path(__file__).parents[1] / "models" / "Confucius4-TTS.json"
    model_config = json.loads(catalog.read_text(encoding="utf-8"))[0]
    sources = model_config["model_src"]
    assert sources["huggingface"]["model_revision"] == "main"
    assert sources["modelscope"]["model_revision"] == "master"
    assert "huggingface-hub>=0.36,<1.0" in model_config["virtualenv"]["packages"]
    assert model_config["virtualenv"]["index_strategy"] == "unsafe-best-match"
    assert all(
        "confuciustts @ git+" not in package
        for package in model_config["virtualenv"]["packages"]
    )
    thirdparty_dir = Path(__file__).parents[3] / "thirdparty"
    vendored = PathFinder.find_spec("confuciustts", [str(thirdparty_dir)])
    assert vendored is not None
    assert vendored.origin == str(thirdparty_dir / "confuciustts" / "__init__.py")
    assert (thirdparty_dir / "external" / "bigvgan" / "bigvgan.py").is_file()

    families = {}
    load_model_family_from_json(str(catalog), families)
    monkeypatch.setitem(
        BUILTIN_AUDIO_MODELS, "Confucius4-TTS", families["Confucius4-TTS"]
    )
    assert match_audio("Confucius4-TTS", "huggingface").model_hub == "huggingface"
    assert match_audio("Confucius4-TTS", "modelscope").model_hub == "modelscope"
    model = create_audio_model_instance(
        "confucius", "Confucius4-TTS", "modelscope", model_path=str(tmp_path)
    )
    assert isinstance(model, Confucius4TTSModel)
    assert model._model_spec.model_hub == "modelscope"


@pytest.mark.parametrize("model_hub", ["huggingface", "modelscope"])
def test_load_uses_selected_snapshot_for_main_weights(
    monkeypatch, tmp_path, model_spec, model_hub
):
    for name in ("t2s_model.safetensors", "s2a_model.pt", "wav2vec2bert_stats.pt"):
        (tmp_path / name).touch()

    calls = {}
    model_spec.model_hub = model_hub
    if model_hub == "modelscope":
        from modelscope.hub import file_download

        def download_auxiliary(repo_id, filename, revision):
            calls.setdefault("auxiliary", []).append((repo_id, filename, revision))
            return str(tmp_path / "auxiliary" / repo_id.split("/")[1] / filename)

        monkeypatch.setattr(file_download, "model_file_download", download_auxiliary)

    def original_download(repo_id, filename, **kwargs):
        calls["external"] = (repo_id, filename)
        return "/external/campplus.bin"

    class FakeRuntime:
        def __init__(self, config_path, device):
            config = yaml.safe_load(Path(config_path).read_text())
            calls["config"] = config
            calls["config_path"] = config_path
            calls["device"] = device
            calls["t2s"] = upstream.hf_hub_download(
                "netease-youdao/Confucius4-TTS", config["paths"]["t2s_checkpoint"]
            )
            calls["s2a"] = upstream.hf_hub_download(
                "netease-youdao/Confucius4-TTS", config["paths"]["s2a_checkpoint"]
            )
            calls["campplus"] = upstream.hf_hub_download(
                "funasr/campplus", "campplus_cn_common.bin"
            )

    upstream = SimpleNamespace(
        hf_hub_download=original_download, ConfuciusTTS=FakeRuntime
    )

    def import_upstream(name):
        assert name == "confuciustts.cli.inference"
        calls["thirdparty_dir"] = confucius_module.sys.path[0]
        return upstream

    monkeypatch.setattr(confucius_module.importlib, "import_module", import_upstream)
    model = Confucius4TTSModel("confucius", str(tmp_path), model_spec, device="cpu")
    model.load()

    assert calls["t2s"] == str(tmp_path / "t2s_model.safetensors")
    assert calls["s2a"] == str(tmp_path / "s2a_model.pt")
    assert calls["config"]["paths"]["tokenizer_path"] == str(tmp_path)
    assert calls["config"]["paths"]["w2v_stat"] == str(
        tmp_path / "wav2vec2bert_stats.pt"
    )
    if model_hub == "modelscope":
        assert calls["auxiliary"] == [
            ("facebook/w2v-bert-2.0", name, "master")
            for name in ("config.json", "preprocessor_config.json", "model.safetensors")
        ] + [
            (
                "iic/speech_campplus_sv_zh-cn_16k-common",
                "campplus_cn_common.bin",
                "master",
            )
        ] + [
            ("nv-community/bigvgan_v2_22khz_80band_256x", name, "master")
            for name in ("config.json", "bigvgan_generator.pt")
        ]
        assert calls["config"]["paths"]["w2v_bert_path"] == str(
            tmp_path / "auxiliary" / "w2v-bert-2.0"
        )
        assert calls["config"]["paths"]["vocoder_path"] == str(
            tmp_path / "auxiliary" / "bigvgan_v2_22khz_80band_256x"
        )
        assert calls["campplus"] == str(
            tmp_path
            / "auxiliary"
            / "speech_campplus_sv_zh-cn_16k-common"
            / "campplus_cn_common.bin"
        )
    else:
        assert calls["config"]["paths"]["w2v_bert_path"] == "facebook/w2v-bert-2.0"
        assert calls["external"] == ("funasr/campplus", "campplus_cn_common.bin")
        assert calls["campplus"] == "/external/campplus.bin"
    assert calls["device"] == "cpu"
    assert calls["thirdparty_dir"] == str(Path(__file__).parents[3] / "thirdparty")
    assert upstream.hf_hub_download is original_download
    assert not Path(calls["config_path"]).exists()


def test_speech_uses_reference_and_cleans_temp_file(monkeypatch, tmp_path, model_spec):
    calls = {}

    class FakeRuntime:
        sample_rate = 22050

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            calls["reference"] = Path(kwargs["prompt_wav"]).read_bytes()
            return torch.tensor([[0.1, 0.2]], dtype=torch.float32)

    def encode(response_format, sample_rate, audio):
        calls["encode"] = response_format, sample_rate, audio.tolist()
        return b"encoded"

    from .. import utils

    monkeypatch.setattr(utils, "audio_to_bytes", encode)
    model = Confucius4TTSModel("confucius", str(tmp_path), model_spec)
    model._model = FakeRuntime()
    assert (
        model.speech(
            "Bonjour", prompt_speech=b"reference", language="fr", response_format="wav"
        )
        == b"encoded"
    )
    assert calls["reference"] == b"reference"
    assert calls["generate"]["lang"] == "fr"
    assert calls["encode"][:2] == ("wav", 22050)
    assert calls["encode"][2][0] == pytest.approx([0.1, 0.2])
    assert not Path(calls["generate"]["prompt_wav"]).exists()


def test_speech_rejects_missing_reference_and_stream(tmp_path, model_spec):
    model = Confucius4TTSModel("confucius", str(tmp_path), model_spec)
    model._model = object()
    with pytest.raises(ValueError, match="prompt_speech"):
        model.speech("Hello")
    with pytest.raises(ValueError, match="streaming"):
        model.speech("Hello", stream=True, prompt_speech=b"reference")


def test_vllm_model_dir_copies_weights_when_symlinks_fail(monkeypatch, tmp_path):
    from xinference.thirdparty.confuciustts.llm import vllm_compat

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"weights")
    temp_dir = tmp_path / "fallback"

    def make_temp_dir(prefix):
        temp_dir.mkdir()
        return str(temp_dir)

    def fail_symlink(*args):
        raise OSError("symlinks unavailable")

    monkeypatch.setattr(vllm_compat.tempfile, "mkdtemp", make_temp_dir)
    monkeypatch.setattr(vllm_compat.os, "symlink", fail_symlink)

    result = vllm_compat.prepare_vllm_model_dir(str(model_dir), str(checkpoint))

    assert result == str(temp_dir)
    assert (temp_dir / "config.json").read_text(encoding="utf-8") == "{}"
    assert (temp_dir / "model.safetensors").read_bytes() == b"weights"


def test_vllm_position_correction_shifts_prompt_prefix():
    from xinference.thirdparty.confuciustts.llm.vllm_compat import (
        correct_confucius_positions,
    )

    class FakePositions:
        def __init__(self):
            self.np = np.array([0, 1, 0])
            self.copied_tokens = None

        def copy_to_gpu(self, num_tokens):
            self.copied_tokens = num_tokens

    positions = FakePositions()
    runner = SimpleNamespace(
        arange_np=np.arange(2),
        input_batch=SimpleNamespace(num_reqs=2, req_ids=["first", "second"]),
        positions=positions,
        requests={
            "first": SimpleNamespace(prompt_token_ids=[1, 2, 3]),
            "second": SimpleNamespace(prompt_token_ids=[4, 5]),
        },
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=3)

    correct_confucius_positions(runner, scheduler_output, np.array([2, 1]))

    assert positions.np.tolist() == [-2, -1, -1]
    assert positions.copied_tokens == 3


def test_text_normalizer_splits_long_segments_even_with_multiple_sentences():
    TextNormalizer = _import_confuciustts_module(
        "confuciustts.frontend.text_normalizer"
    ).TextNormalizer

    text = "Short. " + "longword " * 16 + "Tail."
    segments = TextNormalizer().segment_text(
        text,
        tokenize_fn=lambda value: value.split(),
        language="en",
        max_tokens=5,
        min_tokens=1,
    )

    assert "".join(segments) == text
    assert all(len(segment.split()) <= 5 for segment in segments)


def test_audio_feature_caches_have_a_fixed_bound():
    audio_features = _import_confuciustts_module("confuciustts.utils.audio_features")

    assert audio_features._get_mel_transform.cache_info().maxsize == 16
    assert audio_features._get_mel_spectrogram_basis.cache_info().maxsize == 16


def test_bigvgan_cuda_loader_reports_missing_cuda_home(monkeypatch):
    from xinference.thirdparty.external.bigvgan.alias_free_activation.cuda import load

    monkeypatch.setattr(load.cpp_extension, "CUDA_HOME", None)
    with pytest.raises(RuntimeError, match="CUDA_HOME is not set"):
        load.load()
