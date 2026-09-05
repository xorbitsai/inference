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
import inspect
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .. import fish_speech as fish_speech_module
from .. import load_model_family_from_json
from ..core import create_audio_model_instance
from ..fish_speech import FISH_AUDIO_S1_MINI, FISH_AUDIO_S2_PRO, FishSpeechModel


def _model_spec(model_name):
    return SimpleNamespace(
        model_name=model_name,
        model_family="FishAudio",
        model_ability=[
            "text2audio",
            "text2audio_zero_shot",
            "text2audio_voice_cloning",
        ],
        engine=None,
    )


class _FakeSchema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.mark.parametrize(
    "module_name, expected_dtype",
    [
        (
            "xinference.thirdparty.fish_speech_s1.fish_speech.content_sequence",
            torch.int,
        ),
        (
            "xinference.thirdparty.fish_speech_s2.fish_speech.content_sequence",
            torch.long,
        ),
    ],
)
def test_vendored_content_sequence_encodes_audio_parts(module_name, expected_dtype):
    module = importlib.import_module(module_name)
    token_ids = {
        module.AUDIO_START_TOKEN: 10,
        module.AUDIO_END_TOKEN: 11,
        module.AUDIO_EMBED_TOKEN: 12,
    }
    tokenizer = SimpleNamespace(
        get_token_id=token_ids.__getitem__,
        semantic_begin_id=100,
    )
    features = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    sequence = module.ContentSequence(parts=[module.AudioPart(features=features)])

    encoded = sequence.encode(tokenizer, add_shift=False)

    assert encoded.tokens.dtype == expected_dtype
    assert encoded.tokens.tolist() == [10, 12, 12, 11]
    assert encoded.audio_masks.tolist() == [False, True, True, False]
    assert len(encoded.audio_parts) == 1
    torch.testing.assert_close(encoded.audio_parts[0], features)

    values, audio_masks, audio_parts = sequence.encode_for_inference(
        tokenizer, num_codebooks=2
    )
    assert values.shape == (3, 4)
    assert audio_masks.tolist() == [[False, True, True, False]]
    torch.testing.assert_close(audio_parts, features)


@pytest.mark.parametrize("model_name", [FISH_AUDIO_S1_MINI, FISH_AUDIO_S2_PRO])
def test_load_modern_fish_audio_runtime(monkeypatch, model_name):
    captured = {}
    queue = object()
    decoder = SimpleNamespace(sample_rate=44100)

    def launch_thread_safe_queue(**kwargs):
        captured["llama"] = kwargs
        return queue

    def load_decoder_model(**kwargs):
        captured["decoder"] = kwargs
        return decoder

    class FakeEngine:
        def __init__(self, *args):
            captured["engine"] = args

    components = (
        FakeEngine,
        launch_thread_safe_queue,
        load_decoder_model,
        _FakeSchema,
        _FakeSchema,
    )
    monkeypatch.setattr(
        fish_speech_module,
        "_load_fish_speech_runtime_components",
        lambda name: components,
    )
    monkeypatch.setattr(fish_speech_module, "is_device_available", lambda device: True)

    model = FishSpeechModel(
        "fish", "/models/fish", _model_spec(model_name), device="cpu"
    )
    model.load()

    assert captured["llama"]["checkpoint_path"] == "/models/fish"
    assert captured["llama"]["device"] == "cpu"
    assert captured["decoder"] == {
        "config_name": "modded_dac_vq",
        "checkpoint_path": "/models/fish/codec.pth",
        "device": "cpu",
    }
    assert captured["engine"][0:2] == (queue, decoder)


def test_load_keeps_legacy_decoder_contract(monkeypatch):
    captured = {}

    def load_decoder_model(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(spec_transform=SimpleNamespace(sample_rate=44100))

    components = (
        lambda *args: object(),
        lambda **kwargs: object(),
        load_decoder_model,
        _FakeSchema,
        _FakeSchema,
    )
    monkeypatch.setattr(
        fish_speech_module,
        "_load_fish_speech_runtime_components",
        lambda name: components,
    )
    monkeypatch.setattr(fish_speech_module, "is_device_available", lambda device: True)

    model = FishSpeechModel(
        "fish", "/models/fish", _model_spec("FishSpeech-1.5"), device="cpu"
    )
    model.load()

    assert captured == {
        "config_name": "firefly_gan_vq",
        "checkpoint_path": (
            "/models/fish/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        ),
        "device": "cpu",
    }


def test_modern_speech_builds_reference_request(monkeypatch):
    captured = {}

    class FakeReference(_FakeSchema):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured["reference"] = kwargs

    class FakeRequest(_FakeSchema):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured["request"] = kwargs

    class FakeEngine:
        def inference(self, request):
            yield SimpleNamespace(
                code="final",
                audio=(44100, np.array([0.1, -0.1], dtype=np.float32)),
                error=None,
            )

    from .. import utils as audio_utils

    def audio_to_bytes(response_format, sample_rate, tensor):
        captured["encoded"] = (response_format, sample_rate, tensor.numpy())
        return b"audio"

    monkeypatch.setattr(audio_utils, "audio_to_bytes", audio_to_bytes)
    model = FishSpeechModel(
        "fish", "/models/fish", _model_spec(FISH_AUDIO_S1_MINI), device="cpu"
    )
    model._serve_reference_audio = FakeReference
    model._serve_tts_request = FakeRequest
    model._engine = FakeEngine()
    model._model = SimpleNamespace(sample_rate=44100)

    result = model.speech(
        input="Hello",
        voice="",
        response_format="wav",
        prompt_speech=b"reference-audio",
        prompt_text="Reference text",
        seed=7,
        use_memory_cache="on",
    )

    assert result == b"audio"
    assert captured["reference"] == {
        "audio": b"reference-audio",
        "text": "Reference text",
    }
    assert len(captured["request"]["references"]) == 1
    assert captured["request"]["references"][0].audio == b"reference-audio"
    assert captured["request"]["references"][0].text == "Reference text"
    assert captured["request"]["top_p"] == pytest.approx(0.8)
    assert captured["request"]["repetition_penalty"] == pytest.approx(1.1)
    assert captured["request"]["temperature"] == pytest.approx(0.8)
    assert captured["request"]["use_memory_cache"] == "on"
    assert captured["encoded"][0:2] == ("wav", 44100)
    np.testing.assert_allclose(
        captured["encoded"][2], np.array([[0.1, -0.1]], dtype=np.float32)
    )


def test_stream_ignores_upstream_header_and_final(monkeypatch):
    captured = {}
    segment = np.array([0.1, 0.2], dtype=np.float32)

    class FakeEngine:
        def inference(self, request):
            yield SimpleNamespace(code="header", audio=(44100, b"header"), error=None)
            yield SimpleNamespace(code="segment", audio=(44100, segment), error=None)
            yield SimpleNamespace(code="final", audio=(44100, segment), error=None)

    from .. import utils as audio_utils

    def audio_stream_generator(
        response_format, sample_rate, output_generator, output_chunk_transformer
    ):
        chunks = list(output_generator)
        captured["chunks"] = chunks
        captured["tensor"] = output_chunk_transformer(chunks[0]).numpy()
        yield b"stream"

    monkeypatch.setattr(audio_utils, "audio_stream_generator", audio_stream_generator)
    model = FishSpeechModel(
        "fish", "/models/fish", _model_spec(FISH_AUDIO_S2_PRO), device="cpu"
    )
    model._serve_reference_audio = _FakeSchema
    model._serve_tts_request = _FakeSchema
    model._engine = FakeEngine()
    model._model = SimpleNamespace(sample_rate=44100)

    assert list(model.speech("Hello", "", stream=True)) == [b"stream"]
    assert captured["chunks"] == [segment]
    np.testing.assert_allclose(
        captured["tensor"], np.array([[0.1], [0.2]], dtype=np.float32)
    )


def test_builtin_catalog_has_fish_audio_s1_and_s2_sources():
    models = {}
    load_model_family_from_json("model_spec.json", models)

    expected = {
        FISH_AUDIO_S1_MINI: "fishaudio/s1-mini",
        FISH_AUDIO_S2_PRO: "fishaudio/s2-pro",
    }
    for model_name, model_id in expected.items():
        specs = models[model_name]
        assert {
            spec.model_hub: (spec.model_id, spec.model_revision) for spec in specs
        } == {
            "huggingface": (model_id, "main"),
            "modelscope": (model_id, "master"),
        }
        assert all(
            set(spec.model_ability)
            == {
                "text2audio",
                "text2audio_zero_shot",
                "text2audio_voice_cloning",
            }
            for spec in specs
        )
        assert all("#system_torch#" in spec.virtualenv.packages for spec in specs)
        assert all("#system_torchcodec#" in spec.virtualenv.packages for spec in specs)
        # transformers already provides tqdm. Listing it explicitly makes the
        # virtualenv manager pin the host version, which may not exist on the
        # configured PyTorch wheel index.
        assert all("tqdm" not in spec.virtualenv.packages for spec in specs)


@pytest.mark.parametrize("model_name", [FISH_AUDIO_S1_MINI, FISH_AUDIO_S2_PRO])
def test_create_audio_model_instance_dispatches_modern_fish_audio(
    monkeypatch, model_name
):
    from .. import core

    spec = _model_spec(model_name)
    monkeypatch.setattr(core, "match_audio", lambda *args, **kwargs: spec)

    model = create_audio_model_instance("fish", model_name, model_path="/models/fish")

    assert isinstance(model, FishSpeechModel)


def test_fish_speech(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="FishSpeech-1.5", model_type="audio", compile=False
    )
    model = client.get_model(model_uid)

    input_string = "你好，你是谁？"
    response = model.speech(input_string)
    assert type(response) is bytes
    assert len(response) > 0

    # Test copy voice
    prompt_speech_path = os.path.join(os.path.dirname(__file__), "basic_ref_en.wav")
    with open(prompt_speech_path, "rb") as f:
        prompt_speech = f.read()
    response = model.speech(
        "Hello",
        prompt_speech=prompt_speech,
        prompt_text="Some call me nature, others call me mother nature.",
    )
    assert type(response) is bytes
    assert len(response) > 0

    # Test stream
    input_string = "瑞典王国，通称瑞典，是一个位于斯堪的纳维亚半岛的北欧国家，首都及最大城市为斯德哥尔摩。"
    response = model.speech(input_string, chunk_length=20, stream=True)
    assert inspect.isgenerator(response)
    i = 0
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
        for chunk in response:
            f.write(chunk)
            i += 1
            assert type(chunk) is bytes
            assert len(chunk) > 0
        assert i > 5

    # Test openai API
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    with client.audio.speech.with_streaming_response.create(
        model=model_uid, input=input_string, voice="echo", response_format="pcm"
    ) as response:
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=True) as f:
            response.stream_to_file(f.name)
            assert os.stat(f.name).st_size > 0
