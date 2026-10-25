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

import os.path
import tempfile
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    "launch_kwargs", [{}, {"max_new_tokens": None}, {"max_new_tokens": 440}]
)
@pytest.mark.parametrize(
    "response_format, granularities, timestamps",
    [
        ("json", None, None),
        ("verbose_json", ["segment"], True),
        ("verbose_json", ["word"], "word"),
    ],
)
def test_whisper_generation_kwargs(
    launch_kwargs, response_format, granularities, timestamps
):
    from ..whisper import WhisperModel

    model = WhisperModel(
        "test", "/unused", SimpleNamespace(multilingual=True), **launch_kwargs
    )
    chunk = {"text": "hello", "timestamp": (0, 1)}
    model._model = Mock(return_value={"text": "hello", "chunks": [chunk]})
    for method, task in [
        (model.transcriptions, "transcribe"),
        (model.translations, "translate"),
    ]:
        result = method(
            b"audio",
            language="en",
            response_format=response_format,
            timestamp_granularities=granularities,
        )
        expected = {"task": task, "language": "en"}
        if task == "transcribe" and launch_kwargs.get("max_new_tokens") is not None:
            expected["max_new_tokens"] = launch_kwargs["max_new_tokens"]
        pipeline_kwargs = {"generate_kwargs": expected}
        if timestamps is not None:
            pipeline_kwargs["return_timestamps"] = timestamps
        model._model.assert_called_with(b"audio", **pipeline_kwargs)
        assert result["text"] == "hello"
        if timestamps is not None:
            assert result["words" if timestamps == "word" else "segments"]


@pytest.mark.parametrize(
    "checkpoint_limit, eos_after", [(None, None), (24, None), (None, 0), (None, 5)]
)
def test_whisper_uses_checkpoint_generation_budget(checkpoint_limit, eos_after):
    import torch
    import transformers

    from ..whisper import WhisperModel

    # Tiny local model: the configured maximum includes the decoder prefix.
    config = transformers.WhisperConfig(
        vocab_size=8,
        d_model=4,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=1,
        decoder_attention_heads=1,
        encoder_ffn_dim=8,
        decoder_ffn_dim=8,
        max_source_positions=2,
        max_target_positions=160,
        pad_token_id=0,
        bos_token_id=1,
        decoder_start_token_id=1,
        eos_token_id=3,
        begin_suppress_tokens=[],
    )
    backend = transformers.WhisperForConditionalGeneration(config).eval()
    backend.generation_config.task_to_id = {"transcribe": 2}
    backend.generation_config.max_new_tokens = checkpoint_limit
    backend.generation_config.max_length = config.max_target_positions

    class Continuation(transformers.LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores.fill_(-float("inf"))
            at_eos = eos_after is not None and input_ids.shape[-1] >= eos_after + 2
            scores[:, 3 if at_eos else 4] = 0
            return scores

    def pipeline(audio, generate_kwargs):
        tokens = backend.generate(
            torch.zeros(1, config.num_mel_bins, 4),
            logits_processor=[Continuation()],
            **generate_kwargs,
        )
        return {"text": "a" * (tokens == 4).sum().item()}

    model = WhisperModel("test", "/unused", SimpleNamespace())
    model._model = pipeline
    result = model.transcriptions(b"audio")
    assert len(result["text"]) == (
        eos_after if eos_after is not None else checkpoint_limit or 158
    )
    assert backend.generation_config.max_new_tokens == checkpoint_limit


def test_restful_api_for_whisper(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="whisper-1",
        model_name="whisper-small",
        model_type="audio",
    )
    model = client.get_model(model_uid)
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")
    with open(audio_path, "rb") as f:
        audio = f.read()

    response = model.transcriptions(audio)
    transcription = response["text"].lower()
    assert "my fellow americans" in transcription
    assert "your country" in transcription
    assert "do for you" in transcription

    # Translation requires large-v3 model.
    zh_cn_audio_path = os.path.join(
        os.path.dirname(__file__), "common_voice_zh-CN_38026095.mp3"
    )
    with open(zh_cn_audio_path, "rb") as f:
        zh_cn_audio = f.read()
    response = model.translations(zh_cn_audio)
    translation = response["text"].lower()
    assert "airlines" in translation
    assert "hong kong" in translation

    # If model multilingual is False, it can't be used for translations.
    model_uid2 = client.launch_model(
        model_uid="whisper-2",
        model_name="whisper-tiny.en",
        model_type="audio",
        n_gpu=None,
    )
    model2 = client.get_model(model_uid2)
    with pytest.raises(RuntimeError, match="translations"):
        model2.translations(zh_cn_audio)

    # Test openai API
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    with open(zh_cn_audio_path, "rb") as f:
        completion = client.audio.transcriptions.create(model=model_uid, file=f)
        assert "列表" in completion.text
        assert "香港" in completion.text
        assert "航空" in completion.text

        completion = client.audio.translations.create(model=model_uid, file=f)
        translation = completion.text.lower()
        assert "airlines" in translation
        assert "hong kong" in translation


def test_transcriptions_for_whisper(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_uid="whisper-1",
        model_name="whisper-small",
        model_type="audio",
    )
    model = client.get_model(model_uid)
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")
    with open(audio_path, "rb") as f:
        audio = f.read()

    response = model.transcriptions(audio, response_format="verbose_json")
    assert response["text"]
    assert len(response["segments"]) == 1

    seek_set = set()
    for s in response["segments"]:
        if s["seek"] in seek_set:
            assert False, "incorrect seek"
        seek_set.add(s["seek"])

    response = model.transcriptions(
        audio, response_format="verbose_json", timestamp_granularities=["word"]
    )
    assert response["text"]
    assert len(response["words"]) == 22

    zh_cn_audio_path = os.path.join(
        os.path.dirname(__file__), "common_voice_zh-CN_38026095.mp3"
    )

    # Test openai API
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    with open(zh_cn_audio_path, "rb") as f:
        completion = client.audio.transcriptions.create(
            model=model_uid,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
        assert len(completion.segments) == 1

        completion = client.audio.transcriptions.create(
            model=model_uid,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
        # Sometimes it splits the whole sentence into one word.
        assert len(completion.words) > 0


def test_register_custom_audio():
    from ..custom import (
        CustomAudioModelFamilyV2,
        get_user_defined_audios,
        register_audio,
        unregister_audio,
    )

    # correct
    family_a = CustomAudioModelFamilyV2(
        model_family="my-whisper",
        model_name=f"custom_test_a-{uuid.uuid4().hex[:8]}",
        model_id="test/custom_test_a",
        multilingual=True,
        model_ability=["audio2text"],
    )

    register_audio(family_a, False)
    assert family_a in get_user_defined_audios()

    # name conflict
    family_b = CustomAudioModelFamilyV2(
        model_family="my-whisper",
        model_name=f"custom_test_b-{uuid.uuid4().hex[:8]}",
        model_id="test/custom_test_b",
        multilingual=True,
        model_ability=["audio2text"],
    )
    register_audio(family_b, False)
    assert family_b in get_user_defined_audios()
    with pytest.raises(ValueError):
        register_audio(family_b, False)

    # unregister
    unregister_audio(family_a.model_name)
    assert family_a not in get_user_defined_audios()
    unregister_audio(family_b.model_name)
    assert family_b not in get_user_defined_audios()


def test_persistent_custom_audio():
    from ....constants import XINFERENCE_MODEL_DIR
    from ..custom import (
        CustomAudioModelFamilyV2,
        get_user_defined_audios,
        register_audio,
        unregister_audio,
    )

    temp_dir = tempfile.mkdtemp()

    # correct
    family = CustomAudioModelFamilyV2(
        model_family="my-whisper",
        model_name=f"custom_test_a-{uuid.uuid4().hex[:8]}",
        model_id="test/custom_test_a",
        multilingual=True,
        model_uri=os.path.abspath(temp_dir),
        model_ability=["audio2text"],
    )

    register_audio(family, True)
    assert family in get_user_defined_audios()
    assert f"{family.model_name}.json" in os.listdir(
        os.path.join(XINFERENCE_MODEL_DIR, "v2", "audio")
    )

    unregister_audio(family.model_name)
    assert f"{family.model_name}.json" not in os.listdir(
        os.path.join(XINFERENCE_MODEL_DIR, "v2", "audio")
    )
