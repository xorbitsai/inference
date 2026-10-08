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


def test_megatts(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="MegaTTS3",
        model_type="audio",
        compile=False,
        download_hub="huggingface",
    )
    model = client.get_model(model_uid)

    # Test copy voice
    prompt_speech_path = os.path.join(os.path.dirname(__file__), "bbc_news.wav")
    with open(prompt_speech_path, "rb") as f:
        prompt_speech = f.read()
    prompt_latent_path = os.path.join(os.path.dirname(__file__), "bbc_news.npy")
    with open(prompt_latent_path, "rb") as f:
        prompt_latent = f.read()
    response = model.speech(
        "His death in this conjuncture was a public misfortune.",
        prompt_speech=prompt_speech,
        prompt_latent=prompt_latent,
    )
    assert type(response) is bytes
    assert len(response) > 0


def _new_megatts_model():
    from types import SimpleNamespace

    from ..megatts import MegaTTSModel

    return MegaTTSModel(
        model_uid="megatts-test",
        model_path="unused",
        model_spec=SimpleNamespace(model_ability=["text2audio"]),
    )


def _wav_bytes(samples, sample_rate=16000):
    import io
    import struct
    import wave

    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        return buffer.getvalue()


def test_megatts_rejects_undecodable_prompt():
    import pytest

    pytest.importorskip("pydub")
    from ....core.exceptions import InvalidAudioInputError

    with pytest.raises(InvalidAudioInputError, match="failed to decode"):
        _new_megatts_model()._validate_prompt_speech(b"invalid")


def test_megatts_rejects_prompt_without_samples():
    import pytest

    pytest.importorskip("pydub")
    from ....core.exceptions import InvalidAudioInputError

    with pytest.raises(InvalidAudioInputError, match="contains no samples"):
        _new_megatts_model()._validate_prompt_speech(_wav_bytes([]))


def test_megatts_rejects_silent_prompt():
    import pytest

    pytest.importorskip("pydub")
    from ....core.exceptions import InvalidAudioInputError

    with pytest.raises(InvalidAudioInputError, match="no detectable audio"):
        _new_megatts_model()._validate_prompt_speech(_wav_bytes([0] * 1600))


def test_megatts_rejects_near_silent_prompt():
    import pytest

    pytest.importorskip("pydub")
    from ....core.exceptions import InvalidAudioInputError

    samples = [1 if index % 2 else -1 for index in range(1600)]
    with pytest.raises(InvalidAudioInputError, match="no detectable audio"):
        _new_megatts_model()._validate_prompt_speech(_wav_bytes(samples))


def test_megatts_accepts_audible_prompt():
    import pytest

    pytest.importorskip("pydub")
    samples = [1000 if index % 2 else -1000 for index in range(1600)]

    metrics = _new_megatts_model()._validate_prompt_speech(_wav_bytes(samples))

    assert metrics["duration_ms"] == 100
    assert metrics["sample_rate"] == 16000
    assert metrics["channels"] == 1
    assert metrics["frames"] == 1600


def test_megatts_rejects_empty_or_non_finite_output():
    import numpy as np
    import pytest

    model = _new_megatts_model()

    with pytest.raises(RuntimeError, match="empty audio"):
        model._validate_output_audio(None)
    with pytest.raises(RuntimeError, match="empty audio"):
        model._validate_output_audio(np.array([], dtype=np.float32))
    with pytest.raises(RuntimeError, match="non-finite"):
        model._validate_output_audio(np.array([0.0, np.nan], dtype=np.float32))
    with pytest.raises(RuntimeError, match="non-finite"):
        model._validate_output_audio(np.array([0.0, np.inf], dtype=np.float32))

    model._validate_output_audio(np.array([0.0, 0.1], dtype=np.float32))


def test_megatts_invalid_prompt_stops_before_inference(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import pytest

    from ....core.exceptions import InvalidAudioInputError

    model = _new_megatts_model()
    model._model = SimpleNamespace(preprocess=Mock(), forward=Mock())
    monkeypatch.setattr(
        model,
        "_validate_prompt_speech",
        Mock(side_effect=InvalidAudioInputError("invalid prompt")),
    )

    with pytest.raises(InvalidAudioInputError, match="invalid prompt"):
        model.speech("hello", voice="", prompt_speech=b"audio", prompt_latent=b"latent")

    model._model.preprocess.assert_not_called()
    model._model.forward.assert_not_called()


def test_megatts_requires_prompt_inputs():
    from types import SimpleNamespace
    from unittest.mock import Mock

    import pytest

    from ....core.exceptions import InvalidAudioInputError

    model = _new_megatts_model()
    model._model = SimpleNamespace(preprocess=Mock(), forward=Mock())

    with pytest.raises(InvalidAudioInputError, match="prompt_speech"):
        model.speech("hello", voice="", prompt_latent=b"latent")
    with pytest.raises(InvalidAudioInputError, match="prompt_latent"):
        model.speech("hello", voice="", prompt_speech=b"audio")

    model._model.preprocess.assert_not_called()
    model._model.forward.assert_not_called()


def test_megatts_speech_validates_generated_audio(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import numpy as np
    import pytest

    model = _new_megatts_model()
    monkeypatch.setattr(model, "_validate_prompt_speech", Mock())
    model._model = SimpleNamespace(
        preprocess=Mock(return_value="context"),
        forward=Mock(return_value=None),
    )

    with pytest.raises(RuntimeError, match="empty audio"):
        model.speech("hello", voice="", prompt_speech=b"audio", prompt_latent=b"latent")

    model._model.forward.return_value = np.array([0.0, np.nan], dtype=np.float32)
    with pytest.raises(RuntimeError, match="non-finite"):
        model.speech("hello", voice="", prompt_speech=b"audio", prompt_latent=b"latent")

    model._model.preprocess.assert_called()
    assert model._model.forward.call_count == 2


def test_megatts_speech_success(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from unittest.mock import Mock

    import numpy as np

    model = _new_megatts_model()
    validate_prompt = Mock()
    monkeypatch.setattr(model, "_validate_prompt_speech", validate_prompt)
    model._model = SimpleNamespace(
        sr=24000,
        preprocess=Mock(return_value="context"),
        forward=Mock(return_value=np.array([0.0, 0.1], dtype=np.float32)),
    )

    soundfile = ModuleType("soundfile")

    class SoundFile:
        def __init__(self, output, *args, **kwargs):
            self._output = output

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def write(self, _samples):
            self._output.write(b"encoded audio")

    soundfile.SoundFile = SoundFile
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)

    response = model.speech(
        "hello",
        voice="",
        response_format="wav",
        prompt_speech=b"audio",
        prompt_latent=b"latent",
    )

    assert response == b"encoded audio"
    validate_prompt.assert_called_once_with(b"audio")
    model._model.preprocess.assert_called_once()
    model._model.forward.assert_called_once_with(
        "context", "hello", time_step=32, p_w=1.6, t_w=2.5
    )
