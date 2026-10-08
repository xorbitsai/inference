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
import re

import numpy as np
import pytest

from ..minimax_music3 import MiniMaxMusic3Model


@pytest.mark.parametrize(
    ("response_format", "expected"),
    [
        (None, "wav"),
        ("", "wav"),
        ("WAV", "wav"),
        ("mp3", "mp3"),
        ("flac", "flac"),
        ("ogg", "ogg"),
    ],
)
def test_validate_speech_request_normalizes_response_format(response_format, expected):
    actual = MiniMaxMusic3Model._validate_speech_request(
        "lyrics",
        "warm piano",
        "default",
        response_format,
        1.0,
        False,
        0,
        60,
        {},
    )

    assert actual == expected


def test_validate_speech_request_rejects_unsupported_response_format():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "MiniMax-Music3 supports these response formats: flac, mp3, ogg, wav."
        ),
    ):
        MiniMaxMusic3Model._validate_speech_request(
            "lyrics",
            "warm piano",
            "default",
            "aac",
            1.0,
            False,
            0,
            60,
            {},
        )


def test_audio_to_bytes_preserves_native_wav_output():
    audio = np.zeros((2, 32), dtype=np.float32)

    encoded = MiniMaxMusic3Model._audio_to_bytes(audio, 44100, "wav")

    assert encoded.startswith(b"RIFF")
    assert encoded[8:12] == b"WAVE"


@pytest.mark.parametrize(
    ("response_format", "expected_container"),
    [("flac", "FLAC"), ("mp3", "MP3"), ("ogg", "OGG")],
)
def test_audio_to_bytes_encodes_requested_format(response_format, expected_container):
    soundfile = pytest.importorskip("soundfile", minversion="0.13.1")
    audio = np.zeros((2, 4410), dtype=np.float32)

    encoded = MiniMaxMusic3Model._audio_to_bytes(audio, 44100, response_format)
    info = soundfile.info(io.BytesIO(encoded))

    assert info.format == expected_container
    assert info.samplerate == 44100
    assert info.channels == 2
