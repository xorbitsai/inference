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

from copy import deepcopy

import pytest

from ..utils import ChatModelMixin

AUDIO_URL = "https://example.test/audio.wav"


@pytest.mark.parametrize(
    "audio_url", [AUDIO_URL, {"url": AUDIO_URL}], ids=["str", "object"]
)
def test_transform_messages_audio_url(audio_url):
    messages = [
        {"role": "system", "content": "Listen to the audio."},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_url},
                {"type": "text", "text": "What is that sound?"},
            ],
        },
    ]
    original = deepcopy(messages)

    transformed = ChatModelMixin()._transform_messages(messages)

    assert transformed == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Listen to the audio."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": AUDIO_URL},
                {"type": "text", "text": "What is that sound?"},
            ],
        },
    ]
    assert messages == original


@pytest.mark.parametrize(
    "audio_url, error",
    [
        (None, TypeError),
        (42, TypeError),
        ([], TypeError),
        ({}, KeyError),
        ({"uri": AUDIO_URL}, KeyError),
    ],
    ids=["null", "number", "list", "missing-url", "wrong-key"],
)
def test_transform_messages_rejects_invalid_audio_url(audio_url, error):
    messages = [
        {"role": "user", "content": [{"type": "audio", "audio_url": audio_url}]}
    ]

    with pytest.raises(error):
        ChatModelMixin()._transform_messages(messages)
