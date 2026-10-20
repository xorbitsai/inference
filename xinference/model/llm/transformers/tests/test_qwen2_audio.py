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
import sys
import types
from io import BytesIO

import numpy as np
import pytest

from ..multimodal import qwen2_audio

AUDIO_TOKENS = ["<|AUDIO|>", "<|audio_bos|>", "<|audio_eos|>"]


def _model_with_real_processor(monkeypatch, audio):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2AudioProcessor,
        WhisperFeatureExtractor,
    )

    vocab = {token: i for i, token in enumerate(["[UNK]", "[PAD]", *AUDIO_TOKENS])}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=AUDIO_TOKENS,
    )
    processor = Qwen2AudioProcessor(WhisperFeatureExtractor(), tokenizer)
    monkeypatch.setattr(qwen2_audio, "urlopen", lambda url: BytesIO(b"audio"))
    # The adapter imports librosa only to decode the audio; a stub avoids requiring librosa.
    librosa = types.ModuleType("librosa")
    librosa.load = lambda *args, **kwargs: (audio, 16000)
    monkeypatch.setitem(sys.modules, "librosa", librosa)
    model = object.__new__(qwen2_audio.Qwen2AudioChatModel)
    model._processor = processor
    model._device = "cpu"
    return model, tokenizer


def test_build_inputs_preserves_audio(monkeypatch):
    model, tokenizer = _model_with_real_processor(
        monkeypatch, np.zeros(16000, dtype=np.float32)
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "audio", "audio_url": {"url": "https://test/audio"}}],
        }
    ]

    inputs = model.build_inputs_from_messages(messages, {})

    assert "input_features" in inputs
    audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
    assert (inputs.input_ids == audio_token_id).sum().item() > 1


@pytest.mark.parametrize(
    "content", ["hello", [{"type": "text", "text": "hello"}]], ids=["str", "parts"]
)
def test_build_inputs_text_only(monkeypatch, content):
    model, _ = _model_with_real_processor(monkeypatch, None)

    inputs = model.build_inputs_from_messages(
        [{"role": "user", "content": content}], {}
    )

    assert "input_features" not in inputs
    assert inputs.input_ids.shape[0] == 1
