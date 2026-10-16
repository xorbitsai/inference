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
from types import ModuleType

import pytest
import torch

from .. import utils as audio_utils


def test_audio_stream_generator_reports_missing_torchcodec(monkeypatch):
    torchaudio = ModuleType("torchaudio")
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setitem(sys.modules, "torchcodec.encoders", None)

    with pytest.raises(ImportError, match="Failed to import 'torchcodec'"):
        list(audio_utils.audio_stream_generator("mp3", 24000, iter(()), lambda x: x))


def test_audio_stream_generator_uses_torchcodec_and_flushes_tail(monkeypatch):
    captured = {}

    class FakeAudioStream:
        def __init__(self, encoder):
            self._encoder = encoder

        def add_samples(self, samples):
            captured.setdefault("samples", []).append(samples.clone())
            self._encoder.destination.write(b"chunk")

    class FakeEncoder:
        def add_audio(self, *, sample_rate, num_channels):
            captured["audio_config"] = (sample_rate, num_channels)
            return FakeAudioStream(self)

        def open_file_like(self, destination, *, format):
            self.destination = destination
            captured["format"] = format
            return self

        def __enter__(self):
            self.destination.write(b"header-")
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if captured["format"] == "mp4":
                self.destination.seek(0)
                self.destination.write(b"fixed--")
                self.destination.seek(0, 2)
            self.destination.write(b"tail")

    torchaudio = ModuleType("torchaudio")
    torchcodec = ModuleType("torchcodec")
    torchcodec.__path__ = []
    encoders = ModuleType("torchcodec.encoders")
    encoders.Encoder = FakeEncoder
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    monkeypatch.setitem(sys.modules, "torchcodec", torchcodec)
    monkeypatch.setitem(sys.modules, "torchcodec.encoders", encoders)

    chunks = [torch.ones((4, 1), dtype=torch.float64), torch.ones(3)]
    format_pairs = (
        ("mp3", "mp3"),
        ("AAC", "adts"),
        ("m4a", "mp4"),
        ("wave", "wav"),
    )
    for response_format, container_format in format_pairs:
        captured.clear()
        result = list(
            audio_utils.audio_stream_generator(
                response_format, 24000, chunks, lambda chunk: chunk
            )
        )

        if container_format == "mp4":
            assert result == [b"fixed--chunkchunktail"]
        else:
            assert result == [b"header-chunk", b"chunk", b"tail"]
        assert captured["audio_config"] == (24000, 1)
        assert captured["format"] == container_format
        assert [sample.shape for sample in captured["samples"]] == [(1, 4), (1, 3)]
        assert all(sample.dtype == torch.float32 for sample in captured["samples"])
        assert all(sample.device.type == "cpu" for sample in captured["samples"])


def test_audio_stream_generator_uses_legacy_torchcodec_audio_encoder(monkeypatch):
    captured = {}

    class FakeAudioEncoder:
        def __init__(self, samples, *, sample_rate):
            captured["samples"] = samples.clone()
            captured["sample_rate"] = sample_rate

        def to_tensor(self, *, format):
            captured["format"] = format
            return torch.tensor(list(b"encoded"), dtype=torch.uint8)

    torchaudio = ModuleType("torchaudio")
    torchcodec = ModuleType("torchcodec")
    torchcodec.__path__ = []
    encoders = ModuleType("torchcodec.encoders")
    encoders.AudioEncoder = FakeAudioEncoder
    torchcodec.encoders = encoders
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    monkeypatch.setitem(sys.modules, "torchcodec", torchcodec)
    monkeypatch.setitem(sys.modules, "torchcodec.encoders", encoders)

    chunks = [torch.ones((4, 1), dtype=torch.float64), torch.ones(3)]
    result = list(
        audio_utils.audio_stream_generator("mp3", 24000, chunks, lambda chunk: chunk)
    )

    assert result == [b"encoded"]
    assert captured["sample_rate"] == 24000
    assert captured["format"] == "mp3"
    assert captured["samples"].shape == (1, 7)
    assert captured["samples"].dtype == torch.float32
    assert captured["samples"].device.type == "cpu"


def test_audio_stream_generator_legacy_path_with_real_audio_encoder(monkeypatch):
    torchcodec_encoders = pytest.importorskip("torchcodec.encoders")
    torchcodec_decoders = pytest.importorskip("torchcodec.decoders")
    torchaudio = pytest.importorskip("torchaudio")
    if not hasattr(torchcodec_encoders, "AudioEncoder"):
        pytest.skip("TorchCodec AudioEncoder is unavailable")

    # Exercise the public API exposed by TorchCodec 0.9/0.10 even when the
    # installed test version also provides the newer incremental Encoder.
    monkeypatch.delattr(torchcodec_encoders, "Encoder", raising=False)
    monkeypatch.delattr(torchaudio, "io", raising=False)

    sample_rate = 24000
    chunks = [
        torch.linspace(-0.25, 0.25, 1200).reshape(-1, 1),
        torch.linspace(0.25, -0.25, 1200).reshape(-1, 1),
    ]
    encoded = b"".join(
        audio_utils.audio_stream_generator(
            "wav", sample_rate, chunks, lambda chunk: chunk
        )
    )
    decoded = torchcodec_decoders.AudioDecoder(encoded).get_all_samples()

    assert decoded.sample_rate == sample_rate
    assert decoded.data.shape == (1, 2400)
