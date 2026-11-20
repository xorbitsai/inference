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

import base64
import json
import sys
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from ....core.exceptions import InvalidModelRequestError
from ..zonos3_mini_ggml import GGMLZonos3MiniModel

AUDIO_HEADERS = {
    "X-Audio-Format": "float32",
    "X-Audio-Channels": "1",
    "X-Audio-Sample-Rate": "44100",
}


@pytest.fixture
def gguf_snapshot(tmp_path):
    gguf = tmp_path / "gguf"
    voice = gguf / "voices" / "English" / "Savannah_en-American_F_1.embedding.npy"
    voice.parent.mkdir(parents=True)
    voice.write_bytes(b"embedding")
    for name in ("backbone-ptq.gguf", "zac-keep2.gguf", "spk-encoder.gguf"):
        (gguf / name).touch()
    return voice


def test_ggml_native_command(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv(GGMLZonos3MiniModel.server_path_env, sys.executable)
    model = GGMLZonos3MiniModel("zonos", str(tmp_path), SimpleNamespace())
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv(GGMLZonos3MiniModel.server_path_env, str(tmp_path / "other"))
    assert model._command(12345) == [
        sys.executable,
        str(tmp_path / "gguf" / "backbone-ptq.gguf"),
        "--zac",
        str(tmp_path / "gguf" / "zac-keep2.gguf"),
        "--spk-encoder",
        str(tmp_path / "gguf" / "spk-encoder.gguf"),
        "--tts-default-voices-dir",
        "",
        "--host",
        "127.0.0.1",
        "--port",
        "12345",
    ]


def test_ggml_cpu_device_overrides_worker_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv(GGMLZonos3MiniModel.server_path_env, sys.executable)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    cpu = GGMLZonos3MiniModel("zonos", str(tmp_path), SimpleNamespace(), device="cpu")
    default = GGMLZonos3MiniModel("zonos", str(tmp_path), SimpleNamespace())
    assert cpu._command(12345)[-1] == "--cpu"
    assert default._command(12345)[-1] == "--gpu"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert default._command(12345)[-1] == "--cpu"


def test_native_shutdown_and_startup_failure(tmp_path, gguf_snapshot, monkeypatch):
    monkeypatch.setenv("ZONOS3_FFMPEG", "/trusted/ffmpeg")
    monkeypatch.setenv("PATH", "/trusted/bin")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9")
    model = GGMLZonos3MiniModel("zonos", str(tmp_path), SimpleNamespace())
    monkeypatch.setenv("ZONOS3_FFMPEG", "/other/ffmpeg")
    monkeypatch.setenv("PATH", "/other/bin")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    server = """
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        assert self.path == '/v2/health'
        assert os.environ['PATH'] == '/trusted/bin'
        assert os.environ['ZONOS3_FFMPEG'] == '/trusted/ffmpeg'
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
        assert self.headers['Authorization'] == 'Bearer ' + os.environ['ZONOS3_ACCESS_TOKEN']
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setattr(
        model, "_command", lambda port: [sys.executable, "-c", server, str(port)]
    )
    model.load()
    process = model._process
    try:
        assert process is not None
        assert process.poll() is None
        assert model._voice_embeddings["Savannah"] == gguf_snapshot
    finally:
        model.stop()
    assert process.poll() is not None
    model.stop()

    monkeypatch.setattr(
        model, "_command", lambda _port: [sys.executable, "-c", "raise SystemExit(7)"]
    )
    with pytest.raises(RuntimeError, match=r"exited during startup \(code 7\)"):
        model.load()
    assert model._process is None
    assert model._client is None
    assert model._server_log is None


class SplitPCM(httpx.SyncByteStream):
    def __init__(self):
        self.closed = False
        self.raw = np.array([0.25, -0.5], dtype="<f4").tobytes()

    def __iter__(self):
        yield self.raw[:3]
        yield self.raw[3:5]
        yield self.raw[5:]

    def close(self):
        self.closed = True


def test_native_speech_maps_voice_options_and_split_samples(tmp_path, gguf_snapshot):
    seen = []
    stream = SplitPCM()

    def handle(request):
        assert request.url.path == "/v2/generate"
        seen.append(json.loads(request.content))
        return httpx.Response(200, headers=AUDIO_HEADERS, stream=stream)

    model = GGMLZonos3MiniModel("zonos", str(tmp_path), SimpleNamespace())
    with httpx.Client(
        base_url="http://127.0.0.1", transport=httpx.MockTransport(handle)
    ) as model._client:
        model._voice_embeddings = {"Savannah": gguf_snapshot}
        generator = model.speech(
            "Hello",
            "default",
            response_format="pcm",
            language="en",
            stream=True,
            seed=7,
            top_k=30,
            max_new_tokens=18,
        )
        assert next(generator) == np.array([8191], dtype="<i2").tobytes()
        generator.close()
        assert stream.closed

    assert seen == [
        {
            "text": "Hello",
            "language": "en_us",
            "format": "pcm",
            "speaker_mode": "embedding",
            "speaker_embedding_base64": base64.b64encode(b"embedding").decode(),
            "speaker_embedding_name": gguf_snapshot.name,
            "stream": True,
            "stream_decode": "prefix",
            "seed": 7,
            "topk": 30,
            "max_tokens": 18,
        }
    ]


def test_native_cloning_http_validation(tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"wave")

    def handle(request):
        body = json.loads(request.content)
        assert body["speaker_audio_base64"] == base64.b64encode(b"wave").decode()
        assert "speaker_embedding_base64" not in body
        assert not body["stream"]
        return httpx.Response(400, json={"detail": "Unsupported reference."})

    model = GGMLZonos3MiniModel("zonos", str(tmp_path), SimpleNamespace())
    with httpx.Client(
        base_url="http://127.0.0.1", transport=httpx.MockTransport(handle)
    ) as model._client:
        with pytest.raises(InvalidModelRequestError, match="Unsupported reference"):
            model._generate("Hello", "Savannah", reference, None, {})
