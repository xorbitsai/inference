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

from types import SimpleNamespace

import pytest

from .. import engine
from ..zonos3_mini_mlx import MLXZonos3MiniModel


@pytest.mark.parametrize(
    ("system", "architecture", "matches"),
    [("Darwin", "arm64", True), ("Darwin", "x86_64", False), ("Linux", "arm64", False)],
)
def test_zonos3_mini_mlx_requires_apple_silicon(
    monkeypatch, system, architecture, matches
):
    monkeypatch.setattr(engine.platform, "system", lambda: system)
    monkeypatch.setattr(engine.platform, "machine", lambda: architecture)

    result = engine.MLXZonos3MiniAudioModel.check_host()
    if matches:
        assert result is True
    else:
        available, message = result
        assert available is False
        assert "Apple Silicon" in message


def test_zonos3_mini_mlx_uses_configured_worker_executable(monkeypatch, tmp_path):
    server = tmp_path / "Zonos MLX" / "zonos3-mlx-server"
    server.parent.mkdir()
    server.touch()
    monkeypatch.setenv("XINFERENCE_ZONOS3_MLX_SERVER_PATH", str(server))

    available, message = MLXZonos3MiniModel.check_lib()
    assert available is False
    assert "uv sync --locked --package zonos3-mlx" in message

    server.chmod(0o755)
    assert MLXZonos3MiniModel.check_lib() is True
    model = MLXZonos3MiniModel("zonos", str(tmp_path), SimpleNamespace())
    assert model._command(port=1919) == [
        str(server),
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
        "1919",
    ]
