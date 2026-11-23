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

import base64
import os
import secrets
import shutil
import socket
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryFile
from typing import TYPE_CHECKING, Any, BinaryIO, Dict, Generator, Optional, Tuple, Union

import httpx

from ...core.exceptions import InvalidModelRequestError
from .zonos3_mini import DEFAULT_VOICE, SAMPLE_RATE, Zonos3MiniModel

if TYPE_CHECKING:
    import numpy as np

    from .core import AudioModelFamilyV2

HOST = "127.0.0.1"
STARTUP_TIMEOUT = 180
BACKBONE = "backbone-ptq.gguf"
DECODER = "zac-keep2.gguf"
SPEAKER_ENCODER = "spk-encoder.gguf"
DEVICE_VISIBILITY_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
)


def _log_tail(log: BinaryIO) -> str:
    size = log.seek(0, os.SEEK_END)
    log.seek(max(0, size - 4096))
    return log.read().decode(errors="replace").strip()


class Zonos3MiniNativeModel(Zonos3MiniModel):
    """Run an upstream GGUF backend for the lifetime of the model actor."""

    supports_streaming = True
    server_executable: str
    server_path_env: str
    server_install_hint: str

    _process: Optional[subprocess.Popen] = None
    _client: Optional[httpx.Client] = None
    _server_log: Optional[BinaryIO] = None

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "AudioModelFamilyV2",
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_uid, model_path, model_spec, device=device, **kwargs)
        # Worker resolves executables before the model subpool applies launch envs.
        executable = self._find_server()
        self._server_path = os.path.abspath(executable) if executable else None
        self._server_env = dict(os.environ)

    @classmethod
    def _find_server(cls) -> Optional[str]:
        return shutil.which(
            os.environ.get(cls.server_path_env) or cls.server_executable
        )

    @classmethod
    def _missing_executable(cls) -> str:
        return (
            f"ZONOS3-Mini requires {cls.server_executable}. "
            f"{cls.server_install_hint} Add it to the worker PATH or set {cls.server_path_env}."
        )

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        if cls._find_server() is None:
            return False, cls._missing_executable()
        return True

    @property
    def _gguf_path(self) -> Path:
        return Path(self._model_path).absolute() / "gguf"

    def _command(self, port: int) -> list[str]:
        if self._server_path is None:
            raise ImportError(self._missing_executable())

        return [
            self._server_path,
            str(self._gguf_path / BACKBONE),
            "--zac",
            str(self._gguf_path / DECODER),
            "--spk-encoder",
            str(self._gguf_path / SPEAKER_ENCODER),
            "--tts-default-voices-dir",
            "",
            "--host",
            HOST,
            "--port",
            str(port),
        ]

    def load(self) -> None:
        if self._process is not None:
            return
        for name in (BACKBONE, DECODER, SPEAKER_ENCODER):
            if not (self._gguf_path / name).is_file():
                raise ValueError(f"Missing ZONOS3-Mini model file gguf/{name}.")

        # Use snapshot names, including on Hugging Face caches where the
        # embedding files are symlinks to content-addressed blobs.
        self._voice_embeddings = {
            path.name.split("_", 1)[0]: path
            for path in sorted((self._gguf_path / "voices").rglob("*.embedding.npy"))
        }
        if DEFAULT_VOICE not in self._voice_embeddings:
            raise ValueError(
                f"Missing ZONOS3-Mini {DEFAULT_VOICE} embedding in gguf/voices."
            )

        with socket.socket() as port_socket:
            port_socket.bind((HOST, 0))
            port = port_socket.getsockname()[1]

        command = self._command(port)
        token = secrets.token_urlsafe(32)
        # The runtime can also execute ffmpeg. Keep its binary and library
        # search paths from the worker, while honoring the model's devices.
        env = {
            **self._server_env,
            **{
                key: os.environ[key]
                for key in DEVICE_VISIBILITY_KEYS
                if key in os.environ
            },
            "XINFERENCE_MODEL_UID": self._model_uid,
            "ZONOS3_ACCESS_TOKEN": token,
        }
        try:
            self._server_log = TemporaryFile()
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=self._server_log,
                stderr=subprocess.STDOUT,
                env=env,
            )
            self._client = httpx.Client(
                base_url=f"http://{HOST}:{port}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=httpx.Timeout(300, connect=5),
                trust_env=False,
            )
            deadline = time.monotonic() + STARTUP_TIMEOUT
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError(
                        f"{self.server_executable} exited during startup "
                        f"(code {self._process.returncode}). {_log_tail(self._server_log)}"
                    )
                try:
                    response = self._client.get("/v2/health", timeout=1)
                    if response.is_success and response.json() == {"status": "ok"}:
                        return
                except httpx.TransportError:
                    pass
                time.sleep(0.1)
            raise RuntimeError(
                f"{self.server_executable} did not become ready within "
                f"{STARTUP_TIMEOUT} seconds. {_log_tail(self._server_log)}"
            )
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        process, self._process = self._process, None
        client, self._client = self._client, None
        log, self._server_log = self._server_log, None
        try:
            if client is not None:
                client.close()
        finally:
            try:
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            finally:
                if log is not None:
                    log.close()

    def _audio_chunks(
        self,
        text: str,
        voice: str,
        reference_path: Optional[Path],
        language: Optional[str],
        options: Dict[str, Any],
        stream: bool,
    ) -> Generator["np.ndarray", None, None]:
        import numpy as np

        client = self._client
        if client is None:
            raise RuntimeError("ZONOS3-Mini is not loaded.")

        option_names = {"top_k": "topk", "max_new_tokens": "max_tokens"}
        body = {option_names.get(key, key): value for key, value in options.items()}
        body.update(text=text, format="pcm", stream=stream, speaker_mode="embedding")
        if stream:
            body["stream_decode"] = "prefix"
        if language is not None:
            body["language"] = language
        if reference_path is not None:
            body["speaker_audio_base64"] = base64.b64encode(
                reference_path.read_bytes()
            ).decode("ascii")
        else:
            embedding = self._voice_embeddings.get(voice)
            if embedding is None:
                raise InvalidModelRequestError(
                    f"ZONOS3-Mini voice {voice!r} is absent from the local snapshot."
                )
            body["speaker_embedding_base64"] = base64.b64encode(
                embedding.read_bytes()
            ).decode("ascii")
            body["speaker_embedding_name"] = embedding.name

        with client.stream("POST", "/v2/generate", json=body) as response:
            if response.status_code in {400, 413, 422}:
                response.read()
                raise InvalidModelRequestError(
                    f"{self.server_executable} rejected the speech request: {response.text}"
                )
            response.raise_for_status()
            if (
                response.headers.get("X-Audio-Format") != "float32"
                or response.headers.get("X-Audio-Channels") != "1"
                or response.headers.get("X-Audio-Sample-Rate") != str(SAMPLE_RATE)
            ):
                raise RuntimeError(
                    f"{self.server_executable} returned unexpected audio format."
                )
            pending = b""
            for chunk in response.iter_bytes():
                chunk = pending + chunk
                size = len(chunk) // 4 * 4
                if size:
                    yield np.frombuffer(chunk[:size], dtype="<f4")
                pending = chunk[size:]
            if pending:
                raise RuntimeError(
                    f"{self.server_executable} returned an incomplete PCM sample."
                )

    def _generate(
        self,
        text: str,
        voice: str,
        reference_path: Optional[Path],
        language: Optional[str],
        options: Dict[str, Any],
    ) -> "np.ndarray":
        import numpy as np

        chunks = list(
            self._audio_chunks(text, voice, reference_path, language, options, False)
        )
        if not chunks:
            raise RuntimeError(f"{self.server_executable} returned empty audio.")
        return np.concatenate(chunks)

    def _generate_stream(
        self,
        text: str,
        voice: str,
        reference_path: Optional[Path],
        language: Optional[str],
        options: Dict[str, Any],
    ) -> Generator["np.ndarray", None, None]:
        return self._audio_chunks(text, voice, reference_path, language, options, True)
