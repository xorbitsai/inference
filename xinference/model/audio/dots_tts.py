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
import logging
import math
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ...core.exceptions import InvalidModelRequestError

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


def is_dots_tts_python_supported(
    python_executable: Optional[str] = None,
) -> bool:
    version = sys.version_info[:2]
    if python_executable is not None:
        try:
            result = subprocess.run(
                [
                    python_executable,
                    "-I",
                    "-S",
                    "-c",
                    "import sys; print(sys.version_info[0], sys.version_info[1])",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            major, minor = map(int, result.stdout.split())
            version = (major, minor)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise ValueError(
                "Cannot determine dots.tts.edit model Python version from "
                f"{python_executable!r}."
            ) from exc
    return (3, 10) <= version < (3, 13)


def _load_dots_tts_runtime():
    from ...thirdparty.dots_tts.edit_runtime import DotsTtsEditRuntime

    return DotsTtsEditRuntime


class DotsTTSEditModel:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "AudioModelFamilyV2",
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._device = device
        self._kwargs = kwargs
        self._runtime: Any = None
        self._generation_lock: Any = None
        self._stream_executor: Any = None

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def load(self) -> None:
        if not is_dots_tts_python_supported():
            raise RuntimeError("dots.tts.edit requires Python 3.10 through 3.12.")

        import torch

        try:
            DotsTtsEditRuntime = _load_dots_tts_runtime()
        except ImportError as exc:
            raise ImportError(
                "The bundled dots.tts.edit runtime requires the dependencies "
                "declared in its model specification. Enable the model "
                "virtual environment."
            ) from exc

        device = torch.device(
            self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("dots.tts.edit supports CPU or CUDA devices.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("dots.tts.edit requires an available CUDA device.")
        # The upstream runtime selects CUDA whenever it is visible.
        if device.type == "cpu" and torch.cuda.is_available():
            raise ValueError(
                "dots.tts.edit CPU inference requires CUDA to be hidden from "
                "the model worker."
            )

        options = (self._model_spec.default_model_config or {}).copy()
        options.update(self._kwargs)
        options.pop("quantization", None)
        options.setdefault(
            "precision", "bfloat16" if device.type == "cuda" else "float32"
        )
        allowed = {
            "precision",
            "optimize",
            "max_generate_length",
            "max_sequence_length",
            "vocoder_merge_steps",
            "warmup_on_optimize",
        }
        if options.keys() - allowed:
            raise ValueError(
                "Unsupported dots.tts.edit load option(s): "
                + ", ".join(sorted(options.keys() - allowed))
            )
        if device.type == "cpu" and options["precision"] not in {"float32", "fp32"}:
            raise ValueError(
                "dots.tts.edit CPU inference requires precision='float32'."
            )
        if not Path(self._model_path).is_dir():
            raise FileNotFoundError(f"Model directory not found: {self._model_path}")
        with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
            self._runtime = DotsTtsEditRuntime.from_pretrained(
                self._model_path, **options
            )
            if device.type == "cuda":
                device = torch.device("cuda", torch.cuda.current_device())
        self._device = str(device)
        # The unloaded model crosses the actor boundary before load() runs.
        self._generation_lock = threading.Lock()
        self._stream_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dots-tts-stream"
        )

    def stop(self) -> None:
        if self._stream_executor is not None:
            self._stream_executor.shutdown(wait=True)
            self._stream_executor = None

    @contextmanager
    def _generation_context(self, seed: Optional[int]):
        import torch

        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeError("dots.tts.edit already has an active generation.")
        try:
            device = torch.device(self._device)
            devices = [device.index] if device.type == "cuda" else []
            with (
                torch.random.fork_rng(devices=devices)
                if seed is not None
                else nullcontext()
            ):
                if seed is not None:
                    torch.random.default_generator.manual_seed(seed)
                    if devices:
                        with torch.cuda.device(device):
                            torch.cuda.manual_seed(seed)
                yield
        finally:
            self._generation_lock.release()

    @contextmanager
    def _inference_context(self):
        import torch

        device = torch.device(self._device)
        # Streaming keeps this context on its dedicated inference thread.
        with (
            torch.cuda.device(device) if device.type == "cuda" else nullcontext(),
            torch.inference_mode(),
        ):
            yield

    def _stream_chunks(self, options: dict):
        def generate():
            with self._inference_context():
                yield from self._runtime.generate_stream(**options)

        chunks = generate()
        exhausted = object()

        # Upstream keeps autocast contexts open across yields. Resume and close
        # on one dedicated thread even when ModelActor changes executor threads.
        try:
            while True:
                chunk = self._stream_executor.submit(next, chunks, exhausted).result()
                if chunk is exhausted:
                    return
                yield chunk
        finally:
            self._stream_executor.submit(chunks.close).result()

    @staticmethod
    def _encode_audio(response_format: str, sample_rate: int, audio) -> bytes:
        import soundfile

        options = {"format": response_format.upper()}
        if response_format == "pcm":
            options.update(format="RAW", subtype="PCM_16", endian="LITTLE")
        elif response_format == "ogg":
            options["subtype"] = "VORBIS"
        with io.BytesIO() as output:
            soundfile.write(
                output,
                audio.detach().float().cpu().reshape(-1).numpy(),
                sample_rate,
                **options,
            )
            return output.getvalue()

    def _generate(
        self,
        options: dict,
        reference: Optional[bytes],
        edit: bool,
        response_format: str,
        seed: Optional[int],
        stream: bool,
    ):
        # Keep uploaded audio alive until the upstream iterator is closed.
        with (
            self._generation_context(seed),
            tempfile.TemporaryDirectory(prefix="xinference-dots-tts-") as directory,
        ):
            if reference is not None:
                path = Path(directory) / "reference.wav"
                path.write_bytes(reference)
                options["source_audio_path" if edit else "prompt_audio_path"] = str(
                    path
                )
            if stream:
                with closing(self._stream_chunks(options)) as chunks:
                    if response_format == "pcm":
                        for chunk in chunks:
                            yield self._encode_audio(
                                "pcm", self._runtime.sample_rate, chunk
                            )
                    else:
                        from .utils import audio_stream_generator

                        yield from audio_stream_generator(
                            response_format,
                            self._runtime.sample_rate,
                            chunks,
                            lambda chunk: chunk.detach().float().cpu().reshape(-1, 1),
                        )
            else:
                generate = (
                    self._runtime.generate_edit if edit else self._runtime.generate
                )
                with self._inference_context():
                    result = generate(**options)
                yield self._encode_audio(
                    response_format, result["sample_rate"], result["audio"]
                )

    def speech(
        self,
        input: str,
        voice: Optional[str] = None,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs: Any,
    ):
        if self._runtime is None:
            raise RuntimeError("dots.tts.edit model is not loaded.")
        if not isinstance(input, str) or not input.strip():
            raise InvalidModelRequestError(
                "dots.tts.edit input must be non-empty text."
            )
        if speed != 1.0:
            raise InvalidModelRequestError("dots.tts.edit only supports speed=1.0.")
        if not isinstance(stream, bool):
            raise InvalidModelRequestError("stream must be a boolean.")
        response_format = (response_format or "mp3").lower()
        if response_format not in {"mp3", "wav", "flac", "ogg", "pcm"}:
            raise InvalidModelRequestError("Unsupported dots.tts.edit response_format.")
        if voice:
            logger.warning(
                "dots.tts.edit ignores named voices; use prompt_speech for cloning."
            )

        instruction = kwargs.pop("instruction", None)
        instruct = kwargs.pop("instruct", None)
        if instruction is not None and instruct is not None and instruction != instruct:
            raise InvalidModelRequestError("instruction and instruct must agree.")
        instruction = instruction if instruction is not None else instruct
        template = kwargs.pop("template_name", None)
        task = kwargs.pop(
            "task",
            (
                "edit"
                if instruction is not None
                or template == "edit"
                or kwargs.get("source_audio_path") is not None
                else "tts"
            ),
        )
        if task not in {"tts", "edit"}:
            raise InvalidModelRequestError("task must be 'tts' or 'edit'.")
        edit = task == "edit"
        if not edit and (instruction is not None or template == "edit"):
            raise InvalidModelRequestError("Editing instructions require task='edit'.")
        if edit and stream:
            raise InvalidModelRequestError(
                "dots.tts.edit speech editing does not support streaming."
            )
        if edit and template not in {None, "edit"}:
            raise InvalidModelRequestError(
                "Speech editing requires template_name='edit'."
            )

        reference = kwargs.pop("prompt_speech", None)
        if reference is not None and (
            not isinstance(reference, bytes) or not reference
        ):
            raise InvalidModelRequestError(
                "prompt_speech must contain non-empty audio bytes."
            )
        path_key = "source_audio_path" if edit else "prompt_audio_path"
        path = kwargs.pop(path_key, None)
        if reference is not None and path is not None:
            raise InvalidModelRequestError(
                f"prompt_speech cannot be combined with {path_key}."
            )
        if path is not None and (not isinstance(path, str) or not Path(path).is_file()):
            raise InvalidModelRequestError(
                f"{path_key} must name an existing audio file."
            )
        if edit and reference is None and path is None:
            raise InvalidModelRequestError(
                "Speech editing requires prompt_speech or source_audio_path."
            )
        prompt_text = kwargs.pop("prompt_text", None)
        if not edit and prompt_text and reference is None and path is None:
            raise InvalidModelRequestError("prompt_text requires reference audio.")
        options = {path_key: path} if path is not None else {}
        if edit:
            options["instruction"] = instruction if instruction is not None else input
            source_text = kwargs.pop("source_text", None)
            if (
                source_text is not None
                and prompt_text is not None
                and source_text != prompt_text
            ):
                raise InvalidModelRequestError(
                    "source_text and prompt_text must agree."
                )
            options["source_text"] = (
                source_text if source_text is not None else prompt_text
            )
            options["target_text"] = kwargs.pop("target_text", None)
        else:
            options.update(text=input, prompt_text=prompt_text)
            if template is not None:
                options["template_name"] = template

        allowed = {
            "speaker_scale",
            "ode_method",
            "num_steps",
            "guidance_scale",
            "use_xvector",
            "profile_inference",
            "log_profile_calls",
        }
        if not edit:
            allowed.update({"language", "normalize_text"})
        seed = kwargs.pop("seed", None)
        from ..utils import resolve_media_seed

        try:
            seed = resolve_media_seed(seed)
        except ValueError as exc:
            raise InvalidModelRequestError(str(exc)) from exc
        if kwargs.keys() - allowed:
            raise InvalidModelRequestError(
                "Unsupported dots.tts.edit speech option(s): "
                + ", ".join(sorted(kwargs.keys() - allowed))
            )
        for name in ("speaker_scale", "guidance_scale"):
            if name in kwargs and (
                isinstance(kwargs[name], bool)
                or not isinstance(kwargs[name], (int, float))
                or not math.isfinite(kwargs[name])
                or kwargs[name] < 0
            ):
                raise InvalidModelRequestError(
                    f"{name} must be a finite non-negative number."
                )
        if "num_steps" in kwargs and (
            isinstance(kwargs["num_steps"], bool)
            or not isinstance(kwargs["num_steps"], int)
            or kwargs["num_steps"] < 1
        ):
            raise InvalidModelRequestError("num_steps must be a positive integer.")
        if "use_xvector" in kwargs and not (
            isinstance(kwargs["use_xvector"], bool)
            or edit
            and kwargs["use_xvector"] == "auto"
        ):
            raise InvalidModelRequestError(
                "use_xvector must be boolean (or 'auto' for editing)."
            )
        options.update(kwargs)

        def output():
            try:
                yield from self._generate(
                    options, reference, edit, response_format, seed, stream
                )
            except ValueError as exc:
                raise InvalidModelRequestError(str(exc)) from exc

        if stream:
            return output()
        with closing(output()) as result:
            return next(result)
