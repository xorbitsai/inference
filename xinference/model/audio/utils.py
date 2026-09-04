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
import typing
import wave
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from packaging import version

logger = logging.getLogger(__name__)


def apply_audio_seed(kwargs: dict) -> typing.Optional[int]:
    """Consume a generic audio seed and seed Python, NumPy, and Torch RNGs."""
    from ..utils import resolve_media_seed, set_all_random_seed

    seed = resolve_media_seed(kwargs.pop("seed", None))
    if seed is not None:
        set_all_random_seed(seed)
    return seed


def apply_mlx_audio_seed(kwargs: dict) -> typing.Optional[int]:
    """Consume a generic audio seed and seed MLX's RNG."""
    seed = apply_audio_seed(kwargs)
    if seed is not None:
        import mlx.core as mx

        mx.random.seed(seed)
    return seed


class MLXModelThreadMixin:
    """Pin every call into an MLX model to a single dedicated thread.

    MLX binds its default GPU stream to the OS thread that first touches
    it. ``ModelActor`` runs synchronous model methods (``load``, and the
    per-request inference methods) via ``asyncio.to_thread``, which uses a
    shared thread pool executor and gives no guarantee that two calls for
    the same model land on the same thread. When ``load()`` and a later
    inference call run on different threads, MLX raises
    ``RuntimeError: There is no Stream(gpu, N) in current thread`` and
    crashes the worker process. Routing every call through one persistent
    thread avoids this.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__mlx_executor: typing.Optional[ThreadPoolExecutor] = None

    def _run_on_mlx_thread(self, fn: Callable, *args, **kwargs):
        if self.__mlx_executor is None:
            self.__mlx_executor = ThreadPoolExecutor(max_workers=1)
        return self.__mlx_executor.submit(fn, *args, **kwargs).result()


def _extract_pcm_from_wav_bytes(wav_bytes):
    with io.BytesIO(wav_bytes) as wav_io:
        with wave.open(wav_io, "rb") as wav_file:
            num_frames = wav_file.getnframes()
            return wav_file.readframes(num_frames)


def ensure_sample_rate(
    audio: np.ndarray, old_sample_rate: int, sample_rate: int
) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample

    if old_sample_rate != sample_rate:
        # Calculate the new data length
        new_length = int(len(audio) * sample_rate / old_sample_rate)

        # Resample the data
        resampled_data = resample(audio, new_length)

        # Use BytesIO to save the resampled data to memory
        with io.BytesIO() as buffer:
            # Write the resampled data to the memory buffer
            sf.write(buffer, resampled_data, sample_rate, format="WAV")

            # Reset the buffer position to the beginning
            buffer.seek(0)

            # Read the data from the memory buffer
            audio, sr = sf.read(buffer, dtype="float32")

    return audio


def _audio_stream_generator_with_torchcodec(
    response_format: str,
    sample_rate: int,
    output_generator: typing.Iterator[typing.Any],
    output_chunk_transformer: Callable,
):
    try:
        from torchcodec.encoders import Encoder
    except ImportError as exc:
        raise ImportError(
            "Failed to import 'torchcodec'. It is required for audio streaming "
            "when 'torchaudio.io.StreamWriter' is unavailable. Install a "
            "TorchCodec build compatible with the installed PyTorch version."
        ) from exc

    normalized_format = response_format.lower().lstrip(".")
    response_pcm = normalized_format == "pcm"
    container_format = {
        "aac": "adts",
        "m4a": "mp4",
        "pcm": "wav",
        "wave": "wav",
    }.get(normalized_format, normalized_format)
    finalize_before_read = container_format == "mp4"
    with io.BytesIO() as out:
        encoder = Encoder()
        audio_stream = encoder.add_audio(sample_rate=sample_rate, num_channels=1)
        strip_header = True
        last_pos = 0
        has_audio = False

        def read_encoded_bytes() -> typing.Optional[bytes]:
            nonlocal last_pos, strip_header

            new_last_pos = out.tell()
            if new_last_pos == last_pos:
                return None
            out.seek(last_pos)
            encoded_bytes = out.read()
            last_pos = new_last_pos
            if response_pcm and strip_header:
                # http://soundfile.sapp.org/doc/WaveFormat
                encoded_bytes = _extract_pcm_from_wav_bytes(encoded_bytes)
                strip_header = False
            return encoded_bytes or None

        if response_pcm:
            logger.info(
                f"PCM stream output, num_channels: 1, sample_rate: {sample_rate}"
            )
        with encoder.open_file_like(out, format=container_format):
            for chunk in output_generator:
                trans_chunk = output_chunk_transformer(chunk)
                trans_chunk = trans_chunk.detach().to(device="cpu", dtype=torch.float32)
                if trans_chunk.ndim == 1:
                    trans_chunk = trans_chunk.unsqueeze(1)
                audio_stream.add_samples(trans_chunk.transpose(0, 1).contiguous())
                has_audio = True
                if not finalize_before_read:
                    encoded_bytes = read_encoded_bytes()
                    if encoded_bytes is not None:
                        yield encoded_bytes

        # Some codecs buffer the final packet until the encoder is closed.
        if has_audio:
            if finalize_before_read:
                # MP4 rewrites container metadata while closing, so bytes read
                # before this point would contain a stale header.
                encoded_bytes = out.getvalue()
            else:
                encoded_bytes = read_encoded_bytes()
            if encoded_bytes is not None:
                yield encoded_bytes


def audio_stream_generator(
    response_format: str,
    sample_rate: int,
    output_generator: typing.Iterator[typing.Any],
    output_chunk_transformer: Callable,
):
    import torch
    import torchaudio

    # torchaudio 2.9 removed StreamWriter in favor of TorchCodec.
    stream_writer = getattr(getattr(torchaudio, "io", None), "StreamWriter", None)
    if stream_writer is None:
        yield from _audio_stream_generator_with_torchcodec(
            response_format,
            sample_rate,
            output_generator,
            output_chunk_transformer,
        )
        return

    response_pcm = response_format.lower() == "pcm"
    with io.BytesIO() as out:
        if response_pcm:
            logger.info(
                f"PCM stream output, num_channels: 1, sample_rate: {sample_rate}"
            )
            writer = stream_writer(out, format="wav")
            writer.add_audio_stream(
                sample_rate=sample_rate, num_channels=1, format="s16"
            )
        else:
            writer = stream_writer(out, format=response_format)
            writer.add_audio_stream(sample_rate=sample_rate, num_channels=1)
        strip_header = True
        last_pos = 0
        with writer.open():
            for chunk in output_generator:
                trans_chunk = output_chunk_transformer(chunk)
                if response_pcm:
                    trans_chunk = trans_chunk.to(torch.float32)
                    trans_chunk = (
                        (trans_chunk * 32767).clamp(-32768, 32767).to(torch.int16)
                    )
                writer.write_audio_chunk(0, trans_chunk)
                new_last_pos = out.tell()
                if new_last_pos != last_pos:
                    out.seek(last_pos)
                    encoded_bytes = out.read()
                    if response_pcm and strip_header:
                        # http://soundfile.sapp.org/doc/WaveFormat
                        yield _extract_pcm_from_wav_bytes(encoded_bytes)
                        strip_header = False
                    else:
                        yield encoded_bytes
                    last_pos = new_last_pos


def audio_to_bytes(response_format: str, sample_rate: int, tensor: "torch.Tensor"):
    import torchaudio

    response_pcm = response_format.lower() == "pcm"
    if version.parse(torchaudio.version.__version__) < version.parse("2.9.0"):
        with io.BytesIO() as out:
            if response_pcm:
                logger.debug(f"PCM output, num_channels: 1, sample_rate: {sample_rate}")
                torchaudio.save(
                    out, tensor, sample_rate, format="wav", encoding="PCM_S"
                )
                # http://soundfile.sapp.org/doc/WaveFormat
                return _extract_pcm_from_wav_bytes(out.getvalue())
            else:
                torchaudio.save(out, tensor, sample_rate, format=response_format)
                return out.getvalue()
    else:
        import os
        import tempfile

        # ``NamedTemporaryFile`` keeps its own handle open for the whole
        # ``with`` block. On Windows that handle carries ``O_TEMPORARY``
        # (delete-on-close) and is not opened with ``FILE_SHARE_DELETE``, so
        # any second open of the same path -- both ``torchaudio.save`` and the
        # read-back below -- fails with ``PermissionError: [WinError 32]``.
        # Create the file, close our handle immediately, and clean it up
        # ourselves instead.
        fd, temp_path = tempfile.mkstemp(suffix=f".{response_format}")
        os.close(fd)
        try:
            if response_pcm:
                logger.debug(f"PCM output, num_channels: 1, sample_rate: {sample_rate}")
                torchaudio.save(
                    temp_path,
                    tensor,
                    sample_rate,
                    format="wav",
                    encoding="PCM_S",
                )
                # Read the temporary file and extract PCM data
                with open(temp_path, "rb") as f:
                    wav_bytes = f.read()
                return _extract_pcm_from_wav_bytes(wav_bytes)
            else:
                torchaudio.save(temp_path, tensor, sample_rate, format=response_format)
                # Read the temporary file and return its content
                with open(temp_path, "rb") as f:
                    return f.read()
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                logger.debug("Failed to remove temporary file %s", temp_path)
