# Copyright 2022-2026 XProbe Inc.
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

import numpy as np
import torch
from packaging import version

logger = logging.getLogger(__name__)


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


def audio_stream_generator(
    response_format: str,
    sample_rate: int,
    output_generator: typing.Generator[typing.Any, None, None],
    output_chunk_transformer: Callable,
):
    import torch
    import torchaudio

    response_pcm = response_format.lower() == "pcm"
    with io.BytesIO() as out:
        if response_pcm:
            logger.info(
                f"PCM stream output, num_channels: 1, sample_rate: {sample_rate}"
            )
            writer = torchaudio.io.StreamWriter(out, format="wav")
            writer.add_audio_stream(
                sample_rate=sample_rate, num_channels=1, format="s16"
            )
        else:
            writer = torchaudio.io.StreamWriter(out, format=response_format)
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
        import tempfile

        with tempfile.NamedTemporaryFile(
            delete=True, suffix=f".{response_format}"
        ) as temp_file:
            if response_pcm:
                logger.debug(f"PCM output, num_channels: 1, sample_rate: {sample_rate}")
                torchaudio.save(
                    temp_file.name,
                    tensor,
                    sample_rate,
                    format="wav",
                    encoding="PCM_S",
                )
                # Read the temporary file and extract PCM data
                with open(temp_file.name, "rb") as f:
                    wav_bytes = f.read()
                return _extract_pcm_from_wav_bytes(wav_bytes)
            else:
                torchaudio.save(
                    temp_file.name, tensor, sample_rate, format=response_format
                )
                # Read the temporary file and return its content
                with open(temp_file.name, "rb") as f:
                    return f.read()
