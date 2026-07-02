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
import logging
import os
import tempfile
import wave
from io import BytesIO
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


class _CleanupGenerator:
    def __init__(self, generator, cleanup):
        self._generator = generator
        self._cleanup = cleanup
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._generator)
        except StopIteration:
            self.close()
            raise
        except Exception:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            close = getattr(self._generator, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass
        finally:
            self._cleanup()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class VoxCPMModel:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "AudioModelFamilyV2",
        device: Optional[str] = None,
        **kwargs,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._device = device
        self._model = None
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    @staticmethod
    def _save_temp_audio(audio: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            return f.name

    @staticmethod
    def _cleanup_temp_files(temp_files):
        while temp_files:
            temp_file = temp_files.pop()
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    @staticmethod
    def _as_stream_tensor(chunk):
        import torch

        tensor = torch.as_tensor(chunk, dtype=torch.float32).cpu()
        if tensor.dim() == 1:
            return tensor.unsqueeze(1)
        if tensor.dim() == 2 and tensor.shape[0] == 1:
            return tensor.transpose(0, 1)
        return tensor

    @staticmethod
    def _as_audio_array(audio):
        import numpy as np

        try:
            import torch
        except ImportError:
            torch = None  # type: ignore

        if torch is not None and isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        audio_array = np.asarray(audio, dtype=np.float32)
        if audio_array.ndim == 0:
            audio_array = audio_array.reshape(1)
        elif audio_array.ndim == 2:
            if audio_array.shape[0] == 1:
                audio_array = audio_array[0]
            elif audio_array.shape[1] == 1:
                audio_array = audio_array[:, 0]
            elif audio_array.shape[0] < audio_array.shape[1]:
                audio_array = audio_array.T
        elif audio_array.ndim > 2:
            raise ValueError(f"Unsupported audio shape: {audio_array.shape}")

        return audio_array

    @staticmethod
    def _float_to_pcm16(audio):
        import numpy as np

        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")

    @classmethod
    def _audio_to_bytes(cls, response_format: str, sample_rate: int, audio) -> bytes:
        audio_array = cls._as_audio_array(audio)
        response_format = response_format.lower()

        if response_format == "pcm":
            return cls._float_to_pcm16(audio_array).tobytes()

        if audio_array.ndim == 1:
            channels = 1
        else:
            channels = audio_array.shape[1]

        if response_format == "wav":
            with BytesIO() as out:
                with wave.open(out, "wb") as wav_file:
                    wav_file.setnchannels(channels)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(cls._float_to_pcm16(audio_array).tobytes())
                return out.getvalue()

        import soundfile

        with BytesIO() as out:
            with soundfile.SoundFile(
                out,
                "w",
                sample_rate,
                channels,
                format=response_format.upper(),
            ) as f:
                f.write(audio_array)
            return out.getvalue()

    def _get_sample_rate(self) -> int:
        assert self._model is not None
        return int(getattr(self._model.tts_model, "sample_rate", 48000))

    def load(self):
        try:
            from voxcpm import VoxCPM
        except ImportError:
            error_message = "Failed to import module 'voxcpm'"
            installation_guide = [
                "Please make sure 'voxcpm' is installed. ",
                "You can install it by `pip install voxcpm`\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        compile_opt = self._kwargs.pop("compile", None)
        optimize_opt = self._kwargs.pop("optimize", None)
        optimize = bool(
            optimize_opt if optimize_opt is not None else compile_opt or False
        )
        load_denoiser = bool(self._kwargs.pop("load_denoiser", False))
        zipenhancer_model_id = self._kwargs.pop("zipenhancer_model_id", None)

        logger.info(
            "Loading VoxCPM model, optimize=%s, load_denoiser=%s...",
            optimize,
            load_denoiser,
        )
        load_kwargs = dict(
            load_denoiser=load_denoiser,
            optimize=optimize,
            device=self._device,
        )
        if zipenhancer_model_id is not None:
            load_kwargs["zipenhancer_model_id"] = zipenhancer_model_id
        self._model = VoxCPM.from_pretrained(
            self._model_path, **load_kwargs, **self._kwargs
        )

    def _build_generate_kwargs(
        self,
        input: str,
        voice: str,
        speed: float,
        kwargs,
        temp_files,
    ):
        prompt_speech: Optional[bytes] = kwargs.pop("prompt_speech", None)
        reference_speech: Optional[bytes] = kwargs.pop("reference_speech", None)
        prompt_text: Optional[str] = kwargs.pop("prompt_text", None)
        control_instruction: Optional[str] = kwargs.pop(
            "control_instruction", None
        ) or kwargs.pop("instruct_text", None)

        prompt_wav_path: Optional[str] = kwargs.pop("prompt_wav_path", None)
        reference_wav_path: Optional[str] = kwargs.pop("reference_wav_path", None)

        if prompt_speech is not None:
            prompt_speech_path = self._save_temp_audio(prompt_speech)
            temp_files.append(prompt_speech_path)
            if prompt_text:
                prompt_wav_path = prompt_wav_path or prompt_speech_path
                reference_wav_path = reference_wav_path or prompt_speech_path
            else:
                reference_wav_path = reference_wav_path or prompt_speech_path

        if reference_speech is not None:
            reference_speech_path = self._save_temp_audio(reference_speech)
            temp_files.append(reference_speech_path)
            reference_wav_path = reference_wav_path or reference_speech_path

        openai_voices = {
            "alloy",
            "ash",
            "ballad",
            "coral",
            "echo",
            "fable",
            "nova",
            "onyx",
            "sage",
            "shimmer",
            "verse",
        }
        if not control_instruction and voice and voice.lower() not in openai_voices:
            control_instruction = voice

        text = input
        if control_instruction:
            control_instruction = control_instruction.replace("(", "").replace(")", "")
            control_instruction = control_instruction.replace("（", "").replace(
                "）", ""
            )
            text = f"({control_instruction.strip()}){input}"

        generate_kwargs = {
            "text": text,
            "prompt_wav_path": prompt_wav_path,
            "prompt_text": prompt_text,
            "reference_wav_path": reference_wav_path,
            "cfg_value": float(kwargs.pop("cfg_value", 2.0)),
            "inference_timesteps": int(kwargs.pop("inference_timesteps", 10)),
            "min_len": int(kwargs.pop("min_len", 2)),
            "max_len": int(kwargs.pop("max_len", 4096)),
            "normalize": bool(kwargs.pop("normalize", False)),
            "denoise": bool(kwargs.pop("denoise", False)),
            "retry_badcase": bool(kwargs.pop("retry_badcase", True)),
            "retry_badcase_max_times": int(kwargs.pop("retry_badcase_max_times", 3)),
            "retry_badcase_ratio_threshold": float(
                kwargs.pop("retry_badcase_ratio_threshold", 6.0)
            ),
        }

        if speed != 1.0:
            logger.warning("VoxCPM does not support speed parameter, ignoring it.")

        if kwargs:
            logger.warning("Ignoring unsupported VoxCPM speech kwargs: %s", kwargs)

        return generate_kwargs

    def speech(
        self,
        input: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs,
    ):
        assert self._model is not None

        from .utils import audio_stream_generator

        temp_files = []
        cleanup_temp_files = True
        try:
            generate_kwargs = self._build_generate_kwargs(
                input=input,
                voice=voice,
                speed=speed,
                kwargs=kwargs,
                temp_files=temp_files,
            )

            if stream:
                sample_rate = self._get_sample_rate()
                cleanup_temp_files = False
                cleanup = lambda: self._cleanup_temp_files(temp_files)

                if response_format.lower() == "pcm":

                    def _pcm_stream():
                        for chunk in self._model.generate_streaming(**generate_kwargs):
                            yield self._audio_to_bytes(
                                response_format=response_format,
                                sample_rate=sample_rate,
                                audio=chunk,
                            )

                    return _CleanupGenerator(_pcm_stream(), cleanup)

                stream_generator = audio_stream_generator(
                    response_format=response_format,
                    sample_rate=sample_rate,
                    output_generator=self._model.generate_streaming(**generate_kwargs),
                    output_chunk_transformer=self._as_stream_tensor,
                )
                return _CleanupGenerator(stream_generator, cleanup)

            wav = self._model.generate(**generate_kwargs)
            return self._audio_to_bytes(
                response_format=response_format,
                sample_rate=self._get_sample_rate(),
                audio=wav,
            )
        finally:
            if cleanup_temp_files:
                self._cleanup_temp_files(temp_files)
