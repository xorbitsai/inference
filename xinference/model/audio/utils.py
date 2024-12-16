# Copyright 2022-2024 XProbe Inc.
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

import numpy as np

from .core import AudioModelFamilyV1


def get_model_version(audio_model: AudioModelFamilyV1) -> str:
    return audio_model.model_name


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
