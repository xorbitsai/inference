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


from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from ...core.exceptions import InvalidModelRequestError
from ..utils import allow_trust_remote_code
from .zonos3_mini import SAMPLE_RATE, Zonos3MiniModel

if TYPE_CHECKING:
    import numpy as np

    from .core import AudioModelFamilyV2


class TransformersZonos3MiniModel(Zonos3MiniModel):
    @staticmethod
    def check_model_trust(model_spec: "AudioModelFamilyV2") -> None:
        if not allow_trust_remote_code(model_spec):
            raise ValueError(
                "An explicit ZONOS3-Mini Transformers model_path or a custom "
                "checkpoint requires XINFERENCE_TRUST_REMOTE_CODE=1 on the worker."
            )

    def load(self) -> None:
        self.check_model_trust(self.model_family)

        import torch
        from packaging.version import Version
        from transformers import AutoModel, __version__

        if Version(__version__) < Version("4.57"):
            raise ImportError("ZONOS3-Mini requires transformers>=4.57.")
        device = torch.device(
            self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("ZONOS3-Mini Transformers supports CPU and CUDA.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("ZONOS3-Mini requires an available CUDA device.")

        self._model = (
            AutoModel.from_pretrained(
                self._model_path,
                local_files_only=True,
                trust_remote_code=True,
                dtype=torch.float32 if device.type == "cpu" else torch.bfloat16,
            )
            .eval()
            .to(device)
        )

    def _generate(
        self,
        text: str,
        voice: str,
        reference_path: Optional[Path],
        language: Optional[str],
        options: Dict[str, Any],
    ) -> "np.ndarray":
        import soundfile

        generate_options = options.copy()
        if "top_k" in generate_options:
            generate_options["topk"] = generate_options.pop("top_k")
        if "max_new_tokens" in generate_options:
            generate_options["max_new_frames"] = generate_options.pop("max_new_tokens")
        if language is not None:
            generate_options["language"] = language
        if reference_path is None:
            generate_options["voice"] = voice
        else:
            audio, rate = soundfile.read(reference_path, dtype="float32")
            generate_options.update(speaker_audio=audio, speaker_sample_rate=rate)
        try:
            result = self._model.generate(text, **generate_options)
        except ValueError as exc:
            raise InvalidModelRequestError(str(exc)) from exc
        if result.sampling_rate != SAMPLE_RATE:
            raise RuntimeError(
                f"Unexpected ZONOS3-Mini sample rate: {result.sampling_rate}"
            )
        return result.waveform.detach().float().cpu().numpy()
