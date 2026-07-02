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

import logging
import os

from ...device_utils import get_available_device, get_device_preferred_dtype
from .qwen3_asr import Qwen3ASRModel

logger = logging.getLogger(__name__)


class VLLMQwen3ASRModel(Qwen3ASRModel):
    """Qwen3-ASR served by the vLLM backend of the ``qwen_asr`` package.

    ``qwen_asr`` exposes the same ``transcribe`` API for both its
    transformers backend (``Qwen3ASRModel.from_pretrained``) and its vLLM
    backend (``Qwen3ASRModel.LLM``), so only model loading differs here.
    """

    def load(self):
        try:
            from qwen_asr import Qwen3ASRModel as QwenASR

            QwenASR.LLM
        except ImportError:
            error_message = (
                "Failed to import module 'qwen_asr' with vLLM backend support"
            )
            installation_guide = [
                "Please make sure 'qwen-asr' with vLLM support is installed. ",
                "You can install it by `pip install 'qwen-asr[vllm]'`\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
        except AttributeError:
            raise RuntimeError(
                "The installed 'qwen-asr' does not support the vLLM backend, "
                "please upgrade it by `pip install -U 'qwen-asr[vllm]'`"
            )

        init_kwargs = (
            self._model_spec.default_model_config.copy()
            if getattr(self._model_spec, "default_model_config", None)
            else {}
        )
        init_kwargs.update(self._kwargs)
        # transformers-only args that must not reach vllm.LLM
        for key in ("device_map", "dtype", "torch_dtype"):
            init_kwargs.pop(key, None)

        forced_aligner = init_kwargs.get("forced_aligner")
        if isinstance(forced_aligner, str) and not os.path.exists(forced_aligner):
            resolved = self._resolve_forced_aligner(forced_aligner)
            if resolved:
                init_kwargs["forced_aligner"] = resolved
        if "forced_aligner" in init_kwargs:
            # the forced aligner still runs on transformers
            device = self._device or get_available_device()
            forced_aligner_kwargs = init_kwargs.get("forced_aligner_kwargs") or {}
            forced_aligner_kwargs.setdefault("device_map", device)
            forced_aligner_kwargs.setdefault(
                "dtype", get_device_preferred_dtype(device)
            )
            init_kwargs["forced_aligner_kwargs"] = forced_aligner_kwargs

        logger.debug("Loading Qwen3-ASR model with vLLM, kwargs: %s", init_kwargs)
        self._model = QwenASR.LLM(model=self._model_path, **init_kwargs)
