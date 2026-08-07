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
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from ...device_utils import get_available_device, is_device_available

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2

logger = logging.getLogger(__name__)


class ModelScopeSpeakerEmbeddingModel:
    """Extract speaker embeddings with a ModelScope speaker-verification model."""

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
        self._pipeline = None
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def load(self):
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
        except ImportError as e:
            raise ImportError(
                "Failed to import ModelScope speaker-verification dependencies. "
                "Please install `modelscope[framework]`, `soundfile`, and "
                "`scikit-learn`."
            ) from e

        if self._device is None:
            self._device = get_available_device()
        elif not is_device_available(self._device):
            raise ValueError(f"Device {self._device} is not available!")

        logger.debug(
            "Loading ModelScope speaker embedding model from %s on %s",
            self._model_path,
            self._device,
        )
        self._pipeline = pipeline(
            task=Tasks.speaker_verification,
            model=self._model_path,
            device=self._device,
            **self._kwargs,
        )

    def create_embedding(
        self, audio: bytes, model_uid: Optional[str] = None
    ) -> Dict[str, Any]:
        if not audio:
            raise ValueError("Audio input must not be empty.")
        if self._pipeline is None:
            raise RuntimeError("Speaker embedding model is not loaded.")

        temp_path = None
        try:
            # ModelScope's speaker-verification pipeline accepts file paths or
            # NumPy waveforms, but not encoded audio bytes. Close the temporary
            # file before inference so this also works on Windows.
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(audio)
                temp_path = f.name

            result = self._pipeline([temp_path], output_emb=True)
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

        if not isinstance(result, dict) or "embs" not in result:
            raise RuntimeError(
                f"ModelScope returned an invalid speaker embedding result: {result!r}"
            )

        embeddings = result["embs"]
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach()
        if hasattr(embeddings, "cpu"):
            embeddings = embeddings.cpu()
        if hasattr(embeddings, "numpy"):
            embeddings = embeddings.numpy()

        embedding_array = np.asarray(embeddings, dtype=np.float32)
        if embedding_array.ndim == 2 and embedding_array.shape[0] == 1:
            embedding_array = embedding_array[0]
        if embedding_array.ndim != 1 or embedding_array.size == 0:
            raise RuntimeError(
                "ModelScope returned an unexpected speaker embedding shape: "
                f"{embedding_array.shape}"
            )
        if not np.isfinite(embedding_array).all():
            raise RuntimeError("ModelScope returned a non-finite speaker embedding.")

        return {
            "object": "embedding",
            "model": model_uid or self._model_uid,
            "dimensions": int(embedding_array.size),
            "embedding": embedding_array.tolist(),
        }
