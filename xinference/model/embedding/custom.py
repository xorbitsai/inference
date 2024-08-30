# Copyright 2022-2023 XProbe Inc.
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
from threading import Lock
from typing import List, Optional

from ...constants import XINFERENCE_CACHE_DIR, XINFERENCE_MODEL_DIR
from .core import EmbeddingModelSpec

logger = logging.getLogger(__name__)


UD_EMBEDDING_LOCK = Lock()


class CustomEmbeddingModelSpec(EmbeddingModelSpec):
    model_id: Optional[str]  # type: ignore
    model_revision: Optional[str]  # type: ignore
    model_uri: Optional[str]


UD_EMBEDDINGS: List[CustomEmbeddingModelSpec] = []


def get_user_defined_embeddings() -> List[EmbeddingModelSpec]:
    with UD_EMBEDDING_LOCK:
        return UD_EMBEDDINGS.copy()


def register_embedding(model_spec: CustomEmbeddingModelSpec, persist: bool):
    from ...constants import XINFERENCE_MODEL_DIR
    from ..utils import is_valid_model_name, is_valid_model_uri
    from . import BUILTIN_EMBEDDING_MODELS, MODELSCOPE_EMBEDDING_MODELS

    if not is_valid_model_name(model_spec.model_name):
        raise ValueError(f"Invalid model name {model_spec.model_name}.")

    model_uri = model_spec.model_uri
    if model_uri and not is_valid_model_uri(model_uri):
        raise ValueError(f"Invalid model URI {model_uri}.")

    with UD_EMBEDDING_LOCK:
        for model_name in (
            list(BUILTIN_EMBEDDING_MODELS.keys())
            + list(MODELSCOPE_EMBEDDING_MODELS.keys())
            + [spec.model_name for spec in UD_EMBEDDINGS]
        ):
            if model_spec.model_name == model_name:
                raise ValueError(
                    f"Model name conflicts with existing model {model_spec.model_name}"
                )

        UD_EMBEDDINGS.append(model_spec)

    if persist:
        persist_path = os.path.join(
            XINFERENCE_MODEL_DIR, "embedding", f"{model_spec.model_name}.json"
        )
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        with open(persist_path, mode="w") as fd:
            fd.write(model_spec.json())


def unregister_embedding(model_name: str, raise_error: bool = True):
    with UD_EMBEDDING_LOCK:
        model_spec = None
        for i, f in enumerate(UD_EMBEDDINGS):
            if f.model_name == model_name:
                model_spec = f
                break
        if model_spec:
            UD_EMBEDDINGS.remove(model_spec)

            persist_path = os.path.join(
                XINFERENCE_MODEL_DIR, "embedding", f"{model_spec.model_name}.json"
            )
            if os.path.exists(persist_path):
                os.remove(persist_path)

            cache_dir = os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name)
            if os.path.exists(cache_dir):
                logger.warning(
                    f"Remove the cache of user-defined model {model_spec.model_name}. "
                    f"Cache directory: {cache_dir}"
                )
                if os.path.islink(cache_dir):
                    os.remove(cache_dir)
                else:
                    logger.warning(
                        f"Cache directory is not a soft link, please remove it manually."
                    )
        else:
            if raise_error:
                raise ValueError(f"Model {model_name} not found")
            else:
                logger.warning(f"Custom embedding model {model_name} not found")
