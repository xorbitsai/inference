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
from typing import List, Literal, Optional

from ...constants import XINFERENCE_CACHE_DIR, XINFERENCE_MODEL_DIR
from .core import RerankModelSpec

logger = logging.getLogger(__name__)


UD_RERANK_LOCK = Lock()


class CustomRerankModelSpec(RerankModelSpec):
    model_id: Optional[str]  # type: ignore
    model_revision: Optional[str]  # type: ignore
    model_uri: Optional[str]
    model_type: Literal["rerank"] = "rerank"  # for frontend


UD_RERANKS: List[CustomRerankModelSpec] = []


def get_user_defined_reranks() -> List[CustomRerankModelSpec]:
    with UD_RERANK_LOCK:
        return UD_RERANKS.copy()


def register_rerank(model_spec: CustomRerankModelSpec, persist: bool):
    from ...constants import XINFERENCE_MODEL_DIR
    from ..utils import is_valid_model_name, is_valid_model_uri
    from . import BUILTIN_RERANK_MODELS, MODELSCOPE_RERANK_MODELS

    if not is_valid_model_name(model_spec.model_name):
        raise ValueError(f"Invalid model name {model_spec.model_name}.")

    model_uri = model_spec.model_uri
    if model_uri and not is_valid_model_uri(model_uri):
        raise ValueError(f"Invalid model URI {model_uri}.")

    with UD_RERANK_LOCK:
        for model_name in (
            list(BUILTIN_RERANK_MODELS.keys())
            + list(MODELSCOPE_RERANK_MODELS.keys())
            + [spec.model_name for spec in UD_RERANKS]
        ):
            if model_spec.model_name == model_name:
                raise ValueError(
                    f"Model name conflicts with existing model {model_spec.model_name}"
                )

        UD_RERANKS.append(model_spec)

    if persist:
        persist_path = os.path.join(
            XINFERENCE_MODEL_DIR, "rerank", f"{model_spec.model_name}.json"
        )
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        with open(persist_path, mode="w") as fd:
            fd.write(model_spec.json())


def unregister_rerank(model_name: str, raise_error: bool = True):
    with UD_RERANK_LOCK:
        model_spec = None
        for i, f in enumerate(UD_RERANKS):
            if f.model_name == model_name:
                model_spec = f
                break
        if model_spec:
            UD_RERANKS.remove(model_spec)

            persist_path = os.path.join(
                XINFERENCE_MODEL_DIR, "rerank", f"{model_spec.model_name}.json"
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
                logger.warning(f"Custom rerank model {model_name} not found")
