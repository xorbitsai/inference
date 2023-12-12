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
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from ...constants import XINFERENCE_CACHE_DIR
from ...types import Document, DocumentObj, Rerank
from ..core import ModelDescription
from ..utils import is_model_cached, valid_model_revision

logger = logging.getLogger(__name__)

# Used for check whether the model is cached.
# Init when registering all the builtin models.
MODEL_NAME_TO_REVISION: Dict[str, List[str]] = defaultdict(list)


class RerankModelSpec(BaseModel):
    model_name: str
    language: List[str]
    model_id: str
    model_revision: str
    model_hub: str = "huggingface"


class RerankModelDescription(ModelDescription):
    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        model_spec: RerankModelSpec,
    ):
        super().__init__(address, devices)
        self._model_spec = model_spec

    def to_dict(self):
        return {
            "model_type": "rerank",
            "address": self.address,
            "accelerators": self.devices,
            "model_name": self._model_spec.model_name,
            "language": self._model_spec.language,
            "model_revision": self._model_spec.model_revision,
        }


class RerankModel:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        device: Optional[str] = None,
        model_config: Optional[Dict] = None,
    ):
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._model_config = model_config or dict()
        self._model = None

    def load(self):
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
        except ImportError:
            error_message = "Failed to import module 'SentenceTransformer'"
            installation_guide = [
                "Please make sure 'sentence-transformers' is installed. ",
                "You can install it by `pip install sentence-transformers`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
        self._model = CrossEncoder(
            self._model_path, device=self._device, **self._model_config
        )

    def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int],
        max_chunks_per_doc: Optional[int],
        return_documents: Optional[bool],
    ) -> Rerank:
        assert self._model is not None
        if max_chunks_per_doc is not None:
            raise ValueError("rerank hasn't support `max_chunks_per_doc` parameter.")
        sentence_combinations = [[query, doc] for doc in documents]
        similarity_scores = self._model.predict(sentence_combinations)
        sim_scores_argsort = list(reversed(np.argsort(similarity_scores)))
        if top_n is not None:
            sim_scores_argsort = sim_scores_argsort[:top_n]
        if return_documents:
            docs = [
                DocumentObj(
                    index=int(arg),
                    relevance_score=float(similarity_scores[arg]),
                    document=Document(text=documents[arg]),
                )
                for arg in sim_scores_argsort
            ]
        else:
            docs = [
                DocumentObj(
                    index=int(arg),
                    relevance_score=float(similarity_scores[arg]),
                    document=None,
                )
                for arg in sim_scores_argsort
            ]
        return Rerank(id=str(uuid.uuid1()), results=docs)


def get_cache_status(
    model_spec: RerankModelSpec,
) -> bool:
    return is_model_cached(model_spec, MODEL_NAME_TO_REVISION)


def cache(model_spec: RerankModelSpec):
    # TODO: cache from uri
    from huggingface_hub import snapshot_download as hf_download
    from modelscope.hub.snapshot_download import snapshot_download as ms_download

    from ..utils import retry_download, symlink_local_file

    cache_dir = os.path.realpath(
        os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name)
    )
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "__valid_download")
    if valid_model_revision(meta_path, model_spec.model_revision):
        return cache_dir

    if model_spec.model_hub == "modelscope":
        logger.info(
            f"Download {model_spec.model_name} from modelscope {model_spec.model_id}"
        )
        download_dir = retry_download(
            ms_download,
            model_spec.model_name,
            None,
            model_spec.model_id,
            revision=model_spec.model_revision,
        )
        for subdir, dirs, files in os.walk(download_dir):
            for file in files:
                relpath = os.path.relpath(os.path.join(subdir, file), download_dir)
                symlink_local_file(os.path.join(subdir, file), cache_dir, relpath)
    else:
        logger.info(
            f"Download {model_spec.model_name} from huggingface {model_spec.model_id}"
        )
        retry_download(
            hf_download,
            model_spec.model_name,
            None,
            model_spec.model_id,
            revision=model_spec.model_revision,
            local_dir=cache_dir,
            local_dir_use_symlinks=True,
        )
    with open(meta_path, "w") as f:
        import json

        desc = RerankModelDescription(None, None, model_spec)
        json.dump(desc.to_dict(), f)
    return cache_dir


def create_rerank_model_instance(
    subpool_addr: str, devices: List[str], model_uid: str, model_name: str, **kwargs
) -> Tuple[RerankModel, RerankModelDescription]:
    from ..utils import download_from_modelscope
    from . import BUILTIN_RERANK_MODELS, MODELSCOPE_RERANK_MODELS

    if download_from_modelscope():
        if model_name in MODELSCOPE_RERANK_MODELS:
            logger.debug(f"Rerank model {model_name} found in ModelScope.")
            model_spec = MODELSCOPE_RERANK_MODELS[model_name]
        else:
            logger.debug(
                f"Rerank model {model_name} not found in ModelScope, "
                f"now try to download from huggingface."
            )
            if model_name in BUILTIN_RERANK_MODELS:
                model_spec = BUILTIN_RERANK_MODELS[model_name]
            else:
                raise ValueError(
                    f"Rerank model {model_name} not found, available"
                    f"model list: {BUILTIN_RERANK_MODELS.keys()}"
                )
    else:
        if model_name in BUILTIN_RERANK_MODELS:
            model_spec = BUILTIN_RERANK_MODELS[model_name]
        else:
            raise ValueError(
                f"Rerank model {model_name} not found, available"
                f"model list: {BUILTIN_RERANK_MODELS.keys()}"
            )

    model_path = cache(model_spec)
    model = RerankModel(model_uid, model_path, **kwargs)
    model_description = RerankModelDescription(subpool_addr, devices, model_spec)
    return model, model_description
