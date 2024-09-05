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

import gc
import logging
import os
import threading
import uuid
from collections import defaultdict
from collections.abc import Sequence
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ...constants import XINFERENCE_CACHE_DIR
from ...device_utils import empty_cache
from ...types import Document, DocumentObj, Rerank, RerankTokens
from ..core import CacheableModelSpec, ModelDescription
from ..utils import is_model_cached

logger = logging.getLogger(__name__)

# Used for check whether the model is cached.
# Init when registering all the builtin models.
MODEL_NAME_TO_REVISION: Dict[str, List[str]] = defaultdict(list)
RERANK_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)
RERANK_EMPTY_CACHE_COUNT = int(os.getenv("XINFERENCE_RERANK_EMPTY_CACHE_COUNT", "10"))
assert RERANK_EMPTY_CACHE_COUNT > 0


def get_rerank_model_descriptions():
    import copy

    return copy.deepcopy(RERANK_MODEL_DESCRIPTIONS)


class RerankModelSpec(CacheableModelSpec):
    model_name: str
    language: List[str]
    type: Optional[str] = "unknown"
    max_tokens: Optional[int]
    model_id: str
    model_revision: Optional[str]
    model_hub: str = "huggingface"


class RerankModelDescription(ModelDescription):
    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        model_spec: RerankModelSpec,
        model_path: Optional[str] = None,
    ):
        super().__init__(address, devices, model_path=model_path)
        self._model_spec = model_spec

    def to_dict(self):
        return {
            "model_type": "rerank",
            "address": self.address,
            "accelerators": self.devices,
            "type": self._model_spec.type,
            "model_name": self._model_spec.model_name,
            "language": self._model_spec.language,
            "model_revision": self._model_spec.model_revision,
        }

    def to_version_info(self):
        from .utils import get_model_version

        if self._model_path is None:
            is_cached = get_cache_status(self._model_spec)
            file_location = get_cache_dir(self._model_spec)
        else:
            is_cached = True
            file_location = self._model_path

        return {
            "model_version": get_model_version(self._model_spec),
            "model_file_location": file_location,
            "cache_status": is_cached,
            "language": self._model_spec.language,
        }


def generate_rerank_description(model_spec: RerankModelSpec) -> Dict[str, List[Dict]]:
    res = defaultdict(list)
    res[model_spec.model_name].append(
        RerankModelDescription(None, None, model_spec).to_version_info()
    )
    return res


class _ModelWrapper:
    def __init__(self, module: nn.Module):
        self._module = module
        self._local_data = threading.local()

    @property
    def n_tokens(self):
        return getattr(self._local_data, "n_tokens", 0)

    @n_tokens.setter
    def n_tokens(self, new_n_tokens):
        self._local_data.n_tokens = new_n_tokens

    def __getattr__(self, attr):
        return getattr(self._module, attr)

    def __call__(self, **kwargs):
        attention_mask = kwargs["attention_mask"]
        # when batching, the attention mask 1 means there is a token
        # thus we just sum up it to get the total number of tokens
        self.n_tokens += attention_mask.sum().item()
        return self._module(**kwargs)


class RerankModel:
    def __init__(
        self,
        model_spec: RerankModelSpec,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_fp16: bool = False,
        model_config: Optional[Dict] = None,
    ):
        self._model_spec = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._model_config = model_config or dict()
        self._use_fp16 = use_fp16
        self._model = None
        self._counter = 0
        if model_spec.type == "unknown":
            model_spec.type = self._auto_detect_type(model_path)

    @staticmethod
    def _get_tokenizer(model_path):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer

    @staticmethod
    def _auto_detect_type(model_path):
        """This method may not be stable due to the fact that the tokenizer name may be changed.
        Therefore, we only use this method for unknown model types."""

        type_mapper = {
            "LlamaTokenizerFast": "LLM-based layerwise",
            "GemmaTokenizerFast": "LLM-based",
            "XLMRobertaTokenizerFast": "normal",
        }

        tokenizer = RerankModel._get_tokenizer(model_path)
        rerank_type = type_mapper.get(type(tokenizer).__name__)
        if rerank_type is None:
            logger.warning(
                f"Can't determine the rerank type based on the tokenizer {tokenizer}, use normal type by default."
            )
            return "normal"
        return rerank_type

    def load(self):
        if self._model_spec.type == "normal":
            try:
                from sentence_transformers.cross_encoder import CrossEncoder
            except ImportError:
                error_message = "Failed to import module 'sentence-transformers'"
                installation_guide = [
                    "Please make sure 'sentence-transformers' is installed. ",
                    "You can install it by `pip install sentence-transformers`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
            self._model = CrossEncoder(
                self._model_path,
                device=self._device,
                trust_remote_code=True,
                max_length=getattr(self._model_spec, "max_tokens"),
                **self._model_config,
            )
            if self._use_fp16:
                self._model.model.half()
        else:
            try:
                if self._model_spec.type == "LLM-based":
                    from FlagEmbedding import FlagLLMReranker as FlagReranker
                elif self._model_spec.type == "LLM-based layerwise":
                    from FlagEmbedding import LayerWiseFlagLLMReranker as FlagReranker
                else:
                    raise RuntimeError(
                        f"Unsupported Rank model type: {self._model_spec.type}"
                    )
            except ImportError:
                error_message = "Failed to import module 'FlagEmbedding'"
                installation_guide = [
                    "Please make sure 'FlagEmbedding' is installed. ",
                    "You can install it by `pip install FlagEmbedding`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
            self._model = FlagReranker(self._model_path, use_fp16=self._use_fp16)
        # Wrap transformers model to record number of tokens
        self._model.model = _ModelWrapper(self._model.model)

    def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int],
        max_chunks_per_doc: Optional[int],
        return_documents: Optional[bool],
        return_len: Optional[bool],
        **kwargs,
    ) -> Rerank:
        assert self._model is not None
        if kwargs:
            raise ValueError("rerank hasn't support extra parameter.")
        if max_chunks_per_doc is not None:
            raise ValueError("rerank hasn't support `max_chunks_per_doc` parameter.")
        sentence_combinations = [[query, doc] for doc in documents]
        # reset n tokens
        self._model.model.n_tokens = 0
        if self._model_spec.type == "normal":
            similarity_scores = self._model.predict(
                sentence_combinations, convert_to_numpy=False, convert_to_tensor=True
            ).cpu()
            if similarity_scores.dtype == torch.bfloat16:
                similarity_scores = similarity_scores.float()
        else:
            # Related issue: https://github.com/xorbitsai/inference/issues/1775
            similarity_scores = self._model.compute_score(sentence_combinations)
            if not isinstance(similarity_scores, Sequence):
                similarity_scores = [similarity_scores]

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
        if return_len:
            input_len = self._model.model.n_tokens
            # Rerank Model output is just score or documents
            # while return_documents = True
            output_len = input_len

        # api_version, billed_units, warnings
        # is for Cohere API compatibility, set to None
        metadata = {
            "api_version": None,
            "billed_units": None,
            "tokens": (
                RerankTokens(input_tokens=input_len, output_tokens=output_len)
                if return_len
                else None
            ),
            "warnings": None,
        }

        del similarity_scores
        # clear cache if possible
        self._counter += 1
        if self._counter % RERANK_EMPTY_CACHE_COUNT == 0:
            logger.debug("Empty rerank cache.")
            gc.collect()
            empty_cache()

        return Rerank(id=str(uuid.uuid1()), results=docs, meta=metadata)


def get_cache_dir(model_spec: RerankModelSpec):
    return os.path.realpath(os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name))


def get_cache_status(
    model_spec: RerankModelSpec,
) -> bool:
    return is_model_cached(model_spec, MODEL_NAME_TO_REVISION)


def cache(model_spec: RerankModelSpec):
    from ..utils import cache

    return cache(model_spec, RerankModelDescription)


def create_rerank_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_name: str,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[RerankModel, RerankModelDescription]:
    from ..utils import download_from_modelscope
    from . import BUILTIN_RERANK_MODELS, MODELSCOPE_RERANK_MODELS
    from .custom import get_user_defined_reranks

    model_spec = None
    for ud_spec in get_user_defined_reranks():
        if ud_spec.model_name == model_name:
            model_spec = ud_spec
            break

    if model_spec is None:
        if download_hub == "huggingface" and model_name in BUILTIN_RERANK_MODELS:
            logger.debug(f"Rerank model {model_name} found in Huggingface.")
            model_spec = BUILTIN_RERANK_MODELS[model_name]
        elif download_hub == "modelscope" and model_name in MODELSCOPE_RERANK_MODELS:
            logger.debug(f"Rerank model {model_name} found in ModelScope.")
            model_spec = MODELSCOPE_RERANK_MODELS[model_name]
        elif download_from_modelscope() and model_name in MODELSCOPE_RERANK_MODELS:
            logger.debug(f"Rerank model {model_name} found in ModelScope.")
            model_spec = MODELSCOPE_RERANK_MODELS[model_name]
        elif model_name in BUILTIN_RERANK_MODELS:
            logger.debug(f"Rerank model {model_name} found in Huggingface.")
            model_spec = BUILTIN_RERANK_MODELS[model_name]
        else:
            raise ValueError(
                f"Rerank model {model_name} not found, available"
                f"Huggingface: {BUILTIN_RERANK_MODELS.keys()}"
                f"ModelScope: {MODELSCOPE_RERANK_MODELS.keys()}"
            )
    if not model_path:
        model_path = cache(model_spec)
    use_fp16 = kwargs.pop("use_fp16", False)
    model = RerankModel(
        model_spec, model_uid, model_path, use_fp16=use_fp16, model_config=kwargs
    )
    model_description = RerankModelDescription(
        subpool_addr, devices, model_spec, model_path=model_path
    )
    return model, model_description
