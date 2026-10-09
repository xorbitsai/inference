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

import base64
import io
import math
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from urllib.parse import unquote, urlparse

from ....types import Embedding, EmbeddingUsage
from ..core import EmbeddingModelFamilyV2, EmbeddingSpecV1
from ..embeddinggemma2 import MODEL_NAME, normalize_inputs, validate_dimensions
from .core import XllamaCppEmbeddingModel

# The GGUF repository does not include config_sentence_transformers.json.
PROMPTS = {
    "BitextMining": "task: search result | query: ",
    "Classification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
    "CodeRetrieval": "task: code retrieval | query: ",
    "Document": "title: none | text: ",
    "FactChecking": "task: fact checking | query: ",
    "InstructionRetrieval": "task: code retrieval | query: ",
    "MultilabelClassification": "task: classification | query: ",
    "PairClassification": "task: sentence similarity | query: ",
    "QuestionAnswering": "task: question answering | query: ",
    "Reranking": "task: search result | query: ",
    "Retrieval": "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    "Retrieval-query": "task: search result | query: ",
    "STS": "task: sentence similarity | query: ",
    "SearchQuery": "task: search result | query: ",
    "SentenceSimilarity": "task: sentence similarity | query: ",
    "Summarization": "task: sentence similarity | query: ",
    "document": "title: none | text: ",
    "query": "task: search result | query: ",
}


def _media_url(value: Any, modality: str) -> str:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if isinstance(value, str):
        if value.startswith(("http://", "https://", "data:")):
            return value
        path = unquote(urlparse(value).path) if value.startswith("file://") else value
        mime = mimetypes.guess_type(path)[0] or f"{modality}/octet-stream"
        data = Path(path).read_bytes()
    elif isinstance(value, bytes):
        mime = f"{modality}/octet-stream"
        data = value
    elif modality == "image":
        from PIL import Image

        if not isinstance(value, Image.Image):
            raise ValueError(
                "EmbeddingGemma 2 GGUF image must be a URL, path, bytes, or PIL image."
            )
        buffer = io.BytesIO()
        value.convert("RGB").save(buffer, format="PNG")
        data = buffer.getvalue()
        mime = "image/png"
    else:
        raise ValueError(
            f"EmbeddingGemma 2 GGUF {modality} must be a URL, path, or bytes."
        )
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def prepare_inputs(sentences: Any, **kwargs: Any) -> List[Dict[str, Any]]:
    messages = normalize_inputs(sentences, PROMPTS, **kwargs)
    inputs = []
    for sample in messages:
        content = []
        for part in sample[0]["content"]:
            modality = part["type"]
            if modality == "text":
                content.append(part)
            else:
                url = _media_url(part[modality], modality)
                if modality == "image":
                    content.append({"type": "image_url", "image_url": {"url": url}})
                else:
                    key = "input_audio" if modality == "audio" else "input_video"
                    content.append({"type": key, key: {"url": url}})
        # llama.cpp embeddings require content nested inside each batch item.
        inputs.append({"content": content})
    return inputs


class XllamaCppEmbeddingGemma2Model(XllamaCppEmbeddingModel):
    supports_dimensions = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._dimensions = self._kwargs.get("dimensions")
        validate_dimensions(self._dimensions)

    def _configure_params(self, params: Any) -> None:
        from xllamacpp import ggml_type, llama_pooling_type

        config = self._llamacpp_model_config
        params.n_parallel = 1
        params.n_ctx = config.get("n_ctx", self.model_family.max_tokens)
        params.n_batch = config.get("n_batch", params.n_ctx)
        params.n_ubatch = config.get("n_ubatch", params.n_ctx)
        params.pooling_type = llama_pooling_type.LLAMA_POOLING_TYPE_MEAN
        params.embd_normalize = -1
        # The model requires BF16/FP32 rather than FP16 computation.
        params.cache_type_k = ggml_type.GGML_TYPE_F32
        params.cache_type_v = ggml_type.GGML_TYPE_F32
        if not 0 < params.n_ctx <= self.model_family.max_tokens:
            raise ValueError("EmbeddingGemma 2 GGUF n_ctx must be between 1 and 8192.")
        if min(params.n_batch, params.n_ubatch) < params.n_ctx:
            raise ValueError(
                "EmbeddingGemma 2 GGUF requires n_batch and n_ubatch >= n_ctx for non-causal attention."
            )
        if config.get("n_parallel", 1) != 1 or not config.get("embedding", True):
            raise ValueError(
                "EmbeddingGemma 2 GGUF requires embedding=True and n_parallel=1."
            )
        if config.get("pooling_type", params.pooling_type) != params.pooling_type:
            raise ValueError("EmbeddingGemma 2 GGUF requires mean pooling.")
        if config.get("embd_normalize", -1) != -1:
            raise ValueError(
                "EmbeddingGemma 2 GGUF requires embd_normalize=-1; normalization is applied after dimension truncation."
            )
        for key in ("cache_type_k", "cache_type_v"):
            if config.get(key, ggml_type.GGML_TYPE_F32) not in (
                ggml_type.GGML_TYPE_F32,
                ggml_type.GGML_TYPE_BF16,
            ):
                raise ValueError(
                    "EmbeddingGemma 2 GGUF requires BF16/FP32 KV cache types."
                )

    def load(self) -> None:
        try:
            super().load()
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc}. EmbeddingGemma 2 GGUF requires an xllamacpp build containing "
                "llama.cpp support for gemma-embedding2 (ggml-org/llama.cpp#30054) "
                "and multimodal /v1/embeddings content inputs. "
                "The xllamacpp 2026.9.11063 release does not include this architecture."
            ) from exc

    def _create_embedding(self, sentences: Any, **kwargs: Any) -> Embedding:
        dimensions = kwargs.pop("dimensions", self._dimensions)
        validate_dimensions(dimensions)
        if kwargs.pop("return_sparse", False):
            raise ValueError("EmbeddingGemma 2 does not support sparse embeddings.")
        sentences = self._fix_langchain_openai_inputs(sentences)
        inputs = prepare_inputs(sentences, **kwargs)
        if not inputs:
            return Embedding(
                object="list",
                model=kwargs.get("model_uid"),
                model_replica=self._model_uid,
                data=[],
                usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),
            )
        result = super()._create_embedding(inputs, **kwargs)
        normalize = kwargs.get(
            "normalize_embeddings", kwargs.get("normalize_embedding", True)
        )
        for item in result["data"]:
            vector = item["embedding"]
            if (
                not isinstance(vector, list)
                or len(vector) != self.model_family.dimensions
                or not all(math.isfinite(v) for v in vector)
            ):
                raise RuntimeError("EmbeddingGemma 2 GGUF returned invalid embeddings.")
            vector = vector[:dimensions]
            if normalize:
                norm = math.sqrt(math.fsum(v * v for v in vector))
                if norm:
                    vector = [v / norm for v in vector]
            item["embedding"] = vector
        return result

    @classmethod
    def match_json(
        cls,
        model_family: EmbeddingModelFamilyV2,
        model_spec: EmbeddingSpecV1,
        quantization: str,
    ) -> Union[bool, Tuple[bool, str]]:
        return (
            model_family.model_name == MODEL_NAME
            and model_spec.model_format == "ggufv2"
        )
