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
"""Input and dependency contracts for Nemotron 3.5 multimodal reranking."""

import importlib
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

from packaging.version import Version

from ..llm import media
from ..utils import check_dependency_available, virtual_env_allows_missing_engine

NEMOTRON_RERANK_MODEL_NAME = "Nemotron-3.5-Rerank-8B"


def validate_nemotron_rerank_options(max_chunks_per_doc: Optional[int]) -> None:
    if max_chunks_per_doc is not None:
        raise ValueError("rerank hasn't support `max_chunks_per_doc` parameter.")


def check_nemotron_dependencies(engine: str) -> Union[bool, Tuple[bool, str]]:
    if virtual_env_allows_missing_engine():
        return True
    requirements = (
        [("vllm", "vllm", "0.32.0")]
        if engine == "vllm"
        else [("transformers", "transformers", "5.20.0")]
    )
    if engine == "sentence_transformers":
        requirements.append(("sentence_transformers", "sentence-transformers", "6.1.0"))
    for module_name, package, minimum in requirements:
        result = check_dependency_available(module_name, package)
        if result is not True:
            return result
        module = importlib.import_module(module_name)
        if Version(module.__version__) < Version(minimum):
            return False, f"Nemotron reranker requires {package}>={minimum}"
    if engine == "transformers":
        result = check_dependency_available("accelerate", "accelerate")
        if result is not True:
            return result
    return check_dependency_available("PIL", "Pillow")


def nemotron_content(value: Any, *, query: bool = False) -> List[Dict[str, Any]]:
    """Accept existing rerank dictionaries and OpenAI content blocks."""
    if isinstance(value, str):
        content = [{"type": "text", "text": value}]
    elif isinstance(value, dict):
        if "content" in value:
            content = value["content"]
        else:
            content = []
            if value.get("text") is not None:
                content.append({"type": "text", "text": value["text"]})
            for key in ("image", "image_url"):
                if value.get(key) is not None:
                    content.append({"type": "image", "image": value[key]})
            if any(
                key in value for key in ("video", "video_url", "audio", "audio_url")
            ):
                raise ValueError(
                    "Nemotron reranker supports only text and image documents"
                )
    else:
        raise ValueError("Nemotron reranker inputs must be strings or dictionaries")
    if not isinstance(content, list) or not content:
        raise ValueError("Nemotron reranker content must be a non-empty list")
    normalized = []
    images = 0
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("Nemotron reranker content items must be dictionaries")
        kind = block.get("type")
        if kind == "text":
            if not isinstance(block.get("text"), str):
                raise ValueError("Nemotron reranker text must be a string")
            normalized.append({"type": "text", "text": block["text"]})
        elif kind in ("image", "image_url") and not query:
            image = block.get(kind)
            if isinstance(image, dict):
                image = image.get("url")
            if not isinstance(image, str) or not image.startswith(
                ("http://", "https://", "data:image/")
            ):
                raise ValueError(
                    "Nemotron reranker image must be an image URL or data URL"
                )
            images += 1
            normalized.append({"type": "image", "image": image})
        else:
            raise ValueError(
                "Nemotron reranker requires text queries and text/image documents"
            )
    if images > 1:
        raise ValueError("Nemotron reranker supports at most one image per document")
    return normalized


@dataclass
class _PreparedDocument:
    original: Any
    content: List[Dict[str, Any]]


def _prepare_document(document: Any) -> _PreparedDocument:
    if isinstance(document, _PreparedDocument):
        return document
    content = nemotron_content(document)
    for block in content:
        if block["type"] == "image":
            from PIL import Image, ImageOps

            data = media.load_media_bytes(block["image"])
            with Image.open(BytesIO(data)) as image:
                block["image"] = ImageOps.exif_transpose(image).convert("RGB")
    return _PreparedDocument(document, content)


def prepare_nemotron_request(
    documents: List[Any], query: Any
) -> List[_PreparedDocument]:
    """Prepare one request before adding any of its documents to a batch."""
    if documents:
        nemotron_content(query, query=True)
    return [_prepare_document(document) for document in documents]


def nemotron_messages(query: Any, document: Any) -> List[Dict[str, Any]]:
    return [
        {"role": "query", "content": nemotron_content(query, query=True)},
        {
            "role": "document",
            "content": [dict(block) for block in _prepare_document(document).content],
        },
    ]
