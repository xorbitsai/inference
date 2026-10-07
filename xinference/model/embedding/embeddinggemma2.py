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

import json
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging.version import Version

from ...device_utils import get_available_device
from ...types import Embedding, EmbeddingData, EmbeddingUsage
from ..batch import BatchMixin
from ..utils import check_dependency_available, virtual_env_allows_missing_engine
from .core import EmbeddingModel, EmbeddingModelFamilyV2, EmbeddingSpecV1

MODEL_NAME = "embeddinggemma-2"
MATRYOSHKA_DIMENSIONS = (128, 256, 512, 768)


def validate_dimensions(dimensions: Optional[int]) -> None:
    if dimensions is not None and (
        not isinstance(dimensions, int)
        or isinstance(dimensions, bool)
        or dimensions not in MATRYOSHKA_DIMENSIONS
    ):
        raise ValueError(
            f"EmbeddingGemma 2 dimensions must be one of {MATRYOSHKA_DIMENSIONS}, "
            f"got {dimensions}."
        )


def resolve_dtype(value: Any, device: str) -> torch.dtype:
    if value is None or value == "auto":
        return (
            torch.bfloat16
            if device.startswith("cuda") and torch.cuda.is_bf16_supported()
            else torch.float32
        )
    if value in ("bfloat16", "bf16", torch.bfloat16):
        return torch.bfloat16
    if value in ("float32", "fp32", torch.float32):
        return torch.float32
    raise ValueError(
        "EmbeddingGemma 2 requires bfloat16 or float32; float16 produces invalid embeddings."
    )


def check_dependencies(engine: str) -> Union[bool, Tuple[bool, str]]:
    requirements = [("transformers", "5.19.0")]
    if engine == "sentence_transformers":
        from ..utils import neutralize_broken_torchcodec

        neutralize_broken_torchcodec()
        requirements.append(("sentence-transformers", "6.1.0"))
    for package, minimum in requirements:
        module = package.replace("-", "_")
        available = check_dependency_available(module, package)
        if available != True:
            return available
        try:
            installed = version(package)
        except PackageNotFoundError:
            return False, f"Cannot determine the installed {package} version"
        if Version(installed) < Version(minimum):
            return (
                False,
                f"EmbeddingGemma 2 requires {package}>={minimum}, current: {installed}",
            )
    return True


def load_prompts(model_path: str) -> Dict[str, str]:
    with open(
        os.path.join(model_path, "config_sentence_transformers.json"), encoding="utf-8"
    ) as file:
        return json.load(file).get("prompts", {})


def normalize_inputs(
    inputs: Any, prompts: Dict[str, str], **kwargs: Any
) -> List[List[Dict[str, Any]]]:
    """Keep each input as one ordered, possibly interleaved multimodal sample."""
    prompt = kwargs.get("prompt")
    prompt_name = kwargs.get("prompt_name")
    if prompt is None and prompt_name is not None:
        if prompt_name not in prompts:
            raise ValueError(f"Unknown EmbeddingGemma 2 prompt_name: {prompt_name}")
        prompt = prompts[prompt_name]
    items = [inputs] if isinstance(inputs, (str, dict)) else inputs
    if not isinstance(items, list):
        raise ValueError(
            "EmbeddingGemma 2 input must be text, a multimodal dict, or a list of these."
        )
    result = []
    for item in items:
        if isinstance(item, str):
            content = [{"type": "text", "text": item}]
        elif isinstance(item, dict):
            content = []
            for modality, value in item.items():
                if modality not in ("text", "image", "video", "audio"):
                    raise ValueError(
                        f"Unsupported EmbeddingGemma 2 input field: {modality}"
                    )
                if value is None:
                    continue
                values = value if isinstance(value, list) else [value]
                for part in values:
                    if modality == "text" and not isinstance(part, str):
                        raise ValueError("EmbeddingGemma 2 text must be a string.")
                    content.append({"type": modality, modality: part})
            if not content:
                raise ValueError(
                    "EmbeddingGemma 2 input dict must contain text, image, video, or audio."
                )
        else:
            raise ValueError(
                "EmbeddingGemma 2 input items must be strings or multimodal dicts."
            )
        # Prefixes belong to text. Media-only inputs must not acquire a task prefix.
        if prompt:
            for part in content:
                if part["type"] == "text":
                    part["text"] = prompt + part["text"]
                    break
        result.append([{"role": "user", "content": content}])
    return result


class TransformersEmbeddingGemma2Model(EmbeddingModel, BatchMixin):
    engine = "transformers"
    _device: Optional[str]

    def __init__(self, *args, **kwargs) -> None:
        EmbeddingModel.__init__(self, *args, **kwargs)
        BatchMixin.__init__(self, self.create_embedding, **kwargs)  # type: ignore
        self._dimensions = self._kwargs.get("dimensions")
        validate_dimensions(self._dimensions)
        self._prompts: Dict[str, str] = {}

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        return check_dependencies(cls.engine)

    @classmethod
    def match_json(
        cls,
        model_family: EmbeddingModelFamilyV2,
        model_spec: EmbeddingSpecV1,
        quantization: str,
    ) -> Union[bool, Tuple[bool, str]]:
        if (
            model_family.model_name != MODEL_NAME
            or model_spec.model_format != "pytorch"
        ):
            return False
        return True if virtual_env_allows_missing_engine() else cls.check_lib()

    def load(self) -> None:
        dependencies = self.check_lib()
        if dependencies != True:
            raise ImportError(dependencies[1])  # type: ignore[index]
        from transformers import AutoModel, AutoProcessor

        self._device = self._device or get_available_device()
        dtype = resolve_dtype(self._kwargs.get("torch_dtype"), self._device)
        self._processor = AutoProcessor.from_pretrained(self._model_path)
        self._tokenizer = self._processor.tokenizer
        self._model = (
            AutoModel.from_pretrained(
                self._model_path,
                dtype=dtype,
                **self._kwargs.get("config_kwargs", {}),
            )
            .to(self._device)
            .eval()
        )
        self._prompts = load_prompts(self._model_path)

    def _encode(self, messages: List[List[Dict[str, Any]]], **kwargs: Any):
        embeddings = []
        tokens = 0
        assert self._model is not None
        # Media have variable token lengths. One sample at a time avoids padding
        # mixed image/video/audio tensors and still uses Xinference request batching.
        for sample in messages:
            features = self._processor.apply_chat_template(
                sample,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={
                    "truncation": True,
                    "max_length": self.model_family.max_tokens,
                    **kwargs.get("processing_kwargs", {}),
                },
            ).to(device=self._device, dtype=self._model.dtype)
            with torch.inference_mode():
                hidden = self._model(**features).last_hidden_state.float()
                mask = features["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            embeddings.append(pooled[0])
            tokens += int(features["attention_mask"].sum().item())
        return embeddings, tokens

    def _create_embedding(self, sentences: Any, **kwargs: Any) -> Embedding:
        sentences = self._fix_langchain_openai_inputs(sentences)
        dimensions = kwargs.pop("dimensions", self._dimensions)
        validate_dimensions(dimensions)
        if kwargs.pop("return_sparse", False):
            raise ValueError("EmbeddingGemma 2 does not support sparse embeddings.")
        messages = normalize_inputs(sentences, self._prompts, **kwargs)
        embeddings, tokens = self._encode(messages, **kwargs) if messages else ([], 0)
        normalize = kwargs.get(
            "normalize_embeddings", kwargs.get("normalize_embedding", True)
        )
        data = []
        for index, vector in enumerate(embeddings):
            vector = vector.float()[:dimensions]
            if normalize:
                vector = torch.nn.functional.normalize(vector, p=2, dim=0)
            data.append(
                EmbeddingData(
                    index=index, object="embedding", embedding=vector.tolist()
                )
            )
        self._clean_cache_if_needed(tokens)
        return Embedding(
            object="list",
            model=kwargs.get("model_uid"),
            model_replica=self._model_uid,
            data=data,
            usage=EmbeddingUsage(prompt_tokens=tokens, total_tokens=tokens),
        )


class SentenceTransformerEmbeddingGemma2Model(TransformersEmbeddingGemma2Model):
    engine = "sentence_transformers"

    def load(self) -> None:
        dependencies = self.check_lib()
        if dependencies != True:
            raise ImportError(dependencies[1])  # type: ignore[index]
        from sentence_transformers import SentenceTransformer

        self._device = self._device or get_available_device()
        dtype = resolve_dtype(self._kwargs.get("torch_dtype"), self._device)
        self._model = SentenceTransformer(
            self._model_path,
            device=self._device,
            model_kwargs={"dtype": dtype},
            config_kwargs=self._kwargs.get("config_kwargs"),
        )
        self._model.max_seq_length = self.model_family.max_tokens
        self._tokenizer = self._model.tokenizer
        self._prompts = self._model.prompts

    def _encode(self, messages: List[List[Dict[str, Any]]], **kwargs: Any):
        assert self._model is not None
        tokens = 0

        def count_tokens(module: Any, args: Tuple[Any, ...]) -> None:
            nonlocal tokens
            tokens += int(args[0]["attention_mask"].sum().item())

        # Native encode owns modality routing and preserves batch order. Asking
        # it for all features fails on mixed batches because media tensors have
        # a different leading dimension; count the expanded token mask instead.
        hook = self._model.register_forward_pre_hook(count_tokens)
        try:
            with torch.inference_mode():
                outputs = self._model.encode(
                    messages,
                    prompt="",
                    convert_to_tensor=True,
                    convert_to_numpy=False,
                    batch_size=kwargs.get("batch_size", 1),
                    processing_kwargs=kwargs.get("processing_kwargs"),
                )
        finally:
            hook.remove()
        return list(outputs), tokens
