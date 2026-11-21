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

import os
from abc import abstractmethod
from importlib.metadata import PackageNotFoundError, requires, version
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging.requirements import Requirement
from packaging.version import Version
from urllib3.exceptions import HTTPError

from ...core.exceptions import InvalidModelRequestError
from ...device_utils import get_available_device
from ...types import Embedding, EmbeddingData, EmbeddingUsage
from ..batch import BatchMixin
from ..llm.media import load_media_bytes
from ..utils import check_dependency_available, virtual_env_allows_missing_engine
from .core import EmbeddingModel, EmbeddingModelFamilyV2, EmbeddingSpecV1
from .vllm.core import VLLMEmbeddingModel

MODEL_NAMES = {
    "Nemotron-3.5-Embed-1B-BF16",
    "Nemotron-3.5-Embed-1B-NVFP4",
    "Nemotron-3.5-Embed-8B-BF16",
}
_EXIF_ORIENTATION = 0x0112
_TRANSFORMERS_CHAT_TEMPLATE_RESERVED = frozenset(
    {"self", "conversation", "tokenize", "return_dict", "return_tensors"}
)
_TRANSFORMERS_PROCESSOR_OPTION_GROUPS = (
    "audio_kwargs",
    "common_kwargs",
    "images_kwargs",
    "text_kwargs",
    "videos_kwargs",
)


def check_dependencies(engine: str) -> Union[bool, Tuple[bool, str]]:
    requirements = (
        [("vllm", "0.32.0"), ("transformers", "5.17.0")]
        if engine == "vllm"
        # Transformers 5.20 imports torch.distributed.checkpoint.hf_storage.
        else [("torch", "2.8.0"), ("torchvision", "0.23.0"), ("transformers", "5.20.0")]
    )
    if engine == "sentence_transformers":
        requirements.append(("sentence-transformers", "5.4"))
    installed_versions = {}
    for package, minimum in requirements:
        try:
            installed = Version(version(package))
        except PackageNotFoundError:
            return False, f"Cannot determine the installed {package} version"
        exclusive = package == "sentence-transformers"
        if installed < Version(minimum) or (
            exclusive and installed == Version(minimum)
        ):
            operator = ">" if exclusive else ">="
            return (
                False,
                f"Nemotron 3.5 Embed requires {package}{operator}{minimum}, "
                f"current: {installed}",
            )
        if (
            engine == "vllm"
            and package == "transformers"
            and installed >= Version("5.20.0")
        ):
            return (
                False,
                "Nemotron 3.5 Embed vLLM requires transformers>=5.17.0,<5.20.0.",
            )
        installed_versions[package] = installed
        if package == "torchvision":
            for dependency in requires("torchvision") or []:
                requirement = Requirement(dependency)
                if (
                    requirement.name.lower() == "torch"
                    and (requirement.marker is None or requirement.marker.evaluate())
                    and installed_versions["torch"] not in requirement.specifier
                ):
                    return (
                        False,
                        f"Nemotron 3.5 Embed requires matching Torch and torchvision: "
                        f"torchvision {installed} requires {requirement}, "
                        f"current torch: {installed_versions['torch']}",
                    )
        available = check_dependency_available(package.replace("-", "_"), package)
        if available != True:
            return available
    return True


def normalize_inputs(inputs: Any, **kwargs: Any) -> Tuple[List[Dict[str, Any]], str]:
    """Normalize one text/image document per input and select its retrieval role."""
    input_type = kwargs.get("input_type")
    prompt_name = kwargs.get("prompt_name")
    aliases = {"query": "query", "document": "passage", "passage": "passage"}
    if input_type is not None and (
        not isinstance(input_type, str) or input_type not in aliases
    ):
        raise InvalidModelRequestError(
            "Nemotron 3.5 Embed input_type must be query or passage."
        )
    if prompt_name is not None and (
        not isinstance(prompt_name, str) or prompt_name not in aliases
    ):
        raise InvalidModelRequestError(
            "Nemotron 3.5 Embed prompt_name must be query or document."
        )
    role = aliases.get(input_type or prompt_name or "passage")
    if input_type and prompt_name and aliases[input_type] != aliases[prompt_name]:
        raise InvalidModelRequestError(
            "Nemotron 3.5 Embed input_type and prompt_name conflict."
        )
    items = [inputs] if isinstance(inputs, (str, dict)) else inputs
    if not isinstance(items, list):
        raise InvalidModelRequestError(
            "Nemotron 3.5 Embed input must be text, a dict, or a list."
        )
    samples = []
    for item in items:
        if isinstance(item, str):
            sample: Dict[str, Any] = {"text": item}
        elif isinstance(item, dict):
            if set(item) - {"text", "image", "image_url"}:
                raise InvalidModelRequestError(
                    "Nemotron 3.5 Embed supports only text and image input."
                )
            if "image" in item and "image_url" in item:
                raise InvalidModelRequestError("Provide image or image_url, not both.")
            sample = {key: value for key, value in item.items() if value is not None}
            if "image_url" in sample:
                image = sample.pop("image_url")
                sample["image"] = image.get("url") if isinstance(image, dict) else image
            if "text" in sample and not isinstance(sample["text"], str):
                raise InvalidModelRequestError(
                    "Nemotron 3.5 Embed text must be a string."
                )
            if "image" in sample:
                from PIL import Image

                image = sample["image"]
                if not isinstance(image, (str, Image.Image)) or (
                    isinstance(image, str) and not image
                ):
                    raise InvalidModelRequestError(
                        "Nemotron 3.5 Embed expects one image per document."
                    )
            if not sample:
                raise InvalidModelRequestError(
                    "Nemotron 3.5 Embed input must contain text or image."
                )
        else:
            raise InvalidModelRequestError(
                "Nemotron 3.5 Embed items must be strings or dicts."
            )
        if role == "query" and "image" in sample:
            raise InvalidModelRequestError(
                "Nemotron 3.5 Embed queries accept text only."
            )
        samples.append(sample)
    return samples, "query:" if role == "query" else "passage:"


def make_messages(sample: Dict[str, Any], prefix: str) -> List[Dict[str, Any]]:
    content = []
    if "image" in sample:
        content.append({"type": "image", "image": sample["image"]})
    if "text" in sample:
        content.append({"type": "text", "text": sample["text"]})
    return [
        {"role": "system", "content": prefix},
        {"role": "user", "content": content},
    ]


def _prepare_images(samples: List[Dict[str, Any]]) -> None:
    from PIL import Image, ImageOps

    for sample in samples:
        source = sample.get("image")
        try:
            if isinstance(source, str):
                with Image.open(BytesIO(load_media_bytes(source))) as image:
                    sample["image"] = ImageOps.exif_transpose(image).convert("RGB")
            elif isinstance(source, Image.Image):
                if (
                    source.mode != "RGB"
                    or source.getexif().get(_EXIF_ORIENTATION, 1) != 1
                ):
                    sample["image"] = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, ValueError, HTTPError, Image.DecompressionBombError) as exc:
            raise InvalidModelRequestError(
                f"Cannot load Nemotron 3.5 Embed image: {exc}"
            ) from exc


class _NemotronEmbeddingMixin(EmbeddingModel):
    engine: str
    model_family: EmbeddingModelFamilyV2
    _model: Any
    _device: Optional[str]

    @abstractmethod
    def _encode_samples(
        self, samples: List[Dict[str, Any]], prefix: str, **kwargs: Any
    ) -> Tuple[List[torch.Tensor], List[int]]:
        pass

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
            model_family.model_name not in MODEL_NAMES
            or model_spec.model_format != "pytorch"
        ):
            return False
        if model_family.model_name.endswith("NVFP4") and cls.engine != "vllm":
            return (
                False,
                "Nemotron 3.5 Embed NVFP4 requires vLLM and NVFP4-capable GPUs.",
            )
        return True if virtual_env_allows_missing_engine() else cls.check_lib()

    def _validate_dimensions(self, dimensions: Optional[int]) -> None:
        if dimensions is not None and (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
            or not 0 < dimensions <= self.model_family.dimensions
        ):
            raise InvalidModelRequestError(
                "Nemotron 3.5 Embed dimensions must be a positive integer up to "
                f"{self.model_family.dimensions}."
            )

    def _validate_context_length(self, length: int) -> None:
        if length > self.model_family.max_tokens:
            raise InvalidModelRequestError(
                f"Nemotron 3.5 Embed input exceeds {self.model_family.max_tokens} "
                "tokens; shorten text or resize the image."
            )

    def _validate_request_options(self, kwargs: Dict[str, Any]) -> Optional[int]:
        dimensions = kwargs.get("dimensions", self._kwargs.get("dimensions"))
        self._validate_dimensions(dimensions)
        if kwargs.get("return_sparse", False):
            raise InvalidModelRequestError(
                "Nemotron 3.5 Embed does not support sparse embeddings."
            )
        processing_kwargs = kwargs.get("processing_kwargs")
        if processing_kwargs is not None and not isinstance(processing_kwargs, dict):
            raise InvalidModelRequestError(
                "Nemotron 3.5 Embed processing_kwargs must be an object or null."
            )
        if self.engine == "transformers" and processing_kwargs:
            reserved = _TRANSFORMERS_CHAT_TEMPLATE_RESERVED.intersection(
                processing_kwargs
            )
            if reserved:
                raise InvalidModelRequestError(
                    "Nemotron 3.5 Embed does not accept reserved processing_kwargs: "
                    + ", ".join(sorted(reserved))
                    + "."
                )
            processor_kwargs = processing_kwargs.get("processor_kwargs")
            if processor_kwargs is not None and not isinstance(processor_kwargs, dict):
                raise InvalidModelRequestError(
                    "Nemotron 3.5 Embed processing_kwargs.processor_kwargs "
                    "must be an object or null."
                )
            for path, options in (
                ("processing_kwargs", processing_kwargs),
                ("processing_kwargs.processor_kwargs", processor_kwargs or {}),
            ):
                for group in _TRANSFORMERS_PROCESSOR_OPTION_GROUPS:
                    if group in options and not isinstance(options[group], dict):
                        raise InvalidModelRequestError(
                            f"Nemotron 3.5 Embed {path}.{group} must be an object."
                        )
        return dimensions

    def _prepare_batch_inputs(self, sentences: List[Any], **kwargs: Any) -> List[Any]:
        # Validate before grouping: Python treats True and 1 as equal batch keys.
        self._validate_request_options(kwargs)
        # Leave text and token arrays to the usual truncation and decoding path.
        if not isinstance(sentences, list) or not any(
            isinstance(item, dict) for item in sentences
        ):
            return sentences
        samples, _ = normalize_inputs(sentences, **kwargs)
        _prepare_images(samples)
        return samples

    def _create_embedding(self, sentences: Any, **kwargs: Any) -> Embedding:
        sentences = self._fix_langchain_openai_inputs(sentences)
        samples, prefix = normalize_inputs(sentences, **kwargs)
        dimensions = self._validate_request_options(kwargs)
        _prepare_images(samples)
        vectors, counts = (
            self._encode_samples(samples, prefix, **kwargs) if samples else ([], [])
        )
        normalize = kwargs.get(
            "normalize_embeddings", kwargs.get("normalize_embedding", True)
        )
        data = []
        for index, vector in enumerate(vectors):
            vector = vector.float()[:dimensions]
            if normalize:
                vector = torch.nn.functional.normalize(vector, p=2, dim=0)
            data.append(
                EmbeddingData(
                    index=index, object="embedding", embedding=vector.tolist()
                )
            )
        tokens = sum(counts)
        self._clean_cache_if_needed(tokens)
        return self._with_token_counts(
            Embedding(
                object="list",
                model=kwargs.get("model_uid"),
                model_replica=self._model_uid,
                data=data,
                usage=EmbeddingUsage(prompt_tokens=tokens, total_tokens=tokens),
            ),
            counts,
        )


class TransformersNemotronEmbeddingModel(_NemotronEmbeddingMixin, BatchMixin):
    replica_autoscaling = "batch"
    engine = "transformers"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        EmbeddingModel.__init__(self, *args, **kwargs)
        BatchMixin.__init__(self, self.create_embedding, **kwargs)  # type: ignore
        self._validate_dimensions(self._kwargs.get("dimensions"))

    def load(self) -> None:
        dependencies = self.check_lib()
        if dependencies != True:
            raise ImportError(dependencies[1])  # type: ignore[index]
        from transformers import AutoModel, AutoProcessor

        self._device = self._device or get_available_device()
        model_kwargs = dict(self._kwargs.get("model_kwargs", {}))
        model_kwargs.setdefault("dtype", self._kwargs.get("torch_dtype", "auto"))
        model_kwargs.setdefault("device_map", self._device)
        self._processor = AutoProcessor.from_pretrained(self._model_path)
        self._tokenizer = self._processor.tokenizer
        self._model = AutoModel.from_pretrained(self._model_path, **model_kwargs).eval()

    def _encode_samples(
        self, samples: List[Dict[str, Any]], prefix: str, **kwargs: Any
    ):
        vectors, counts = [], []
        processing_kwargs = kwargs.get("processing_kwargs") or {}
        for sample in samples:
            messages = make_messages(sample, prefix)
            try:
                features = self._processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    **processing_kwargs,
                )
            except (TypeError, ValueError) as exc:
                if not processing_kwargs:
                    raise
                raise InvalidModelRequestError(
                    f"Invalid Nemotron 3.5 Embed processing_kwargs: {exc}"
                ) from exc
            self._validate_context_length(features["input_ids"].shape[-1])
            if features.get("attention_mask") is None:
                raise InvalidModelRequestError(
                    "Nemotron 3.5 Embed requires return_attention_mask=True."
                )
            features = features.to(self._model.device)
            if "pixel_values" in features:
                features["pixel_values"] = features["pixel_values"].to(
                    self._model.dtype
                )
            with torch.inference_mode():
                hidden = self._model(**features).last_hidden_state.float()
                mask = features["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors.append(pooled[0])
            counts.append(int(features["attention_mask"].sum().item()))
        return vectors, counts


class SentenceTransformerNemotronEmbeddingModel(TransformersNemotronEmbeddingModel):
    engine = "sentence_transformers"

    def load(self) -> None:
        dependencies = self.check_lib()
        if dependencies != True:
            raise ImportError(dependencies[1])  # type: ignore[index]
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.models import Normalize

        self._device = self._device or get_available_device()
        model_kwargs = dict(self._kwargs.get("model_kwargs", {}))
        model_kwargs.setdefault("dtype", self._kwargs.get("torch_dtype", "auto"))
        self._model = SentenceTransformer(
            self._model_path, device=self._device, model_kwargs=model_kwargs
        )
        # Leave pooled vectors raw so the adapter can truncate dimensions
        # and then apply the caller's normalization option.
        if isinstance(self._model[-1], Normalize):
            del self._model[-1]
        self._model.max_seq_length = self.model_family.max_tokens
        self._tokenizer = self._model.tokenizer

    def _encode_samples(
        self, samples: List[Dict[str, Any]], prefix: str, **kwargs: Any
    ):
        vectors, counts = [], []

        def count_tokens(module: Any, args: Tuple[Any, ...]) -> None:
            mask = args[0]["attention_mask"]
            self._validate_context_length(mask.shape[-1])
            counts.append(int(mask.sum().item()))

        hook = self._model.register_forward_pre_hook(count_tokens)
        try:
            # The model card uses batch_size=1 for mixed image/text documents.
            # Each forward then supplies the exact token count for its input.
            for sample in samples:
                with torch.inference_mode():
                    vector = self._model.encode(
                        [sample],
                        prompt_name="query" if prefix == "query:" else "document",
                        batch_size=1,
                        convert_to_tensor=True,
                        convert_to_numpy=False,
                        show_progress_bar=False,
                        processing_kwargs=kwargs.get("processing_kwargs"),
                    )
                vectors.append(vector[0])
        finally:
            hook.remove()
        return vectors, counts


class VLLMNemotronEmbeddingModel(_NemotronEmbeddingMixin, VLLMEmbeddingModel):
    replica_autoscaling = "batch"
    engine = "vllm"

    def load(self) -> None:
        dependencies = self.check_lib()
        if dependencies != True:
            raise ImportError(dependencies[1])  # type: ignore[index]
        # Xinference loads models in a thread pool. Forking that thread can
        # inherit locked PyTorch/OpenMP state and hang while cloning weights.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from transformers import AutoProcessor
        from vllm import LLM
        from vllm.config import PoolerConfig

        self._validate_dimensions(self._kwargs.get("dimensions"))
        options = dict(self._kwargs)
        for key in ("batch_size", "batch_interval", "dimensions"):
            options.pop(key, None)
        options.setdefault("runner", "pooling")
        options.setdefault("max_model_len", self.model_family.max_tokens)
        options.setdefault(
            "pooler_config",
            PoolerConfig(task="embed", pooling_type="MEAN", use_activation=True),
        )
        if self.model_family.model_name.endswith("NVFP4"):
            options.setdefault("compilation_config", {"compile_mm_encoder": True})
        self._processor = AutoProcessor.from_pretrained(self._model_path)
        self._model = LLM(model=self._model_path, **options)
        self._tokenizer = self._model.get_tokenizer()
        if self._pooling_load is not None:
            self._pooling_load.configure_sync(self._model)

    def _encode_samples(
        self, samples: List[Dict[str, Any]], prefix: str, **kwargs: Any
    ):
        from vllm import PoolingParams
        from vllm.exceptions import VLLMValidationError

        prompts = []
        for sample in samples:
            prompt = {
                "prompt": self._processor.apply_chat_template(
                    make_messages(sample, prefix), tokenize=False
                )
            }
            if "image" in sample:
                prompt["multi_modal_data"] = {"image": sample["image"]}
            if kwargs.get("processing_kwargs"):
                prompt["mm_processor_kwargs"] = kwargs["processing_kwargs"]
            prompts.append(prompt)
        try:
            outputs = self._embed_sync(
                prompts,
                pooling_params=PoolingParams(use_activation=False),
                tokenization_kwargs={"add_special_tokens": False},
            )
        except VLLMValidationError as exc:
            raise InvalidModelRequestError(str(exc)) from exc
        return (
            [torch.tensor(output.outputs.embedding) for output in outputs],
            [len(output.prompt_token_ids) for output in outputs],
        )
