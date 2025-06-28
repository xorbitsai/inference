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

import abc
import gc
import logging
import os
from abc import abstractmethod
from collections import defaultdict
from typing import Annotated, Dict, Iterable, List, Literal, Optional, Tuple, Union

from ..._compat import ROOT_KEY, BaseModel, ErrorWrapper, Field, ValidationError
from ...constants import XINFERENCE_CACHE_DIR
from ...device_utils import empty_cache
from ..core import CacheableQuantModelSpec, ModelDescription, VirtualEnvSettings
from ..utils import (
    IS_NEW_HUGGINGFACE_HUB,
    create_symlink,
    generate_quant_model_file_names,
    retry_download,
    symlink_local_file,
    valid_model_revision,
)
from .embed_family import (
    BUILTIN_EMBEDDING_MODELS,
    BUILTIN_MODELSCOPE_EMBEDDING_MODELS,
    match_embedding,
)

logger = logging.getLogger(__name__)

# Used for check whether the model is cached.
# Init when registering all the builtin models.
EMBEDDING_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)
EMBEDDING_EMPTY_CACHE_COUNT = int(
    os.getenv("XINFERENCE_EMBEDDING_EMPTY_CACHE_COUNT", "10")
)
EMBEDDING_EMPTY_CACHE_TOKENS = int(
    os.getenv("XINFERENCE_EMBEDDING_EMPTY_CACHE_TOKENS", "8192")
)
assert EMBEDDING_EMPTY_CACHE_COUNT > 0
assert EMBEDDING_EMPTY_CACHE_TOKENS > 0


def get_embedding_model_descriptions():
    import copy

    return copy.deepcopy(EMBEDDING_MODEL_DESCRIPTIONS)


class TransformersEmbeddingSpecV1(BaseModel):
    model_format: Literal["transformers"]
    # Must in order that `str` first, then `int`
    model_id: Optional[str]
    model_revision: Optional[str]
    quantizations: List[str]
    virtualenv: Optional[VirtualEnvSettings]


class LlamaCppEmbeddingSpecV1(CacheableQuantModelSpec):
    model_format: Literal["ggufv2"]
    model_id: str
    model_revision: Optional[str]
    quantizations: List[str]
    model_file_name_template: str
    quantization_parts: Optional[Dict[str, List[str]]]


EmbeddingSpecV1 = Annotated[
    Union[TransformersEmbeddingSpecV1, LlamaCppEmbeddingSpecV1],
    Field(discriminator="model_format"),
]


# this class define the basic info of embedding model
class EmbeddingModelFamilyV1(BaseModel):
    model_name: str
    dimensions: int
    max_tokens: int
    language: List[str]
    model_specs: List["EmbeddingSpecV1"]
    model_hub: str = "huggingface"


class EmbeddingModelDescription(ModelDescription):
    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        model_family: EmbeddingModelFamilyV1,
        model_spec: EmbeddingSpecV1,
        quantization: Optional[str],
        model_path: Optional[str] = None,
    ):
        super().__init__(address, devices, model_path=model_path)
        self._model_family = model_family
        self._model_spec = model_spec
        self._quantization = quantization

    @property
    def spec(self):
        return self._model_family

    def to_dict(self):
        return {
            "model_type": "embedding",
            "address": self.address,
            "accelerators": self.devices,
            "model_name": self._model_family.model_name,
            "dimensions": self._model_family.dimensions,
            "max_tokens": self._model_family.max_tokens,
            "language": self._model_family.language,
            "model_revision": self._model_family.model_revision,
        }

    def to_version_info(self):
        file_location, is_cached = get_file_location(
            self._model_family, self._model_spec, self._quantization
        )
        return {
            "model_version": get_model_version(
                self._model_family, self._model_spec, self._quantization
            ),
            "model_file_location": file_location,
            "cache_status": is_cached,
            "dimensions": self._model_family.dimensions,
            "max_tokens": self._model_family.max_tokens,
        }


def get_model_version(
    embedding_model: EmbeddingModelFamilyV1,
    embedding_spec: EmbeddingSpecV1,
    quantization: str,
) -> str:
    return f"{embedding_model.model_name}--{embedding_model.max_tokens}--{embedding_model.dimensions}--{embedding_spec.model_format}--{quantization}"


def get_file_location(
    model_family: EmbeddingModelFamilyV1,
    spec: EmbeddingSpecV1,
    quantization: str,
) -> Tuple[str, bool]:
    cache_dir = _get_cache_dir(
        model_family, spec, quantization, create_if_not_exist=False
    )
    cache_status = get_cache_status(model_family, spec, quantization)
    if isinstance(cache_status, list):
        is_cached = None
        for q, cs in zip(spec.quantizations, cache_status):
            if q == quantization:
                is_cached = cs
                break
    else:
        is_cached = cache_status
    assert isinstance(is_cached, bool)

    if spec.model_format in ["transformers"]:
        return cache_dir, is_cached
    elif spec.model_format in ["ggufv2"]:
        assert isinstance(spec, LlamaCppEmbeddingSpecV1)
        filename = spec.model_file_name_template.format(quantization=quantization)
        model_path = os.path.join(cache_dir, filename)
        return model_path, is_cached
    else:
        raise ValueError(f"Not supported model format {spec.model_format}")


def _get_cache_dir(
    model_family: EmbeddingModelFamilyV1,
    model_spec: EmbeddingSpecV1,
    quantization: Optional[str] = None,
    create_if_not_exist=True,
):
    # If the model id contains quantization, then we should give each
    # quantization a dedicated cache dir.
    quant_suffix = ""
    if model_spec.model_id and "{" in model_spec.model_id and quantization is not None:
        quant_suffix = quantization
    else:
        for q in model_spec.quantizations:
            if model_spec.model_id and q in model_spec.model_id:
                quant_suffix = q
                break

    # some model name includes ".", e.g. qwen1.5-chat
    # if the model does not require trust_remote_code, it's OK
    # because no need to import modeling_xxx.py from the path
    # but when the model need to trust_remote_code,
    # e.g. internlm2.5-chat, the import will fail,
    # but before the model may have been downloaded,
    # thus we check it first, if exist, return it,
    # otherwise, we replace the "." with "_" in model name
    old_cache_dir_name = f"{model_family.model_name}-{model_spec.model_format}"
    if quant_suffix:
        old_cache_dir_name += f"-{quant_suffix}"
    old_cache_dir = os.path.realpath(
        os.path.join(XINFERENCE_CACHE_DIR, old_cache_dir_name)
    )
    if os.path.exists(old_cache_dir):
        return old_cache_dir
    else:
        cache_dir_name = (
            f"{model_family.model_name.replace('.', '_')}-{model_spec.model_format}"
        )
        if quant_suffix:
            cache_dir_name += f"-{quant_suffix}"
        cache_dir = os.path.realpath(os.path.join(XINFERENCE_CACHE_DIR, cache_dir_name))
        if create_if_not_exist and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return cache_dir


def _get_meta_path(
    cache_dir: str,
    model_format: str,
    model_hub: str,
    quantization: Optional[str] = None,
):
    if model_format == "transformers":
        return os.path.join(cache_dir, f"__valid_download_{model_hub}")
    elif model_format == "ggufv2":
        assert quantization is not None
        return os.path.join(cache_dir, f"__valid_download_{model_hub}_{quantization}")
    else:
        raise ValueError(f"Unsupported format: {model_format}")


def _generate_meta_file(
    meta_path: str,
    llm_family: EmbeddingModelFamilyV1,
    llm_spec: EmbeddingSpecV1,
    quantization: Optional[str] = None,
):
    assert not valid_model_revision(
        meta_path, llm_spec.model_revision
    ), f"meta file {meta_path} should not be valid"
    with open(meta_path, "w") as f:
        import json

        desc = EmbeddingModelDescription(None, None, llm_family, llm_spec, quantization)
        json.dump(desc.to_dict(), f)


def _skip_download(
    cache_dir: str,
    model_format: str,
    model_hub: str,
    model_revision: Optional[str],
    quantization: Optional[str] = None,
) -> bool:
    if model_format in ["transformers"]:
        model_hub_to_meta_path = {
            "huggingface": _get_meta_path(
                cache_dir, model_format, "huggingface", quantization
            ),
            "modelscope": _get_meta_path(
                cache_dir, model_format, "modelscope", quantization
            ),
        }
        if valid_model_revision(model_hub_to_meta_path[model_hub], model_revision):
            logger.info(f"Cache {cache_dir} exists")
            return True
        else:
            for hub, meta_path in model_hub_to_meta_path.items():
                if hub != model_hub and os.path.exists(meta_path):
                    # PyTorch models from modelscope can also be loaded by transformers.
                    logger.warning(f"Cache {cache_dir} exists, but it was from {hub}")
                    return True
            return False
    elif model_format == "ggufv2":
        assert quantization is not None
        return os.path.exists(
            _get_meta_path(cache_dir, model_format, model_hub, quantization)
        )
    else:
        raise ValueError(f"Unsupported format: {model_format}")


def generate_embedding_description(
    model_family: EmbeddingModelFamilyV1,
) -> Dict[str, List[Dict]]:
    res = defaultdict(list)
    for spec in model_family.model_specs:
        for q in spec.quantizations:
            res[model_family.model_name].append(
                EmbeddingModelDescription(
                    None, None, model_family, spec, q
                ).to_version_info()
            )
    return res


def cache_from_modelscope(
    llm_family: EmbeddingModelFamilyV1,
    llm_spec: EmbeddingSpecV1,
    quantization: Optional[str] = None,
) -> str:
    """
    Cache model from Modelscope. Return the cache directory.
    """
    from modelscope.hub.file_download import model_file_download
    from modelscope.hub.snapshot_download import snapshot_download

    cache_dir = _get_cache_dir(llm_family, llm_spec)
    if _skip_download(
        cache_dir,
        llm_spec.model_format,
        llm_spec.model_hub,
        llm_spec.model_revision,
        quantization,
    ):
        return cache_dir

    if llm_spec.model_format in ["transformers"]:
        download_dir = retry_download(
            snapshot_download,
            llm_family.model_name,
            {"model_format": llm_spec.model_format},
            llm_spec.model_id,
            revision=llm_spec.model_revision,
        )
        create_symlink(download_dir, cache_dir)

    elif llm_spec.model_format in ["ggufv2"]:
        file_names, final_file_name, need_merge = generate_quant_model_file_names(
            llm_spec, quantization
        )

        for filename in file_names:
            download_path = retry_download(
                model_file_download,
                llm_family.model_name,
                {"model_format": llm_spec.model_format},
                llm_spec.model_id,
                filename,
                revision=llm_spec.model_revision,
            )
            symlink_local_file(download_path, cache_dir, filename)
    else:
        raise ValueError(f"Unsupported format: {llm_spec.model_format}")

    meta_path = _get_meta_path(
        cache_dir,
        llm_spec.model_format,
        llm_spec.model_hub,
        quantization,
    )
    _generate_meta_file(meta_path, llm_family, llm_spec, quantization)

    return cache_dir


def cache_from_huggingface(
    llm_family: EmbeddingModelFamilyV1,
    llm_spec: EmbeddingSpecV1,
    quantization: Optional[str] = None,
) -> str:
    """
    Cache model from Hugging Face. Return the cache directory.
    """
    import huggingface_hub

    cache_dir = _get_cache_dir(llm_family, llm_spec)
    if _skip_download(
        cache_dir,
        llm_spec.model_format,
        llm_spec.model_hub,
        llm_spec.model_revision,
        quantization,
    ):
        return cache_dir

    use_symlinks = {}
    if not IS_NEW_HUGGINGFACE_HUB:
        use_symlinks = {"local_dir_use_symlinks": True, "local_dir": cache_dir}

    if llm_spec.model_format in ["transformers"]:
        download_dir = retry_download(
            huggingface_hub.snapshot_download,
            llm_family.model_name,
            {
                "model_format": llm_spec.model_format,
            },
            llm_spec.model_id,
            revision=llm_spec.model_revision,
            **use_symlinks,
        )
        if IS_NEW_HUGGINGFACE_HUB:
            create_symlink(download_dir, cache_dir)

    elif llm_spec.model_format in ["ggufv2"]:
        assert isinstance(llm_spec, LlamaCppEmbeddingSpecV1)
        file_names, final_file_name, need_merge = generate_quant_model_file_names(
            llm_spec,
            quantization,
        )

        for file_name in file_names:
            download_file_path = retry_download(
                huggingface_hub.hf_hub_download,
                llm_family.model_name,
                {
                    "model_format": llm_spec.model_format,
                },
                llm_spec.model_id,
                revision=llm_spec.model_revision,
                filename=file_name,
                **use_symlinks,
            )
            if IS_NEW_HUGGINGFACE_HUB:
                symlink_local_file(download_file_path, cache_dir, file_name)
    else:
        raise ValueError(f"Unsupported model format: {llm_spec.model_format}")

    meta_path = _get_meta_path(
        cache_dir,
        llm_spec.model_format,
        llm_spec.model_hub,
        quantization,
    )
    _generate_meta_file(meta_path, llm_family, llm_spec, quantization)

    return cache_dir


def cache(
    model_family: EmbeddingModelFamilyV1,
    model_spec: EmbeddingSpecV1,
    quantization: Optional[str] = None,
) -> str:
    if model_spec.model_hub == "huggingface":
        logger.info(f"Caching from Hugging Face: {model_spec.model_id}")
        return cache_from_huggingface(model_family, model_spec, quantization)
    elif model_spec.model_hub == "modelscope":
        logger.info(f"Caching from Modelscope: {model_spec.model_id}")
        return cache_from_modelscope(model_family, model_spec, quantization)
    else:
        raise ValueError(f"Unknown model hub: {model_spec.model_hub}")


def _check_revision(
    model_family: EmbeddingModelFamilyV1,
    model_spec: EmbeddingSpecV1,
    builtin: Iterable,
    meta_path: str,
    quantization: Optional[str] = None,
) -> bool:
    for family in builtin:
        if model_family.model_name == family.model_name:
            specs = family.model_specs
            for spec in specs:
                if spec.model_format == "transformers" and (
                    quantization is None or quantization in spec.quantizations
                ):
                    return valid_model_revision(meta_path, spec.model_revision)
    return False


def get_cache_status(
    model_family: EmbeddingModelFamilyV1,
    model_spec: EmbeddingSpecV1,
    quantization: Optional[str] = None,
) -> bool:
    """
    Checks if a model's cache status is available based on the model format and quantization.
    Supports different directories and model formats.
    """

    def check_file_status(meta_path: str) -> bool:
        return os.path.exists(meta_path)

    def check_revision_status(
        meta_path: str, families: Iterable, quantization: Optional[str] = None
    ) -> bool:
        return _check_revision(
            model_family, model_spec, families, meta_path, quantization
        )

    def handle_quantization(q: Union[str, None]) -> bool:
        specific_cache_dir = _get_cache_dir(
            model_family, model_spec, q, create_if_not_exist=False
        )
        meta_paths = {
            "huggingface": _get_meta_path(
                specific_cache_dir, model_spec.model_format, "huggingface", q
            ),
            "modelscope": _get_meta_path(
                specific_cache_dir, model_spec.model_format, "modelscope", q
            ),
        }
        if model_spec.model_format == "transformers":
            return check_revision_status(
                meta_paths["huggingface"], BUILTIN_EMBEDDING_MODELS.values(), q
            ) or check_revision_status(
                meta_paths["modelscope"],
                BUILTIN_MODELSCOPE_EMBEDDING_MODELS.values(),
                q,
            )
        else:
            return check_file_status(meta_paths["huggingface"]) or check_file_status(
                meta_paths["modelscope"]
            )

    if model_spec.model_id and "{" in model_spec.model_id:
        return (
            [handle_quantization(q) for q in model_spec.quantizations]
            if quantization is None
            else handle_quantization(quantization)
        )
    else:
        return (
            [handle_quantization(q) for q in model_spec.quantizations]
            if model_spec.model_format != "transformers"
            else handle_quantization(None)
        )


class EmbeddingModel(abc.ABC):
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_family: EmbeddingModelFamilyV1,
        model_spec: EmbeddingSpecV1,
        quantization: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._model = None
        self._tokenizer = None
        self._counter = 0
        self._model_family = model_family
        self._model_spec = model_spec
        self._quantization = quantization
        self._model_name = self._model_family.model_name
        self._kwargs = kwargs

    @classmethod
    @abstractmethod
    def check_lib(cls) -> bool:
        pass

    @classmethod
    @abstractmethod
    def match_json(
        cls,
        model_family: EmbeddingModelFamilyV1,
        model_spec: EmbeddingSpecV1,
        quantization: str,
    ) -> bool:
        pass

    @classmethod
    def match(
        cls,
        model_family: EmbeddingModelFamilyV1,
        model_spec: EmbeddingSpecV1,
        quantization: str,
    ):
        """
        Return if the model_spec can be matched.
        """
        if not cls.check_lib():
            return False
        return cls.match_json(model_family, model_spec, quantization)

    @abstractmethod
    def load(self):
        """
        Load embedding model
        """

    def _fix_langchain_openai_inputs(
        self, sentences: Union[str, List[str], Dict[str, str], List[Dict[str, str]]]
    ):
        # Check if sentences is a two-dimensional list of integers
        if (
            isinstance(sentences, list)
            and len(sentences) > 0
            and isinstance(sentences[0], list)
            and len(sentences[0]) > 0
            and isinstance(sentences[0][0], int)
        ):
            # List[List[int]] stands for encoded inputs
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            lines_decoded = []

            for line in sentences:
                try:
                    # Decode each token into bytes, then join them into a complete string
                    output = b"".join(
                        enc.decode_single_token_bytes(token) for token in line
                    )
                    # Convert the byte sequence into a UTF-8 encoded string
                    decoded_line = output.decode("utf-8")
                    lines_decoded.append(decoded_line)
                except (ValueError, TypeError, UnicodeDecodeError) as e:
                    raise ValidationError([ErrorWrapper(e, loc=ROOT_KEY)], self)

            # Update sentences to be the list of decoded strings
            if len(lines_decoded) == 1:
                sentences = lines_decoded[0]
            else:
                sentences = lines_decoded
        return sentences

    @staticmethod
    # copied from sentence-transformers
    def _text_length(text):
        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    @abstractmethod
    def create_embedding(
        self,
        sentences: Union[str, List[str]],
        **kwargs,
    ):
        """
        Creating embeddings from sentences.

        Parameters
        ----------
        sentences: Union[str, List[str]]
            Input text to embed, encoded as a string or array of tokens.
            To embed multiple inputs in a single request, pass an array of strings or array of token arrays.

        Returns
        -------
        Embedding
           The resulted Embedding vector that can be easily consumed by machine learning models and algorithms.
        """

    def convert_ids_to_tokens(
        self,
        batch_token_ids: Union[List[Union[int, str]], List[List[Union[int, str]]]],
        **kwargs,
    ) -> Union[List[str]]:
        """
        Convert token ids to tokens
        """
        assert self._model is not None
        if isinstance(batch_token_ids, (int, str)):
            return self._tokenizer.decode([int(str(batch_token_ids))])[0]

        batch_decoded_texts: List[str] = []

        # check if it's a nested list
        if (
            isinstance(batch_token_ids, list)
            and batch_token_ids
            and isinstance(batch_token_ids[0], list)
        ):
            for token_ids in batch_token_ids:
                token_ids = [int(token_id) for token_id in token_ids]  # type: ignore
                batch_decoded_texts.append(
                    self._tokenizer.convert_ids_to_tokens(token_ids)
                )
        else:
            batch_token_ids = [int(token_id) for token_id in batch_token_ids]  # type: ignore
            batch_decoded_texts = self._tokenizer.convert_ids_to_tokens(batch_token_ids)
        return batch_decoded_texts

    def _clean_cache_if_needed(self, all_token_nums: int):
        # clean cache if possible
        self._counter += 1
        if (
            self._counter % EMBEDDING_EMPTY_CACHE_COUNT == 0
            or all_token_nums >= EMBEDDING_EMPTY_CACHE_TOKENS
        ):
            logger.debug(
                "Empty embedding cache, calling count %s, all_token_nums %s",
                self._counter,
                all_token_nums,
            )
            gc.collect()
            empty_cache()


def create_embedding_model_instance(
    subpool_addr: str,
    devices: Optional[List[str]],
    model_uid: str,
    model_name: str,
    model_engine: Optional[str],
    model_format: Optional[str] = None,
    quantization: Optional[str] = None,
    download_hub: Optional[
        Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[EmbeddingModel, EmbeddingModelDescription]:
    model_family, model_spec, quantization = match_embedding(
        model_name, model_format, quantization, download_hub
    )
    if model_path is None:
        model_path = cache(model_family, model_spec, quantization)

    if model_engine is None:
        # unlike LLM and for compatibility,
        # we use sentence_transformers as the default engine for all models
        model_engine = "sentence_transformers"

    from .embed_family import check_engine_by_model_name_and_engine

    embedding_cls = check_engine_by_model_name_and_engine(
        model_engine, model_name, model_format, quantization
    )
    devices = devices or ["cpu"]
    # model class should be one of flag, fastembed, sentence_transformers
    model = embedding_cls(
        model_uid, model_path, model_family, model_spec, quantization, devices, **kwargs
    )
    model_description = EmbeddingModelDescription(
        subpool_addr,
        devices,
        model_family,
        model_spec,
        quantization,
        model_path,
    )
    return model, model_description
