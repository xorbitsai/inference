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
from typing import Annotated, Dict, List, Literal, Optional, Union

from ..._compat import ROOT_KEY, BaseModel, ErrorWrapper, Field, ValidationError
from ...device_utils import empty_cache
from ..core import VirtualEnvSettings
from ..utils import ModelInstanceInfoMixin
from .embed_family import match_embedding

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
    model_format: Literal["pytorch"]
    model_hub: str = "huggingface"
    model_id: Optional[str]
    model_uri: Optional[str]
    model_revision: Optional[str]
    quantization: str


class LlamaCppEmbeddingSpecV1(BaseModel):
    model_format: Literal["ggufv2"]
    model_hub: str = "huggingface"
    model_id: Optional[str]
    model_uri: Optional[str]
    model_revision: Optional[str]
    quantization: str
    model_file_name_template: str
    model_file_name_split_template: Optional[str]
    quantization_parts: Optional[Dict[str, List[str]]]


EmbeddingSpecV1 = Annotated[
    Union[TransformersEmbeddingSpecV1, LlamaCppEmbeddingSpecV1],
    Field(discriminator="model_format"),
]


# this class define the basic info of embedding model
class EmbeddingModelFamilyV2(BaseModel, ModelInstanceInfoMixin):
    version: Literal[2]
    model_name: str
    dimensions: int
    max_tokens: int
    language: List[str]
    model_specs: List["EmbeddingSpecV1"]
    cache_config: Optional[dict]
    virtualenv: Optional[VirtualEnvSettings]

    class Config:
        extra = "allow"

    def to_description(self):
        spec = self.model_specs[0]
        return {
            "model_type": "embedding",
            "address": getattr(self, "address", None),
            "accelerators": getattr(self, "accelerators", None),
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "max_tokens": self.max_tokens,
            "language": self.language,
            "model_hub": spec.model_hub,
            "model_revision": spec.model_revision,
            "quantization": spec.quantization,
        }

    def to_version_info(self):
        from .cache_manager import EmbeddingCacheManager

        cache_manager = EmbeddingCacheManager(self)

        return {
            "model_version": get_model_version(self),
            "model_file_location": cache_manager.get_cache_dir(),
            "cache_status": cache_manager.get_cache_status(),
            "dimensions": self.dimensions,
            "max_tokens": self.max_tokens,
        }


def get_model_version(embedding_model: EmbeddingModelFamilyV2) -> str:
    spec = embedding_model.model_specs[0]
    return f"{embedding_model.model_name}--{embedding_model.max_tokens}--{embedding_model.dimensions}--{spec.model_format}--{spec.quantization}"


def generate_embedding_description(
    model_family: EmbeddingModelFamilyV2,
) -> Dict[str, List[Dict]]:
    res = defaultdict(list)
    specs = [x for x in model_family.model_specs if x.model_hub == "huggingface"]
    for spec in specs:
        family = model_family.copy()
        family.model_specs = [spec]
        res[model_family.model_name].append(family.to_version_info())
    return res


class EmbeddingModel(abc.ABC):
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_family: EmbeddingModelFamilyV2,
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
        self.model_family = model_family
        self._model_spec = model_family.model_specs[0]
        self._quantization = quantization
        self._model_name = self.model_family.model_name
        self._kwargs = kwargs

    @classmethod
    @abstractmethod
    def check_lib(cls) -> bool:
        pass

    @classmethod
    @abstractmethod
    def match_json(
        cls,
        model_family: EmbeddingModelFamilyV2,
        model_spec: EmbeddingSpecV1,
        quantization: str,
    ) -> bool:
        pass

    @classmethod
    def match(
        cls,
        model_family: EmbeddingModelFamilyV2,
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
) -> EmbeddingModel:
    from .cache_manager import EmbeddingCacheManager

    model_family = match_embedding(model_name, model_format, quantization, download_hub)
    if model_path is None:
        cache_manager = EmbeddingCacheManager(model_family)
        model_path = cache_manager.cache()

    if model_engine is None:
        # unlike LLM and for compatibility,
        # we use sentence_transformers as the default engine for all models
        model_engine = "sentence_transformers"

    from .embed_family import check_engine_by_model_name_and_engine

    embedding_cls = check_engine_by_model_name_and_engine(
        model_engine, model_name, model_format, quantization
    )
    model = embedding_cls(
        model_uid,
        model_path,
        model_family,
        quantization,
        **kwargs,
    )
    return model
