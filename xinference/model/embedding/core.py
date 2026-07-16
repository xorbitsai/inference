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

import abc
import gc
import inspect
import logging
import os
from abc import abstractmethod
from collections import defaultdict
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

from xoscar import extensible

from ..._compat import ROOT_KEY, BaseModel, ErrorWrapper, Field, ValidationError
from ...device_utils import empty_cache
from ...types import Embedding
from ...utils import make_hashable
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

# Char-per-token estimate used by ``EmbeddingModel._truncate_sentences`` for the
# character-based fallback (engines without a Python tokenizer, e.g. llama_cpp,
# or when the tokenizer call itself fails). English-biased; CJK may run longer.
_EMBEDDING_TRUNCATE_CHAR_PER_TOKEN = int(
    os.getenv("XINFERENCE_EMBEDDING_TRUNCATE_CHAR_PER_TOKEN", "4")
)
assert _EMBEDDING_TRUNCATE_CHAR_PER_TOKEN > 0


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
    # Provenance: True only for bundled built-in models (gates trust_remote_code).
    is_builtin: bool = False

    class Config:
        extra = "allow"

    def to_description(self):
        spec = self.model_specs[0]
        return {
            "model_type": "embedding",
            "address": getattr(self, "address", None),
            "accelerators": getattr(self, "accelerators", None),
            "model_name": self.model_name,
            "model_format": spec.model_format,
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
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        pass

    @classmethod
    @abstractmethod
    def match_json(
        cls,
        model_family: EmbeddingModelFamilyV2,
        model_spec: EmbeddingSpecV1,
        quantization: str,
    ) -> Union[bool, Tuple[bool, str]]:
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
        lib_result = cls.check_lib()
        if lib_result != True:
            return False
        match_result = cls.match_json(model_family, model_spec, quantization)
        return match_result == True

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
    def _create_embedding(
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

    @extensible
    def create_embedding(
        self,
        sentences: Union[str, List[str]],
        **kwargs,
    ):
        truncate_prompt_tokens = kwargs.pop("truncate_prompt_tokens", None)
        if truncate_prompt_tokens is not None:
            sentences = self._truncate_sentences(sentences, truncate_prompt_tokens)
        return self._create_embedding(sentences, **kwargs)

    @create_embedding.batch  # type: ignore
    def create_embedding(self, args_list, kwargs_list):
        grouped = defaultdict(
            lambda: {"sentences": [], "offsets": [], "kwargs": None, "indices": []}
        )

        # 1. Group by kwargs hash
        for i, (args, kwargs) in enumerate(zip(args_list, kwargs_list)):
            sentences, extra_kwargs = self._extract_sentences_kwargs(args, kwargs)
            if isinstance(sentences, str):
                sentences = [sentences]

            key = make_hashable(extra_kwargs)
            group = grouped[key]
            group["kwargs"] = extra_kwargs

            current_offset = len(group["sentences"])
            group["offsets"].append((current_offset, len(sentences)))
            group["sentences"].extend(sentences)
            group["indices"].append(i)  # remember original position

        results_with_index = []

        # 2. Process each group separately
        for key, group in grouped.items():
            sentences = group["sentences"]
            kwargs = group["kwargs"]
            offsets = group["offsets"]
            indices = group["indices"]

            # Honor ``truncate_prompt_tokens`` for embeddings (previously only
            # implemented for vLLM LLMs). Pop it here so it never leaks into
            # the engine's encode()/forward(), and bound the input length to
            # avoid O(L^2) attention OOM on long documents.
            truncate_prompt_tokens = kwargs.pop("truncate_prompt_tokens", None)
            if truncate_prompt_tokens is not None:
                sentences = self._truncate_sentences(sentences, truncate_prompt_tokens)

            embedding_list = self._create_embedding(sentences, **kwargs)
            usage = embedding_list.get("usage", {})
            model_uid = kwargs.get("model_uid", "unknown")

            # 3. Split and attach original index
            for (offset, n), idx in zip(offsets, indices):
                data = embedding_list["data"][offset : offset + n]
                result = Embedding(
                    object="list",
                    model=model_uid,
                    model_replica=self._model_uid,
                    data=data,
                    usage=usage,
                )
                results_with_index.append((idx, result))

        # 4. Sort by original call order
        results_with_index.sort(key=lambda x: x[0])
        results = [r for _, r in results_with_index]
        return results

    def _extract_sentences_kwargs(self, args, kwargs):
        """
        Extract the 'sentences' argument and remaining kwargs from (*args, **kwargs)
        for a given function.

        This uses inspect.signature(func).bind_partial() to automatically match
        both positional and keyword arguments, while handling bound methods
        (functions with 'self' as the first parameter).

        Args:
            func: The target function whose parameters define how to bind args/kwargs.
            args: The positional arguments passed to the function.
            kwargs: The keyword arguments passed to the function.

        Returns:
            A tuple (sentences, extra_kwargs), where:
              - sentences: The extracted 'sentences' argument (never None).
              - extra_kwargs: Remaining keyword arguments excluding 'sentences'.

        Raises:
            KeyError: If 'sentences' argument is not found.
            TypeError: If args/kwargs do not match the function signature.
        """
        sig = inspect.signature(self._create_embedding)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        if "sentences" not in bound.arguments:
            raise KeyError("'sentences' argument not found in args/kwargs")

        sentences = bound.arguments["sentences"]
        extra_kwargs = {k: v for k, v in kwargs.items() if k != "sentences"}
        return sentences, extra_kwargs

    def _truncate_sentences(
        self,
        sentences: Union[
            str,
            List[str],
            List[int],
            List[List[int]],
            Dict[str, str],
            List[Dict[str, str]],
        ],
        truncate_prompt_tokens: Optional[int],
    ) -> Union[
        str,
        List[str],
        List[int],
        List[List[int]],
        Dict[str, str],
        List[Dict[str, str]],
    ]:
        """Truncate input to ``truncate_prompt_tokens`` tokens before encoding.

        Mirrors vLLM LLM semantics (``xinference/model/llm/vllm/core.py``):
        ``None`` = no truncation, ``>0`` = cap at N tokens, ``==0`` = explicit
        empty (``max_length=0``), ``<0`` = cap at the model's own ``max_tokens``.

        Structure-preserving: token arrays (``List[int]`` / ``List[List[int]]``)
        are sliced to N ids; multimodal dicts (``Dict[str, str]`` /
        ``List[Dict[str, str]]``) keep their keys and non-text values, and only
        the ``text`` field is truncated (``image`` / ``video`` / ``audio`` and
        other media fields are preserved verbatim). Plain ``str`` / ``List[str]`` are
        truncated token-based (sentence_transformers / flag / vllm) with a
        character-based fallback (llama_cpp, or tokenizer failure).

        This method must never raise: a tokenizer/decode failure degrades to a
        character-based cut so embedding serving is not interrupted. It runs
        *before* ``_fix_langchain_openai_inputs`` decodes token arrays, so it
        must not stringify structured inputs (that would corrupt them -- e.g.
        ``[[1, 2, 3]]`` embedded as the literal text ``"[[1,"``).
        """
        if truncate_prompt_tokens is None:
            return sentences

        # Resolve the effective token limit, matching vLLM LLM ``_tokenize``:
        #   None  -> no truncation (handled above)
        #   >0    -> cap at N tokens
        #   ==0   -> explicit max_length=0 (empty input)
        #   <0    -> cap at the model's own max_tokens
        n_tokens: Optional[int]
        if truncate_prompt_tokens > 0:
            n_tokens = truncate_prompt_tokens
        elif truncate_prompt_tokens == 0:
            n_tokens = 0
        else:  # <0: cap at the model's own max_tokens
            n_tokens = getattr(self.model_family, "max_tokens", None)
            if not n_tokens:  # None / 0 -> unknown, give up truncation safely
                logger.debug(
                    "truncate_prompt_tokens<0 but max_tokens unknown for %s; skip",
                    self._model_uid,
                )
                return sentences

        return self._truncate_value(sentences, n_tokens)

    def _truncate_value(self, value: Any, n_tokens: int) -> Any:
        """Recursively truncate ``value`` to ``n_tokens`` while preserving its
        structure (see ``_truncate_sentences``)."""
        # Token arrays: a single doc as a flat list of token ids.
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], int):
            return value[:n_tokens]
        # Multiple docs, each a list of token ids.
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
            return [inner[:n_tokens] for inner in value]
        # Multimodal / structured: truncate ONLY the ``text`` field. The
        # other Jina multimodal fields (``image`` / ``video`` / ``audio``)
        # carry URLs, file paths or base64 payloads; truncating them corrupts
        # the media (e.g. a 222-char base64 was cut to 32 chars at n=8). Keys
        # and non-text values are preserved verbatim.
        if isinstance(value, dict):
            return {
                k: (
                    self._truncate_text(v, n_tokens)
                    if k == "text" and isinstance(v, str)
                    else v
                )
                for k, v in value.items()
            }
        # A list of texts or of dicts: recurse per element.
        if isinstance(value, list):
            return [self._truncate_value(item, n_tokens) for item in value]
        if isinstance(value, str):
            return self._truncate_text(value, n_tokens)
        # Numbers / None / other primitives: leave untouched.
        return value

    def _truncate_text(self, s: str, n_tokens: int) -> str:
        """Token-based truncation of a single string, with a character-based
        fallback (llama_cpp has no Python tokenizer, or the call may raise).

        ``n_tokens == 0`` matches vLLM ``max_length=0``: both the tokenizer
        path (empty ``input_ids``) and the char fallback (``s[:0]``) yield "".
        """
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is not None:
            try:
                ids = tokenizer(
                    s,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=n_tokens,
                ).input_ids
                return tokenizer.decode(ids, skip_special_tokens=True)
            except Exception:
                logger.debug(
                    "token-based truncation failed for %s, fallback to char",
                    self._model_uid,
                    exc_info=True,
                )
        # Fallback: character-based cut (llama_cpp, or tokenizer failure).
        return s[: n_tokens * _EMBEDDING_TRUNCATE_CHAR_PER_TOKEN]

    def _get_batch_size(self, *args, **kwargs) -> int:
        sentences = self._extract_sentences_kwargs(args, kwargs)[0]
        if isinstance(sentences, list):
            return len(sentences)
        else:
            return 1

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

    enable_virtual_env = kwargs.pop("enable_virtual_env", None)
    model_family = match_embedding(model_name, model_format, quantization, download_hub)
    if model_path is None:
        cache_manager = EmbeddingCacheManager(model_family)
        model_path = cache_manager.cache()

    if model_engine is None:
        # unlike LLM and for compatibility,
        # we use sentence_transformers as the default engine for all models
        model_engine = "sentence_transformers"

    from .embed_family import (
        check_engine_by_model_name_and_engine,
        check_engine_by_model_name_and_engine_with_virtual_env,
    )

    if enable_virtual_env is None:
        from ...constants import XINFERENCE_ENABLE_VIRTUAL_ENV

        enable_virtual_env = XINFERENCE_ENABLE_VIRTUAL_ENV

    if enable_virtual_env:
        embedding_cls = check_engine_by_model_name_and_engine_with_virtual_env(
            model_engine,
            model_name,
            model_format,
            quantization,
            model_family=model_family,
        )
    else:
        embedding_cls = check_engine_by_model_name_and_engine(
            model_engine,
            model_name,
            model_format,
            quantization,
        )
    model = embedding_cls(
        model_uid,
        model_path,
        model_family,
        quantization,
        **kwargs,
    )
    return model
