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
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple, Union, no_type_check

import numpy as np
import torch

from ...device_utils import empty_cache
from ...types import Embedding, EmbeddingData, EmbeddingUsage
from ..core import CacheableModelSpec, ModelDescription
from ..utils import get_cache_dir, is_model_cached

logger = logging.getLogger(__name__)

# Used for check whether the model is cached.
# Init when registering all the builtin models.
MODEL_NAME_TO_REVISION: Dict[str, List[str]] = defaultdict(list)
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


class EmbeddingModelSpec(CacheableModelSpec):
    model_name: str
    dimensions: int
    max_tokens: int
    language: List[str]
    model_id: str
    model_revision: Optional[str]
    model_hub: str = "huggingface"


class EmbeddingModelDescription(ModelDescription):
    def __init__(
        self,
        address: Optional[str],
        devices: Optional[List[str]],
        model_spec: EmbeddingModelSpec,
        model_path: Optional[str] = None,
    ):
        super().__init__(address, devices, model_path=model_path)
        self._model_spec = model_spec

    def to_dict(self):
        return {
            "model_type": "embedding",
            "address": self.address,
            "accelerators": self.devices,
            "model_name": self._model_spec.model_name,
            "dimensions": self._model_spec.dimensions,
            "max_tokens": self._model_spec.max_tokens,
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
            "dimensions": self._model_spec.dimensions,
            "max_tokens": self._model_spec.max_tokens,
        }


def generate_embedding_description(
    model_spec: EmbeddingModelSpec,
) -> Dict[str, List[Dict]]:
    res = defaultdict(list)
    res[model_spec.model_name].append(
        EmbeddingModelDescription(None, None, model_spec).to_version_info()
    )
    return res


def cache(model_spec: EmbeddingModelSpec):
    from ..utils import cache

    return cache(model_spec, EmbeddingModelDescription)


def get_cache_status(
    model_spec: EmbeddingModelSpec,
) -> bool:
    return is_model_cached(model_spec, MODEL_NAME_TO_REVISION)


class EmbeddingModel:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: EmbeddingModelSpec,
        device: Optional[str] = None,
        **kwargs,
    ):
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._model = None
        self._counter = 0
        self._model_spec = model_spec
        self._kwargs = kwargs

    def load(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            error_message = "Failed to import module 'SentenceTransformer'"
            installation_guide = [
                "Please make sure 'sentence-transformers' is installed. ",
                "You can install it by `pip install sentence-transformers`\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        class XSentenceTransformer(SentenceTransformer):
            def to(self, *args, **kwargs):
                pass

        torch_dtype = None
        if torch_dtype_str := self._kwargs.get("torch_dtype"):
            try:
                torch_dtype = getattr(torch, torch_dtype_str)
                if torch_dtype not in [
                    torch.float16,
                    torch.float32,
                    torch.bfloat16,
                ]:
                    logger.warning(
                        f"Load embedding model with unsupported torch dtype :  {torch_dtype_str}. Using default torch dtype: fp32."
                    )
                    torch_dtype = torch.float32
            except AttributeError:
                logger.warning(
                    f"Load embedding model with  unknown torch dtype '{torch_dtype_str}'. Using default torch dtype: fp32."
                )
                torch_dtype = torch.float32

        from ..utils import patch_trust_remote_code

        patch_trust_remote_code()
        if (
            "gte" in self._model_spec.model_name.lower()
            and "qwen2" in self._model_spec.model_name.lower()
        ):
            model_kwargs = {"device_map": "auto"}
            if torch_dtype:
                model_kwargs["torch_dtype"] = torch_dtype
            self._model = XSentenceTransformer(
                self._model_path,
                device=self._device,
                model_kwargs=model_kwargs,
            )
        else:
            model_kwargs = {"torch_dtype": torch_dtype} if torch_dtype else None
            self._model = SentenceTransformer(
                self._model_path, device=self._device, model_kwargs=model_kwargs
            )

    def create_embedding(self, sentences: Union[str, List[str]], **kwargs):
        from sentence_transformers import SentenceTransformer

        kwargs.setdefault("normalize_embeddings", True)

        # copied from sentence-transformers, and modify it to return tokens num
        @no_type_check
        def encode(
            model: SentenceTransformer,
            sentences: Union[str, List[str]],
            prompt_name: Optional[str] = None,
            prompt: Optional[str] = None,
            batch_size: int = 32,
            show_progress_bar: bool = None,
            output_value: str = "sentence_embedding",
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = None,
            normalize_embeddings: bool = False,
        ):
            """
            Computes sentence embeddings

            :param sentences: the sentences to embed
            :param batch_size: the batch size used for the computation
            :param show_progress_bar: Output a progress bar when encode sentences
            :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
            :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
            :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
            :param device: Which torch.device to use for the computation
            :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

            :return:
               By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
            """
            import torch
            from sentence_transformers.util import batch_to_device
            from tqdm.autonotebook import trange

            model.eval()
            if show_progress_bar is None:
                show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO
                    or logger.getEffectiveLevel() == logging.DEBUG
                )

            if convert_to_tensor:
                convert_to_numpy = False

            if output_value != "sentence_embedding":
                convert_to_tensor = False
                convert_to_numpy = False

            input_was_string = False
            if isinstance(sentences, str) or not hasattr(
                sentences, "__len__"
            ):  # Cast an individual sentence to a list with length 1
                sentences = [sentences]
                input_was_string = True

            if prompt is None:
                if prompt_name is not None:
                    try:
                        prompt = model.prompts[prompt_name]
                    except KeyError:
                        raise ValueError(
                            f"Prompt name '{prompt_name}' not found in the configured prompts dictionary with keys {list(model.prompts.keys())!r}."
                        )
                elif model.default_prompt_name is not None:
                    prompt = model.prompts.get(model.default_prompt_name, None)
            else:
                if prompt_name is not None:
                    logger.warning(
                        "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. "
                        "Ignoring the `prompt_name` in favor of `prompt`."
                    )

            extra_features = {}
            if prompt is not None:
                sentences = [prompt + sentence for sentence in sentences]

                # Some models (e.g. INSTRUCTOR, GRIT) require removing the prompt before pooling
                # Tracking the prompt length allow us to remove the prompt during pooling
                tokenized_prompt = model.tokenize([prompt])
                if "input_ids" in tokenized_prompt:
                    extra_features["prompt_length"] = (
                        tokenized_prompt["input_ids"].shape[-1] - 1
                    )

            if device is None:
                device = model._target_device

            if (
                "gte" in self._model_spec.model_name.lower()
                and "qwen2" in self._model_spec.model_name.lower()
            ):
                model.to(device)

            all_embeddings = []
            all_token_nums = 0
            length_sorted_idx = np.argsort(
                [-model._text_length(sen) for sen in sentences]
            )
            sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

            for start_index in trange(
                0,
                len(sentences),
                batch_size,
                desc="Batches",
                disable=not show_progress_bar,
            ):
                sentences_batch = sentences_sorted[
                    start_index : start_index + batch_size
                ]
                features = model.tokenize(sentences_batch)
                features = batch_to_device(features, device)
                features.update(extra_features)
                # when batching, the attention mask 1 means there is a token
                # thus we just sum up it to get the total number of tokens
                all_token_nums += features["attention_mask"].sum().item()

                with torch.no_grad():
                    out_features = model.forward(features)

                    if output_value == "token_embeddings":
                        embeddings = []
                        for token_emb, attention in zip(
                            out_features[output_value], out_features["attention_mask"]
                        ):
                            last_mask_id = len(attention) - 1
                            while (
                                last_mask_id > 0 and attention[last_mask_id].item() == 0
                            ):
                                last_mask_id -= 1

                            embeddings.append(token_emb[0 : last_mask_id + 1])
                    elif output_value is None:  # Return all outputs
                        embeddings = []
                        for sent_idx in range(len(out_features["sentence_embedding"])):
                            row = {
                                name: out_features[name][sent_idx]
                                for name in out_features
                            }
                            embeddings.append(row)
                    else:  # Sentence embeddings
                        embeddings = out_features[output_value]
                        embeddings = embeddings.detach()
                        if normalize_embeddings:
                            embeddings = torch.nn.functional.normalize(
                                embeddings, p=2, dim=1
                            )

                        # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                        if convert_to_numpy:
                            embeddings = embeddings.cpu()

                    all_embeddings.extend(embeddings)

            all_embeddings = [
                all_embeddings[idx] for idx in np.argsort(length_sorted_idx)
            ]

            if convert_to_tensor:
                if len(all_embeddings):
                    all_embeddings = torch.stack(all_embeddings)
                else:
                    all_embeddings = torch.Tensor()
            elif convert_to_numpy:
                all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

            if input_was_string:
                all_embeddings = all_embeddings[0]

            return all_embeddings, all_token_nums

        if (
            "gte" in self._model_spec.model_name.lower()
            and "qwen2" in self._model_spec.model_name.lower()
        ):
            all_embeddings, all_token_nums = encode(
                self._model,
                sentences,
                prompt_name="query",
                convert_to_numpy=False,
                **kwargs,
            )
        else:
            all_embeddings, all_token_nums = encode(
                self._model,
                sentences,
                convert_to_numpy=False,
                **kwargs,
            )
        if isinstance(sentences, str):
            all_embeddings = [all_embeddings]
        embedding_list = []
        for index, data in enumerate(all_embeddings):
            embedding_list.append(
                EmbeddingData(index=index, object="embedding", embedding=data.tolist())
            )
        usage = EmbeddingUsage(
            prompt_tokens=all_token_nums, total_tokens=all_token_nums
        )
        result = Embedding(
            object="list",
            model=self._model_uid,
            data=embedding_list,
            usage=usage,
        )

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

        return result


def match_embedding(
    model_name: str,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
) -> EmbeddingModelSpec:
    from ..utils import download_from_modelscope
    from . import BUILTIN_EMBEDDING_MODELS, MODELSCOPE_EMBEDDING_MODELS
    from .custom import get_user_defined_embeddings

    # first, check whether it is a user-defined embedding model
    for model_spec in get_user_defined_embeddings():
        if model_name == model_spec.model_name:
            return model_spec

    if download_hub == "modelscope" and model_name in MODELSCOPE_EMBEDDING_MODELS:
        logger.debug(f"Embedding model {model_name} found in ModelScope.")
        return MODELSCOPE_EMBEDDING_MODELS[model_name]
    elif download_hub == "huggingface" and model_name in BUILTIN_EMBEDDING_MODELS:
        logger.debug(f"Embedding model {model_name} found in Huggingface.")
        return BUILTIN_EMBEDDING_MODELS[model_name]
    elif download_from_modelscope() and model_name in MODELSCOPE_EMBEDDING_MODELS:
        logger.debug(f"Embedding model {model_name} found in ModelScope.")
        return MODELSCOPE_EMBEDDING_MODELS[model_name]
    elif model_name in BUILTIN_EMBEDDING_MODELS:
        logger.debug(f"Embedding model {model_name} found in Huggingface.")
        return BUILTIN_EMBEDDING_MODELS[model_name]
    else:
        raise ValueError(
            f"Embedding model {model_name} not found, available"
            f"Huggingface: {BUILTIN_EMBEDDING_MODELS.keys()}"
            f"ModelScope: {MODELSCOPE_EMBEDDING_MODELS.keys()}"
        )


def create_embedding_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_name: str,
    download_hub: Optional[Literal["huggingface", "modelscope", "csghub"]] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[EmbeddingModel, EmbeddingModelDescription]:
    model_spec = match_embedding(model_name, download_hub)
    if model_path is None:
        model_path = cache(model_spec)

    model = EmbeddingModel(model_uid, model_path, model_spec, **kwargs)
    model_description = EmbeddingModelDescription(
        subpool_addr, devices, model_spec, model_path=model_path
    )
    return model, model_description
