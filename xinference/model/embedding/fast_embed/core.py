# Copyright 2022-2025 XProbe Inc.
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
from typing import List, Optional, Union

try:
    from fastembed import (
        LateInteractionTextEmbedding,
        SparseTextEmbedding,
        TextEmbedding,
    )
except ImportError:
    LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding = None, None, None

import numpy as np
import torch

from ....types import Embedding, EmbeddingData, EmbeddingUsage
from ..core import EmbeddingModel, EmbeddingModelSpec

logger = logging.getLogger(__name__)

# fastembed provide model name with namespace, we only need the model name
FAST_EMBEDDER_DENSE_MODEL_LIST = (
    [
        model_info["model"].split("/")[-1]
        for model_info in TextEmbedding.list_supported_models()
    ]
    if TextEmbedding
    else []
)
FAST_EMBEDDER_SPARSE_MODEL_LIST = (
    [
        model_info["model"].split("/")[-1]
        for model_info in SparseTextEmbedding.list_supported_models()
    ]
    if SparseTextEmbedding
    else []
)
FAST_EMBEDDER_LATE_INTERACTION_MODEL_LIST = (
    [
        model_info["model"].split("/")[-1]
        for model_info in LateInteractionTextEmbedding.list_supported_models()
    ]
    if LateInteractionTextEmbedding
    else []
)

FAST_EMBEDDER_MODEL_LIST = (
    FAST_EMBEDDER_DENSE_MODEL_LIST
    + FAST_EMBEDDER_SPARSE_MODEL_LIST
    + FAST_EMBEDDER_LATE_INTERACTION_MODEL_LIST
)


class FastEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: EmbeddingModelSpec,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_uid, model_path, model_spec, device, **kwargs)
        self._device_ids = kwargs.pop("device_ids", None)
        self._load_type = kwargs.pop("load_type", "dense")

    def load(self):
        # TODO: load model
        try:
            # sparse, sparse and dense, dense. and LateInteractionTextEmbedding(ColBERT)

            from fastembed import (
                LateInteractionTextEmbedding,
                SparseTextEmbedding,
                TextEmbedding,
            )
        except ImportError:
            error_message = "Failed to import module 'fastembed'"
            installation_guide = [
                "Please make sure 'fastembed' is installed. ",
                "You can install it by `pip install fastembed`\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

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
                        f"BGE engine only support fp16, but got {torch_dtype_str}. Using default torch dtype: fp16."
                    )
                    torch_dtype = torch.float16
            except AttributeError:
                logger.warning(
                    f"Load embedding model with  unknown torch dtype '{torch_dtype_str}'. Using default torch dtype: fp32."
                )
                torch_dtype = torch.float16

        if torch_dtype and torch_dtype == torch.float16:
            load_model_kwargs = {"use_fp16": True}
        else:
            load_model_kwargs = {}

        load_type = self._load_type
        if self._model_name in FAST_EMBEDDER_DENSE_MODEL_LIST and load_type == "dense":
            # The loading model in fast embed requires the use of _model_spec.model_id, which means the format that requires the 'model manufacturer name/model name'
            self._model = TextEmbedding(
                model_name=self._model_spec.model_id,
                cache_dir=self._model_path,
                cuda=self._device == "cuda",
                device_ids=self._device,
                **load_model_kwargs,
            )
        elif (
            self._model_name in FAST_EMBEDDER_SPARSE_MODEL_LIST
            and load_type == "sparse"
        ):
            self._model = SparseTextEmbedding(
                model_name=self.self._model_spec.model_id,
                cache_dir=self._model_path,
                cuda=self._device == "cuda",
                **load_model_kwargs,
            )
            # TODO: The method of late_interaction is not supported for the time being
        elif (
            self._model_name in FAST_EMBEDDER_LATE_INTERACTION_MODEL_LIST
            and load_type == "late_interaction"
        ):
            self._model = LateInteractionTextEmbedding(
                model_name=self._model_name,
                cache_dir=self._model_path,
                cuda=self._device == "cuda",
                device_ids=self._device_ids,
                **load_model_kwargs,
            )

        else:
            raise ValueError(
                f"Unsupported model: {self._model_name}, load_class: {load_type}. please check the model and load_class is match.you can change the load_class by setting the load_type in kwargs."
            )

    def create_embedding(self, sentences: Union[str, List[str]], **kwargs):
        # TODO: embed text

        sentences = self._fix_langchain_openai_inputs(sentences)
        model_uid = kwargs.pop("model_uid", None)

        def encode(
            model: Union[
                TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
            ],
            sentences: Union[str, List[str]],
            batch_size: int = 32,
            show_progress_bar: bool = False,
            output_value: str = "sentence_embedding",
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = "cuda",
            normalize_embeddings: bool = False,
            **kwargs,
        ):
            """
            Computes sentence embeddings with fastembed model

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
            # model.eval()
            if show_progress_bar is None:
                show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO
                    or logger.getEffectiveLevel() == logging.DEBUG
                )
            # Can we drop this function? What's this part to do? Can users expand themself support?
            if convert_to_tensor:
                convert_to_numpy = False

            if output_value != "sentence_embedding":
                convert_to_tensor = False
                convert_to_numpy = False

            # If the input is a single sentence, convert it to a list
            if isinstance(sentences, str) or not hasattr(
                sentences, "__len__"
            ):  # Cast an individual sentence to a list with length 1
                sentences = [str(sentences)]

            # Sort sentences by length
            length_sorted_idx = np.argsort(
                [-self._text_length(sen) for sen in sentences]
            )

            from tqdm.autonotebook import trange

            # from sentence_transformers.util import batch_to_device
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
                # fast embed dose not support this feature
                # features = model.tokenize(sentences_batch)
                # features = batch_to_device(features, device)
                # when batching, the attention mask 1 means there is a token
                # thus we just sum up it to get the total number of tokens
                # all_token_nums += features["attention_mask"].sum().item()

                with torch.no_grad():
                    # out_features = model.embed(sentences_batch, **kwargs)

                    # Each model has its corresponding different output_value
                    if isinstance(model, TextEmbedding):
                        out_features = model.embed(sentences_batch, **kwargs)
                    elif isinstance(model, SparseTextEmbedding):
                        out_features = model.embed(sentences_batch, **kwargs)
                    elif isinstance(model, LateInteractionTextEmbedding):
                        out_features = model.embed(sentences_batch, **kwargs)

            embeddings = list(out_features)

            if convert_to_tensor:
                if len(embeddings):
                    embeddings_list = list(model.embed(sentences_batch, **kwargs))
                    tensors = [torch.from_numpy(arr) for arr in embeddings_list]
                    embeddings = torch.stack(tensors)
                else:
                    embeddings = torch.Tensor()
            elif convert_to_numpy:
                # Need to handle the case when device=gpu
                embeddings = np.asarray(embeddings)
            return embeddings

        all_embeddings = encode(self._model, sentences, **kwargs)
        embedding_list = []
        for index, data in enumerate(all_embeddings):
            embedding_list.append(
                EmbeddingData(index=index, object="embedding", embedding=data.tolist())
            )
        # fast embed doesn't support tokenize, is this method necessary? Maybe we should just leave it empty
        usage = EmbeddingUsage(prompt_tokens=-1, total_tokens=-1)
        result = Embedding(
            object=("list" if kwargs.get("return_sparse") else "dict"),  # type: ignore
            model=model_uid,
            model_replica=self._model_uid,
            data=embedding_list,
            usage=usage,
        )
        return result

    def convert_ids_to_tokens(
        self,
        batch_token_ids: Union[List[Union[int, str]], List[List[Union[int, str]]]],
        **kwargs,
    ):
        assert self._model is not None
        if isinstance(batch_token_ids, (int, str)):
            return self._model.tokenizer.decode([int(str(batch_token_ids))])[0]

        # check if it's a nested list
        if (
            isinstance(batch_token_ids, list)
            and batch_token_ids
            and isinstance(batch_token_ids[0], list)
        ):
            return self._model.model.tokenizer.decode_batch(batch_token_ids)
        else:
            return self._model.tokenizer.decode(batch_token_ids)

    # Return True for any model supported by fastembed. The specific loading class is determined by the user.
    @classmethod
    def match(cls, model_spec: EmbeddingModelSpec):
        if model_spec.model_name in FAST_EMBEDDER_MODEL_LIST:
            return True
        return False
