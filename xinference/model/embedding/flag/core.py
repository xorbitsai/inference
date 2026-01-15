# Copyright 2022-2026 XProbe Inc.
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
from typing import List, Optional, Tuple, Union, no_type_check

import numpy as np
import torch

try:
    from FlagEmbedding.inference.embedder.model_mapping import (
        support_native_bge_model_list,
    )

    flag_installed = True
except ImportError:
    flag_installed = False

from ....device_utils import get_available_device
from ....types import Embedding, EmbeddingData, EmbeddingUsage
from ...batch import BatchMixin
from ...utils import check_dependency_available
from ..core import EmbeddingModel, EmbeddingModelFamilyV2, EmbeddingSpecV1

FLAG_EMBEDDER_MODEL_LIST = support_native_bge_model_list() if flag_installed else []
logger = logging.getLogger(__name__)


class FlagEmbeddingModel(EmbeddingModel, BatchMixin):
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_family: EmbeddingModelFamilyV2,
        quantization: Optional[str] = None,
        device: Optional[str] = None,
        return_sparse: bool = False,
        **kwargs,
    ):
        EmbeddingModel.__init__(
            self, model_uid, model_path, model_family, quantization, device, **kwargs
        )
        BatchMixin.__init__(self, self.create_embedding, **kwargs)  # type: ignore
        self._return_sparse = return_sparse

    def load(self):
        # add truncate_dim args hint
        if self._kwargs and "dimensions" in self._kwargs:
            raise NotImplementedError(
                "Flag embedder does not support dimensions argument now."
            )
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            error_message = "Failed to import module 'BGEM3FlagModel'"
            installation_guide = [
                "Please make sure 'FlagEmbedding' is installed. ",
                "You can install it by `pip install FlagEmbedding`\n",
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
            model_kwargs = {"use_fp16": True}
        else:
            model_kwargs = {}
        self._model = BGEM3FlagModel(
            self._model_path,
            device=self._device,
            trust_remote_code=True,
            return_sparse=self._return_sparse,
            **model_kwargs,
        )
        self._tokenizer = self._model.tokenizer

    def _create_embedding(
        self,
        sentences: Union[str, List[str]],
        **kwargs,
    ):
        from FlagEmbedding import BGEM3FlagModel

        # flag embed dose not have this param
        # kwargs.setdefault("normalize_embeddings", True)
        model_uid = kwargs.pop("model_uid", None)

        @no_type_check
        def encode(
            model: Union[BGEM3FlagModel],
            sentences: Union[str, List[str]],
            batch_size: int = 32,
            show_progress_bar: bool = None,
            output_value: str = "sparse_embedding",
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = None,
            normalize_embeddings: bool = False,
            **kwargs,
        ):
            """
            Computes sentence embeddings with bge-m3 model
            Nothing special here, just replace sentence-transformer with FlagEmbedding
            TODO: think about how to solve the redundant code of encode method in the future

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
            from tqdm.autonotebook import trange

            if show_progress_bar is None:
                show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO
                    or logger.getEffectiveLevel() == logging.DEBUG
                )

            if convert_to_tensor:
                convert_to_numpy = False

            if output_value != "sparse_embedding":
                convert_to_tensor = False
                convert_to_numpy = False

            input_was_string = False
            if isinstance(sentences, str) or not hasattr(
                sentences, "__len__"
            ):  # Cast an individual sentence to a list with length 1
                sentences = [sentences]
                input_was_string = True

            if device is None:
                device = get_available_device()
                logger.info(f"Use pytorch device_name: {device}")

            all_embeddings = []

            length_sorted_idx = np.argsort(
                [-self._text_length(sen) for sen in sentences]
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

                with torch.no_grad():
                    out_features = model.encode(sentences_batch, **kwargs)

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
                    # for sparse embedding
                    else:
                        # TODO: Here need check if we can return density_vecs and lexical_weights at the same time
                        if kwargs.get("return_sparse"):
                            embeddings = out_features["lexical_weights"]
                        else:
                            embeddings = out_features["dense_vecs"]

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

            return all_embeddings

        all_embeddings = encode(
            self._model,
            sentences,
            convert_to_numpy=False,
            **kwargs,
        )

        if isinstance(sentences, str):
            all_embeddings = [all_embeddings]
        embedding_list = []
        for index, data in enumerate(all_embeddings):
            if kwargs.get("return_sparse"):
                embedding_list.append(
                    EmbeddingData(
                        index=index,
                        object="sparse_embedding",
                        embedding={k: float(v) for k, v in data.items()},
                    )
                )
            else:
                embedding_list.append(
                    EmbeddingData(
                        index=index, object="embedding", embedding=data.tolist()
                    )
                )
        usage = EmbeddingUsage(prompt_tokens=-1, total_tokens=-1)
        result = Embedding(
            object=("list" if kwargs.get("return_sparse") else "dict"),  # type: ignore
            model=model_uid,
            model_replica=self._model_uid,
            data=embedding_list,
            usage=usage,
        )

        # clean cache if possible
        # TODO: support token statistics
        self._clean_cache_if_needed(all_token_nums=0)

        return result

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        dep_check = check_dependency_available("FlagEmbedding", "FlagEmbedding")
        if dep_check != True:
            return dep_check
        return True

    @classmethod
    def match_json(
        cls,
        model_family: EmbeddingModelFamilyV2,
        model_spec: EmbeddingSpecV1,
        quantization: str,
    ) -> Union[bool, Tuple[bool, str]]:
        if model_spec.model_format not in ["pytorch"]:
            return False, "FlagEmbedding engine only supports pytorch format"
        if model_family.model_name not in FLAG_EMBEDDER_MODEL_LIST:
            return False, f"{model_family.model_name} is not supported by FlagEmbedding"
        return True
