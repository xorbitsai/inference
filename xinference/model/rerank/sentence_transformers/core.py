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
import gc
import importlib.util
import inspect
import logging
import os
import threading
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from xoscar import extensible

from ....device_utils import empty_cache
from ....types import Document, DocumentObj, Meta, Rerank, RerankTokens
from ....utils import make_hashable
from ...batch import BatchMixin
from ...utils import check_dependency_available, is_flash_attn_available
from ..core import (
    RERANK_EMPTY_CACHE_COUNT,
    RerankModel,
    RerankModelFamilyV2,
    RerankSpecV1,
)
from ..utils import preprocess_sentence

logger = logging.getLogger(__name__)


class _ModelWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.model = module
        self._local_data = threading.local()

    @property
    def n_tokens(self):
        return getattr(self._local_data, "n_tokens", 0)

    @n_tokens.setter
    def n_tokens(self, value):
        self._local_data.n_tokens = value

    @property
    def input_tokens(self):
        if not hasattr(self._local_data, "input_tokens"):
            self._local_data.input_tokens = []
        return self._local_data.input_tokens

    @input_tokens.setter
    def input_tokens(self, value):
        self._local_data.input_tokens = value

    @property
    def input_ids(self):
        if not hasattr(self._local_data, "input_ids"):
            self._local_data.input_ids = []
        return self._local_data.input_ids

    @input_ids.setter
    def input_ids(self, value):
        self._local_data.input_ids = value

    def forward(self, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        # when batching, the attention mask 1 means there is a token
        # thus we just sum up it to get the total number of tokens
        if attention_mask is not None:
            self.n_tokens += attention_mask.sum().item()
            per_sample_tokens = attention_mask.sum(dim=1).tolist()
            # sentence transformers always process batch size > 1
            self.input_tokens.extend(per_sample_tokens)
            self.input_ids.extend(kwargs.get("input_ids"))

        return self.model(**kwargs)

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return getattr(self.model, attr)


class SentenceTransformerRerankModel(RerankModel, BatchMixin):
    def __init__(self, *args, **kwargs) -> None:
        RerankModel.__init__(self, *args, **kwargs)
        BatchMixin.__init__(self, self.rerank, **kwargs)  # type: ignore
        self._vl_reranker = None

    def load(self):
        # TODO: Split FlagReranker and sentence_transformers into different model_engines like FlagRerankModel
        logger.info("Loading rerank model: %s", self._model_path)
        enable_flash_attn = self._kwargs.pop(
            "enable_flash_attn", is_flash_attn_available()
        )

        self._kwargs.pop("batch_size", None)
        self._kwargs.pop("batch_interval", None)

        if enable_flash_attn:
            logger.warning(
                "flash_attn can only support fp16 and bf16, will force set `use_fp16` to True"
            )
            self._use_fp16 = True

        if self.model_family.model_name.startswith("Qwen3-VL-Reranker"):
            module_path = os.path.join(
                self._model_path, "scripts", "qwen3_vl_reranker.py"
            )
            if not os.path.exists(module_path):
                raise FileNotFoundError(
                    f"Missing qwen3_vl_reranker.py under {self._model_path}. "
                    "Please verify the model repository files."
                )
            spec = importlib.util.spec_from_file_location(
                "qwen3_vl_reranker", module_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load reranker module from {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._vl_reranker = module.Qwen3VLReranker(
                model_name_or_path=self._model_path, **self._kwargs
            )
            return

        if (
            self.model_family.type == "normal"
            and "qwen3" not in self.model_family.model_name.lower()
            and "jina-reranker-v3" not in self.model_family.model_name.lower()
        ):
            try:
                import sentence_transformers
                from sentence_transformers.cross_encoder import CrossEncoder

                if sentence_transformers.__version__ < "3.1.0":
                    raise ValueError(
                        "The sentence_transformers version must be greater than 3.1.0. "
                        "Please upgrade your version via `pip install -U sentence_transformers` or refer to "
                        "https://github.com/UKPLab/sentence-transformers"
                    )
            except ImportError:
                error_message = "Failed to import module 'sentence-transformers'"
                installation_guide = [
                    "Please make sure 'sentence-transformers' is installed. ",
                    "You can install it by `pip install sentence-transformers`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
            self._model = CrossEncoder(
                self._model_path,
                device=self._device,
                trust_remote_code=True,
                max_length=getattr(self.model_family, "max_tokens"),
                **self._kwargs,
            )
            if self._use_fp16:
                self._model.model.half()
            self._tokenizer = self._model.tokenizer
        elif (
            "qwen3" in self.model_family.model_name.lower()
            or "jina-reranker-v3" in self.model_family.model_name.lower()
        ):
            # qwen3-reranker
            # now we use transformers
            # TODO: support engines for rerank models
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                error_message = "Failed to import module 'transformers'"
                installation_guide = [
                    "Please make sure 'transformers' is installed. ",
                    "You can install it by `pip install transformers`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

            tokenizer = AutoTokenizer.from_pretrained(
                self._model_path, padding_side="left"
            )
            model_kwargs = {"device_map": "auto"}
            if enable_flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                model_kwargs["torch_dtype"] = torch.float16
            model_kwargs.update(self._kwargs)
            logger.debug("Loading qwen3 rerank with kwargs %s", model_kwargs)
            model = self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path, **model_kwargs
            ).eval()
            max_length = getattr(self.model_family, "max_tokens")

            prefix = (
                "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
                'and the Instruct provided. Note that the answer can only be "yes" or "no".'
                "<|im_end|>\n<|im_start|>user\n"
            )
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

            def process_inputs(pairs):
                inputs = tokenizer(
                    pairs,
                    padding=False,
                    truncation="longest_first",
                    return_attention_mask=False,
                    max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
                )
                for i, ele in enumerate(inputs["input_ids"]):
                    inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
                inputs = tokenizer.pad(
                    inputs, padding=True, return_tensors="pt", max_length=max_length
                )
                for key in inputs:
                    inputs[key] = inputs[key].to(model.device)
                return inputs

            token_false_id = tokenizer.convert_tokens_to_ids("no")
            token_true_id = tokenizer.convert_tokens_to_ids("yes")

            @torch.inference_mode()
            def compute_logits(inputs, **kwargs):
                batch_scores = model(**inputs).logits[:, -1, :]
                true_vector = batch_scores[:, token_true_id]
                false_vector = batch_scores[:, token_false_id]
                batch_scores = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                scores = batch_scores[:, 1].exp().tolist()
                return scores

            self.process_inputs = process_inputs
            self.compute_logits = compute_logits

            self._tokenizer = tokenizer
        else:
            try:
                if self.model_family.type == "LLM-based":
                    from FlagEmbedding import FlagLLMReranker as FlagReranker
                elif self.model_family.type == "LLM-based layerwise":
                    from FlagEmbedding import LayerWiseFlagLLMReranker as FlagReranker
                else:
                    raise RuntimeError(
                        f"Unsupported Rank model type: {self.model_family.type}"
                    )
            except ImportError:
                error_message = "Failed to import module 'FlagEmbedding'"
                installation_guide = [
                    "Please make sure 'FlagEmbedding' is installed. ",
                    "You can install it by `pip install FlagEmbedding`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
            self._model = FlagReranker(self._model_path, use_fp16=self._use_fp16)
            self._tokenizer = self._model.tokenizer
        # Wrap transformers model to record number of tokens
        self._model.model = _ModelWrapper(self._model.model)

    def _rerank(
        self,
        documents: List[str],
        query: List[str],
        top_n: Optional[int],
        max_chunks_per_doc: Optional[int],
        return_documents: Optional[bool],
        return_len: Optional[bool],
        **kwargs,
    ) -> List[Any]:
        if self._vl_reranker is not None:
            return self._rerank_vl(documents, query, **kwargs)
        assert self._model is not None
        if max_chunks_per_doc is not None:
            raise ValueError("rerank hasn't support `max_chunks_per_doc` parameter.")
        logger.info("Rerank with kwargs: %s, model: %s", kwargs, self._model)

        sentence_combinations = [
            [
                preprocess_sentence(
                    pre_query,
                    kwargs.get("instruction", None),
                    self.model_family.model_name,
                ),
                doc,
            ]
            for doc, pre_query in zip(documents, query)
        ]
        # reset n tokens
        self._model.model.n_tokens = 0
        # reset input_tokens
        self._model.model.input_tokens = []
        # reset input_tokens
        self._model.model.input_ids = []
        if (
            self.model_family.type == "normal"
            and "qwen3" not in self.model_family.model_name.lower()
            and "jina-reranker-v3" not in self.model_family.model_name.lower()
        ):
            logger.debug("Passing processed sentences: %s", sentence_combinations)
            similarity_scores = self._model.predict(
                sentence_combinations,
                convert_to_numpy=False,
                convert_to_tensor=True,
                **kwargs,
            ).cpu()
            if similarity_scores.dtype == torch.bfloat16:
                similarity_scores = similarity_scores.float()
        elif (
            "qwen3" in self.model_family.model_name.lower()
            or "jina-reranker-v3" in self.model_family.model_name.lower()
        ):

            def format_instruction(instruction, query, doc):
                if instruction is None:
                    instruction = "Given a web search query, retrieve relevant passages that answer the query"
                output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                    instruction=instruction, query=query, doc=doc
                )
                return output

            # reduce memory usage.
            micro_bs = 4
            similarity_scores = []
            for i in range(0, len(documents), micro_bs):
                sub_docs = documents[i : i + micro_bs]
                sub_queries = query[i : i + micro_bs]
                pairs = [
                    format_instruction(kwargs.get("instruction", None), pre_query, doc)
                    for doc, pre_query in zip(sub_docs, sub_queries)
                ]
                # Tokenize the input texts
                inputs = self.process_inputs(pairs)
                similarity_scores.extend(self.compute_logits(inputs))
        else:
            # Related issue: https://github.com/xorbitsai/inference/issues/1775
            similarity_scores = self._model.compute_score(
                sentence_combinations, **kwargs
            )

            if not isinstance(similarity_scores, Sequence):
                similarity_scores = [similarity_scores]
            elif (
                isinstance(similarity_scores, list)
                and len(similarity_scores) > 0
                and isinstance(similarity_scores[0], Sequence)
            ):
                similarity_scores = similarity_scores[0]

        return similarity_scores

    def _normalize_vl_text(self, val: Any) -> Dict[str, Any]:
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            return {"text": val}
        raise ValueError("Unsupported input type for Qwen3-VL reranker.")

    def _rerank_vl(self, documents: List[Any], query: List[Any], **kwargs) -> List[Any]:
        if self._vl_reranker is None:
            raise RuntimeError("Qwen3-VL reranker is not initialized.")

        if len(query) == 0 or len(documents) == 0:
            return []

        query_obj = self._normalize_vl_text(query[0])
        doc_objs = [self._normalize_vl_text(doc) for doc in documents]

        payload: Dict[str, Any] = {
            "query": query_obj,
            "documents": doc_objs,
        }
        if "instruction" in kwargs:
            payload["instruction"] = kwargs.get("instruction")
        if "fps" in kwargs:
            payload["fps"] = kwargs.get("fps")
        if "max_frames" in kwargs:
            payload["max_frames"] = kwargs.get("max_frames")

        scores = self._vl_reranker.process(payload)
        return [float(score) for score in scores]

    @extensible
    def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int] = None,
        max_chunks_per_doc: Optional[int] = None,
        return_documents: Optional[bool] = True,
        return_len: Optional[bool] = False,
        **kwargs,
    ) -> Rerank:
        documents_size = len(documents)
        query_list = [query] * documents_size

        similarity_scores = self._rerank(
            documents,
            query_list,
            top_n,
            max_chunks_per_doc,
            return_documents,
            return_len,
            **kwargs,
        )
        sim_scores_argsort = list(reversed(np.argsort(similarity_scores)))
        if top_n is not None:
            sim_scores_argsort = sim_scores_argsort[:top_n]
        if return_documents:
            docs = [
                DocumentObj(
                    index=int(arg),
                    relevance_score=float(similarity_scores[arg]),
                    document=Document(text=documents[arg]),
                )
                for arg in sim_scores_argsort
            ]
        else:
            docs = [
                DocumentObj(
                    index=int(arg),
                    relevance_score=float(similarity_scores[arg]),
                    document=None,
                )
                for arg in sim_scores_argsort
            ]
        if return_len:
            input_len = self._model.model.n_tokens
            # Rerank Model output is just score or documents
            # while return_documents = True
            output_len = input_len

        # api_version, billed_units, warnings
        # is for Cohere API compatibility, set to None
        metadata = Meta(
            api_version=None,
            billed_units=None,
            tokens=(
                RerankTokens(input_tokens=input_len, output_tokens=output_len)
                if return_len
                else None
            ),
            warnings=None,
        )

        # clear cache if possible
        self._counter += 1
        if self._counter % RERANK_EMPTY_CACHE_COUNT == 0:
            logger.debug("Empty rerank cache.")
            gc.collect()
            empty_cache()

        return Rerank(id=str(uuid.uuid1()), results=docs, meta=metadata)

    @rerank.batch  # type: ignore
    def rerank(self, args_list, kwargs_list):
        grouped = defaultdict(
            lambda: {
                "documents": [],
                "query": [],
                "offsets": [],
                "kwargs": None,
                "indices": [],
            }
        )

        # 1. Group by kwargs hash
        for i, (args, kwargs) in enumerate(zip(args_list, kwargs_list)):

            documents, query, extra_kwargs = self._extract_rerank_kwargs(args, kwargs)

            key = make_hashable(extra_kwargs)
            group = grouped[key]
            group["kwargs"] = extra_kwargs

            current_offset = len(group["documents"])
            documents_size = len(documents)
            group["offsets"].append((current_offset, documents_size))
            group["documents"].extend(documents)
            group["query"].extend([query] * documents_size)
            group["indices"].append(i)  # remember original position

        results_with_index = []

        # 2. Process each group separately
        for key, group in grouped.items():
            documents = group["documents"]
            query = group["query"]
            kwargs = group["kwargs"]
            offsets = group["offsets"]
            indices = group["indices"]
            score_list = self._rerank(documents, query, **kwargs)
            top_n = kwargs.pop("top_n", None)
            return_documents = kwargs.pop("return_documents", None)
            return_len = kwargs.pop("return_len", None)

            # 3. Split and attach original index
            for (offset, n), idx in zip(offsets, indices):
                tmp_documents = group["documents"][offset : offset + n]
                tmp_queries = group["query"][offset : offset + n]
                data = score_list[offset : offset + n]
                sim_scores_argsort = list(reversed(np.argsort(data)))
                if top_n is not None:
                    sim_scores_argsort = sim_scores_argsort[:top_n]
                if return_documents:
                    docs = [
                        DocumentObj(
                            index=int(arg),
                            relevance_score=float(data[arg]),
                            document=Document(text=tmp_documents[arg]),
                        )
                        for arg in sim_scores_argsort
                    ]
                else:
                    docs = [
                        DocumentObj(
                            index=int(arg),
                            relevance_score=float(data[arg]),
                            document=None,
                        )
                        for arg in sim_scores_argsort
                    ]
                if return_len:
                    if (
                        self.model_family.type == "normal"
                        and "qwen3" not in self.model_family.model_name.lower()
                        and "jina-reranker-v3"
                        not in self.model_family.model_name.lower()
                    ):
                        input_len = sum(
                            self._model.model.input_tokens[offset : offset + n]
                        )
                    elif (
                        "qwen3" in self.model_family.model_name.lower()
                        or "jina-reranker-v3" in self.model_family.model_name.lower()
                    ):
                        input_len = sum(
                            self._model.model.input_tokens[offset : offset + n]
                        )
                    else:
                        # flagEmbedding reranker, forward twice
                        # input_ids was sorted, so we should traverse all prompt to match them
                        input_len = 0
                        for doc, q in zip(tmp_documents, tmp_queries):
                            for index, input_id in enumerate(
                                self._model.model.input_ids
                            ):
                                prompt = self._tokenizer.decode(input_id)
                                if self.flag_engine_match_query_doc(prompt, q, doc):
                                    input_len += (
                                        self._model.model.input_tokens[index] * 2
                                    )
                                    break
                    # Rerank Model output is just score or documents
                    # while return_documents = True
                    output_len = input_len

                # api_version, billed_units, warnings
                # is for Cohere API compatibility, set to None
                metadata = Meta(
                    api_version=None,
                    billed_units=None,
                    tokens=(
                        RerankTokens(input_tokens=input_len, output_tokens=output_len)
                        if return_len
                        else None
                    ),
                    warnings=None,
                )

                # clear cache if possible
                self._counter += 1
                if self._counter % RERANK_EMPTY_CACHE_COUNT == 0:
                    logger.debug("Empty rerank cache.")
                    gc.collect()
                    empty_cache()
                result = Rerank(id=str(uuid.uuid1()), results=docs, meta=metadata)
                results_with_index.append((idx, result))

        # 4. Sort by original call order
        results_with_index.sort(key=lambda x: x[0])
        results = [r for _, r in results_with_index]
        return results

    def _extract_rerank_kwargs(self, args, kwargs):
        """
        Extract the 'documents' and 'query' argument and remaining kwargs from (*args, **kwargs)
        for a given function.

        This uses inspect.signature(func).bind_partial() to automatically match
        both positional and keyword arguments, while handling bound methods
        (functions with 'self' as the first parameter).

        Args:
            func: The target function whose parameters define how to bind args/kwargs.
            args: The positional arguments passed to the function.
            kwargs: The keyword arguments passed to the function.

        Returns:
            A tuple (documents, query, extra_kwargs), where:
              - documents: The extracted 'documents' argument (never None).
              - query: The extracted 'query' argument (never None).
              - extra_kwargs: Remaining keyword arguments excluding 'documents' and 'query'.

        Raises:
            KeyError: If 'documents' or 'query' argument is not found.
            TypeError: If args/kwargs do not match the function signature.
        """
        sig = inspect.signature(self._rerank)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        if "documents" not in bound.arguments or "query" not in bound.arguments:
            raise KeyError("'documents' or 'query' argument not found in args/kwargs")

        documents = bound.arguments["documents"]
        query = bound.arguments["query"]

        extra_args = {
            k: v
            for k, v in bound.arguments.items()
            if k not in ("documents", "query", "kwargs")
        }
        extra_kwargs = {**extra_args, **bound.arguments.get("kwargs", {})}
        return documents, query, extra_kwargs

    def _get_batch_size(self, *args, **kwargs) -> int:
        reranks = self._extract_rerank_kwargs(args, kwargs)[0]
        if isinstance(reranks, list):
            return len(reranks)
        else:
            return 1

    @staticmethod
    def flag_engine_match_query_doc(prompt: str, query: str, doc: str) -> bool:
        """
        prompt: str, query: str, doc: str
        """
        try:
            a_start = prompt.index("A: ") + len("A: ")
        except ValueError:
            return False

        # from A: after exact match query
        a_slice = prompt[a_start : a_start + len(query)]
        if a_slice != query:
            return False

        doc_prompt = prompt[a_start + len(query) :]
        try:
            b_start = doc_prompt.index("B: ") + len("B: ")
        except ValueError:
            return False

        # from B: after exact match doc
        b_slice = doc_prompt[b_start : b_start + len(doc)]
        if b_slice != doc:
            return False
        sep = "\n"
        default_prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."

        remain_prompt = doc_prompt[b_start + len(doc) :]
        if remain_prompt != f" {sep} {default_prompt}":
            return False

        return True

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        return True

    @classmethod
    def match_json(
        cls,
        model_family: RerankModelFamilyV2,
        model_spec: RerankSpecV1,
        quantization: str,
    ) -> Union[bool, Tuple[bool, str]]:
        if model_family.model_name.startswith("Qwen3-VL-Reranker"):
            dep_check = check_dependency_available("transformers", "transformers")
            if dep_check != True:
                return dep_check
            dep_check = check_dependency_available("qwen_vl_utils", "qwen_vl_utils")
            if dep_check != True:
                return dep_check
            dep_check = check_dependency_available("PIL", "Pillow")
            if dep_check != True:
                return dep_check
            dep_check = check_dependency_available("scipy", "scipy")
            if dep_check != True:
                return dep_check
            if model_spec.model_format not in ["pytorch"]:
                return False, "Qwen3-VL reranker supports pytorch format only"
            return True
        dep_check = check_dependency_available(
            "sentence_transformers", "sentence-transformers"
        )
        if dep_check != True:
            return dep_check
        if model_spec.model_format not in ["pytorch"]:
            return False, "SentenceTransformer rerank engine requires pytorch format"
        return True
