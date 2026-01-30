import gc
import inspect
import logging
import uuid
from collections import defaultdict
from typing import Any, List, Optional, Tuple, Union

from xoscar import extensible

from ....device_utils import empty_cache, is_vacc_available
from ....types import Document, DocumentObj, Meta, Rerank, RerankTokens
from ....utils import make_hashable
from ...batch import BatchMixin
from ...utils import check_dependency_available
from ..core import (
    RERANK_EMPTY_CACHE_COUNT,
    RerankModel,
    RerankModelFamilyV2,
    RerankSpecV1,
)

logger = logging.getLogger(__name__)
SUPPORTED_MODELS_PREFIXES = ["bge", "gte", "text2vec", "m3e", "gte", "Qwen3"]


class VLLMRerankModel(RerankModel, BatchMixin):
    def __init__(self, *args, **kwargs) -> None:
        RerankModel.__init__(self, *args, **kwargs)
        BatchMixin.__init__(self, self.rerank, **kwargs)  # type: ignore

    def load(self):
        try:
            if is_vacc_available():
                import vllm_vacc  # noqa: F401
            from packaging.version import Version
            from vllm import LLM
            from vllm import __version__ as vllm_version

        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        self._kwargs.pop("batch_size", None)
        self._kwargs.pop("batch_interval", None)

        if self.model_family.model_name in {
            "Qwen3-Reranker-0.6B",
            "Qwen3-Reranker-4B",
            "Qwen3-Reranker-8B",
        }:
            if "hf_overrides" not in self._kwargs:
                self._kwargs["hf_overrides"] = {
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "classifier_from_token": ["no", "yes"],
                    "is_original_qwen3_reranker": True,
                }
            elif isinstance(self._kwargs["hf_overrides"], dict):
                self._kwargs["hf_overrides"].update(
                    architectures=["Qwen3ForSequenceClassification"],
                    classifier_from_token=["no", "yes"],
                    is_original_qwen3_reranker=True,
                )
        if Version(vllm_version) >= Version("0.13.0"):
            self._model = LLM(model=self._model_path, **self._kwargs)
        else:
            self._model = LLM(model=self._model_path, task="score", **self._kwargs)
        self._tokenizer = self._model.get_tokenizer()

    def _rerank(
        self,
        documents: List[str],
        query: Union[str, List[str]],
        top_n: Optional[int] = None,
        max_chunks_per_doc: Optional[int] = None,
        return_documents: Optional[bool] = None,
        return_len: Optional[bool] = None,
        **kwargs,
    ) -> list[Any]:
        """
        Rerank the documents based on the query using the VLLM model.

        Args:
            documents (List[str]): List of documents to be reranked.
            query (str): The query string to rank the documents against.
            top_n (Optional[int]): The number of top documents to return.
            max_chunks_per_doc (Optional[int]): Maximum chunks per document.
            return_documents (Optional[bool]): Whether to return the documents.
            return_len (Optional[bool]): Whether to return the length of the documents.

        Returns:
            Rerank: The reranked results.
        """
        if kwargs:
            raise RuntimeError("Unexpected keyword arguments: {}".format(kwargs))
        assert self._model is not None

        documents_size = len(documents)
        if isinstance(query, str):
            query_list = [query] * documents_size
        else:
            query_list = query

        if self.model_family.model_name in {
            "Qwen3-Reranker-0.6B",
            "Qwen3-Reranker-4B",
            "Qwen3-Reranker-8B",
        }:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
            prefix = (
                "<|im_start|>system\nJudge whether the Document meets the requirements based on"
                " the Query and the Instruct provided. "
                'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            )
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
            document_template = "<Document>: {doc}{suffix}"
            processed_queries = [
                query_template.format(
                    prefix=prefix, instruction=instruction, query=query
                )
                for query in query_list
            ]
            processed_documents = [
                document_template.format(doc=doc, suffix=suffix) for doc in documents
            ]
            outputs = self._model.score(
                processed_documents,
                processed_queries,
                use_tqdm=False,
            )

        else:
            outputs = self._model.score(
                documents,
                query_list,
                use_tqdm=False,
            )
        # clear cache if possible
        self._counter += 1
        if self._counter % RERANK_EMPTY_CACHE_COUNT == 0:
            logger.debug("Empty rerank cache.")
            gc.collect()
            empty_cache()
        return outputs

    @extensible
    def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int] = None,
        max_chunks_per_doc: Optional[int] = None,
        return_documents: Optional[bool] = None,
        return_len: Optional[bool] = None,
        **kwargs,
    ) -> Rerank:
        """
        Rerank the documents based on the query using the VLLM model.

        Args:
            documents (List[str]): List of documents to be reranked.
            query (str): The query string to rank the documents against.
            top_n (Optional[int]): The number of top documents to return.
            max_chunks_per_doc (Optional[int]): Maximum chunks per document.
            return_documents (Optional[bool]): Whether to return the documents.
            return_len (Optional[bool]): Whether to return the length of the documents.

        Returns:
            Rerank: The reranked results.
        """
        documents_size = len(documents)
        query_list = [query] * documents_size
        outputs = self._rerank(
            documents,
            query_list,
            top_n,
            max_chunks_per_doc,
            return_documents,
            return_len,
            **kwargs,
        )
        scores = map(lambda scoreoutput: scoreoutput.outputs.score, outputs)
        documents = list(map(lambda doc: Document(text=doc), documents))
        document_parts = list(zip(range(documents_size), scores, documents))
        document_parts.sort(key=lambda x: x[1], reverse=True)
        if top_n is not None:
            document_parts = document_parts[:top_n]
        reranked_docs = list(
            map(
                lambda doc: DocumentObj(
                    index=doc[0],
                    relevance_score=doc[1],
                    document=doc[2] if return_documents else None,
                ),
                document_parts,
            )
        )
        tokens = sum(map(lambda x: len(x.prompt_token_ids), outputs))
        metadata = Meta(
            api_version=None,
            billed_units=None,
            tokens=(
                RerankTokens(input_tokens=tokens, output_tokens=tokens)
                if return_len
                else None
            ),
            warnings=None,
        )
        return Rerank(id=str(uuid.uuid4()), results=reranked_docs, meta=metadata)

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
                data = score_list[offset : offset + n]
                scores = map(lambda scoreoutput: scoreoutput.outputs.score, data)
                tmp_documents = list(map(lambda doc: Document(text=doc), tmp_documents))
                document_parts = list(zip(range(n), scores, tmp_documents))
                document_parts.sort(key=lambda x: x[1], reverse=True)
                if top_n is not None:
                    document_parts = document_parts[:top_n]
                reranked_docs = list(
                    map(
                        lambda doc: DocumentObj(
                            index=doc[0],
                            relevance_score=doc[1],
                            document=doc[2] if return_documents else None,
                        ),
                        document_parts,
                    )
                )
                tokens = sum(map(lambda x: len(x.prompt_token_ids), data))
                metadata = Meta(
                    api_version=None,
                    billed_units=None,
                    tokens=(
                        RerankTokens(input_tokens=tokens, output_tokens=tokens)
                        if return_len
                        else None
                    ),
                    warnings=None,
                )
                result = Rerank(
                    id=str(uuid.uuid4()), results=reranked_docs, meta=metadata
                )
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

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        dep_check = check_dependency_available("vllm", "vLLM")
        if dep_check != True:
            return dep_check
        return True

    @classmethod
    def match_json(
        cls,
        model_family: RerankModelFamilyV2,
        model_spec: RerankSpecV1,
        quantization: str,
    ) -> Union[bool, Tuple[bool, str]]:
        if model_family.model_name.startswith("Qwen3-VL-Reranker"):
            return False, "Qwen3-VL reranker requires vLLM>=0.14.0"
        if model_spec.model_format not in ["pytorch"]:
            return False, "vLLM rerank engine only supports pytorch format"
        prefix = model_family.model_name.split("-", 1)[0]
        if prefix not in SUPPORTED_MODELS_PREFIXES:
            return (
                False,
                f"Model family {model_family.model_name} is not supported by vLLM rerank engine",
            )
        return True
