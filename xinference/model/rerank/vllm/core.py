import importlib.util
import uuid
from typing import List, Optional

from ....types import Document, DocumentObj, Meta, Rerank, RerankTokens
from ...utils import cache_clean
from ..core import RerankModel, RerankModelFamilyV2, RerankSpecV1

SUPPORTED_MODELS_PREFIXES = ["bge", "gte", "text2vec", "m3e", "gte", "Qwen3"]


class VLLMRerankModel(RerankModel):
    def load(self):
        try:
            from vllm import LLM

        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

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
        self._model = LLM(model=self._model_path, task="score", **self._kwargs)
        self._tokenizer = self._model.get_tokenizer()

    @cache_clean
    def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int],
        max_chunks_per_doc: Optional[int],
        return_documents: Optional[bool],
        return_len: Optional[bool],
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
        if kwargs:
            raise RuntimeError("Unexpected keyword arguments: {}".format(kwargs))
        assert self._model is not None
        documents_size = len(documents)
        query_list = [query] * documents_size

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

    @classmethod
    def check_lib(cls) -> bool:
        return importlib.util.find_spec("vllm") is not None

    @classmethod
    def match_json(
        cls,
        model_family: RerankModelFamilyV2,
        model_spec: RerankSpecV1,
        quantization: str,
    ) -> bool:
        if model_spec.model_format in ["pytorch"]:
            prefix = model_family.model_name.split("-", 1)[0]
            if prefix in SUPPORTED_MODELS_PREFIXES:
                return True
        return False
