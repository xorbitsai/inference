import importlib
import uuid
from typing import Optional, List

from ....types import Rerank, DocumentObj, Document, Meta, RerankTokens
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

        self._model = LLM(model=self._model_path, task="score")
        self._tokenizer = self._model.get_tokenizer()

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
        documents_size = len(documents)
        model = self._model
        query_list = [query] * documents_size
        outputs = model.score(
            documents,
            query_list,
            use_tqdm=False,
        )
        scores = map(lambda x: x.outputs.score, outputs)
        documents = map(lambda x: Document(text=x), documents)
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
            tokens=RerankTokens(input_tokens=tokens, output_tokens=tokens)
            if return_len
            else None,
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
