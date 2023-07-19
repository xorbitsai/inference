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


from typing import Any, List, Optional

from langchain.embeddings.base import Embeddings


class XinferenceEmbeddings(Embeddings):

    """Wrapper around xinference embedding models.

    To use, you should have the xinference library installed:

    .. code-block:: bash

        pip install xinference

    Check out: https://github.com/xorbitsai/inference

    To run, you need to start a Xinference supervisor on one server and Xinference workers on the other servers

    Example:
        Starting the supervisor:
        .. code-block:: bash

            $ xinference-supervisor

        Starting the worker:
        .. code-block:: bash
            $ xinference-worker

    Then you can access xinference embedding models.

    Example:
        .. code-block:: python

            from langchain.embeddings import XinferenceEmbeddings
            xinference = XinferenceEmbeddings(
                server_url="http://0.0.0.0:9997",
                model_name="orca",
                model_size_in_billions=3,
                quantization="q4_0",
                n_ctx=100,
                embedding="True"
            )

    Make sure to set embedding="True"

    """

    client: Any
    server_url: Optional[str] = None
    """Server URL to run the xinference server on"""
    model_name: Optional[str] = None
    """Model name to use. See 'xinference list --all' for all builtin models."""
    model_size_in_billions: Optional[int] = None
    """model size in billions"""
    model_format: Optional[str] = None
    """format of the model"""
    quantization: Optional[str] = None
    """quantization of the model"""
    model_kwargs: Optional[dict] = None

    def __init__(
        self,
        server_url: Optional[str] = None,
        model_name: Optional[str] = None,
        model_size_in_billions: Optional[int] = None,
        model_format: Optional[str] = None,
        quantization: Optional[str] = None,
        **kwargs: Any,
    ):
        try:
            from xinference.client import RESTfulClient
        except ImportError as e:
            raise ImportError(
                "Could not import xinference. Make sure to install it with "
                "'pip install xinference'"
            ) from e

        super().__init__()

        if model_name is None:
            raise ValueError(ValueError(f"Please provide the model name"))

        if server_url is None:
            raise ValueError(ValueError(f"Please provide server URL"))

        self.model_name = model_name
        self.model_size_in_billions = model_size_in_billions
        self.model_format = model_format
        self.quantization = quantization
        self.model_kwargs = kwargs
        self.client = RESTfulClient(server_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Xinference.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        model_uid = self.client.launch_model(
            self.model_name,
            self.model_size_in_billions,
            self.model_format,
            self.quantization,
            **(self.model_kwargs),
        )

        model = self.client.get_model(model_uid)

        embeddings = [
            model.create_embedding(text)["data"][0]["embedding"] for text in texts
        ]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query of documents using Xinference.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """

        model_uid = self.client.launch_model(
            self.model_name,
            self.model_size_in_billions,
            self.model_format,
            self.quantization,
            **(self.model_kwargs),
        )

        model = self.client.get_model(model_uid)

        embedding_res = model.create_embedding(text)

        embedding = embedding_res["data"][0]["embedding"]

        return list(map(float, embedding))


if __name__ == "__main__":
    xinference = XinferenceEmbeddings(
        server_url="http://0.0.0.0:9997",
        model_name="orca",
        model_size_in_billions=3,
        quantization="q4_0",
        n_ctx=100,
        embedding="True",
    )

    print(xinference.embed_query("This is a test query"))

    print(xinference.embed_documents(["text A", "test B"]))
