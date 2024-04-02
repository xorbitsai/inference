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

import pytest

from ..restful.restful_client import Client as RESTfulClient
from ..restful.restful_client import RESTfulEmbeddingModelHandle


def test_client_auth(setup_with_auth):
    endpoint, _ = setup_with_auth
    client = RESTfulClient(endpoint)
    with pytest.raises(RuntimeError):
        client.list_models()

    client.login("user2", "pass2")
    assert len(client.list_models()) == 0

    with pytest.raises(RuntimeError):
        client.launch_model(model_name="bge-small-en-v1.5", model_type="embedding")

    client.login("user3", "pass3")
    model_uid = client.launch_model(
        model_name="bge-small-en-v1.5", model_type="embedding"
    )
    model = client.get_model(model_uid=model_uid)
    assert isinstance(model, RESTfulEmbeddingModelHandle)

    completion = model.create_embedding("write a poem.")
    assert len(completion["data"][0]["embedding"]) == 384

    with pytest.raises(RuntimeError):
        client.terminate_model(model_uid=model_uid)

    client.login("user1", "pass1")
    assert len(client.list_models()) == 1
    client.terminate_model(model_uid=model_uid)
    assert len(client.list_models()) == 0

    # test with api-key
    client = RESTfulClient(endpoint, api_key="sk-wrongapikey12")
    with pytest.raises(RuntimeError):
        client.list_models()

    client = RESTfulClient(endpoint, api_key="sk-72tkvudyGLPMi")
    assert len(client.list_models()) == 0

    with pytest.raises(RuntimeError):
        client.launch_model(model_name="bge-small-en-v1.5", model_type="embedding")

    client = RESTfulClient(endpoint, api_key="sk-ZOTLIY4gt9w11")
    model_uid = client.launch_model(
        model_name="bge-small-en-v1.5", model_type="embedding"
    )
    model = client.get_model(model_uid=model_uid)
    assert isinstance(model, RESTfulEmbeddingModelHandle)

    completion = model.create_embedding("write a poem.")
    assert len(completion["data"][0]["embedding"]) == 384

    with pytest.raises(RuntimeError):
        client.terminate_model(model_uid=model_uid)

    client = RESTfulClient(endpoint, api_key="sk-3sjLbdwqAhhAF")
    assert len(client.list_models()) == 1

    # test with openai SDK
    from openai import AuthenticationError, OpenAI, PermissionDeniedError

    client_ai = OpenAI(base_url=endpoint + "/v1", api_key="sk-wrongapikey12")
    with pytest.raises(AuthenticationError):
        client_ai.models.list()

    client_ai = OpenAI(base_url=endpoint + "/v1", api_key="sk-72tkvudyGLPMi")
    assert len(client_ai.models.list().data) == 1
    with pytest.raises(PermissionDeniedError):
        chat_completion = client_ai.embeddings.create(
            model="bge-small-en-v1.5",
            input="write a poem.",
        )

    client_ai = OpenAI(base_url=endpoint + "/v1", api_key="sk-ZOTLIY4gt9w11")
    chat_completion = client_ai.embeddings.create(
        model="bge-small-en-v1.5",
        input="write a poem.",
    )
    assert len(chat_completion.data[0].embedding) == 384

    client_ai = OpenAI(base_url=endpoint + "/v1", api_key="sk-3sjLbdwqAhhAF")
    client.terminate_model(model_uid)
    assert len(client.list_models()) == 0
    assert len(client_ai.models.list().data) == 0
