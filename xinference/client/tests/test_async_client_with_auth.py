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

import pytest

from ..restful.async_restful_client import AsyncClient as AsyncRESTfulClient
from ..restful.async_restful_client import AsyncRESTfulEmbeddingModelHandle


@pytest.mark.asyncio
async def test_async_client_auth(setup_with_auth):
    endpoint, _ = setup_with_auth
    client = AsyncRESTfulClient(endpoint)
    with pytest.raises(RuntimeError):
        await client.list_models()

    await client.login("user2", "pass2")
    assert len(await client.list_models()) == 0

    with pytest.raises(RuntimeError):
        await client.launch_model(
            model_name="bge-small-en-v1.5", model_type="embedding"
        )

    await client.login("user3", "pass3")
    model_uid = await client.launch_model(
        model_name="bge-small-en-v1.5", model_type="embedding"
    )
    model = await client.get_model(model_uid=model_uid)
    assert isinstance(model, AsyncRESTfulEmbeddingModelHandle)

    completion = await model.create_embedding("write a poem.")
    assert len(completion["data"][0]["embedding"]) == 384

    with pytest.raises(RuntimeError):
        await client.terminate_model(model_uid=model_uid)

    await client.login("user1", "pass1")
    assert len(await client.list_models()) == 1
    await client.terminate_model(model_uid=model_uid)
    assert len(await client.list_models()) == 0

    # test with api-key
    client = AsyncRESTfulClient(endpoint, api_key="sk-wrongapikey12")
    with pytest.raises(RuntimeError):
        await client.list_models()

    client = AsyncRESTfulClient(endpoint, api_key="sk-72tkvudyGLPMi")
    assert len(await client.list_models()) == 0

    with pytest.raises(RuntimeError):
        await client.launch_model(
            model_name="bge-small-en-v1.5", model_type="embedding"
        )

    client = AsyncRESTfulClient(endpoint, api_key="sk-ZOTLIY4gt9w11")
    model_uid = await client.launch_model(
        model_name="bge-small-en-v1.5", model_type="embedding"
    )
    model = await client.get_model(model_uid=model_uid)
    assert isinstance(model, AsyncRESTfulEmbeddingModelHandle)

    completion = await model.create_embedding("write a poem.")
    assert len(completion["data"][0]["embedding"]) == 384

    with pytest.raises(RuntimeError):
        await client.terminate_model(model_uid=model_uid)

    client = AsyncRESTfulClient(endpoint, api_key="sk-3sjLbdwqAhhAF")
    assert len(await client.list_models()) == 1

    # test with openai SDK
    from openai import AsyncOpenAI, AuthenticationError, PermissionDeniedError

    client_ai = AsyncOpenAI(base_url=endpoint + "/v1", api_key="sk-wrongapikey12")
    with pytest.raises(AuthenticationError):
        await client_ai.models.list()

    client_ai = AsyncOpenAI(base_url=endpoint + "/v1", api_key="sk-72tkvudyGLPMi")
    assert len((await client_ai.models.list()).data) == 1
    with pytest.raises(PermissionDeniedError):
        chat_completion = await client_ai.embeddings.create(
            model="bge-small-en-v1.5",
            input="write a poem.",
        )

    client_ai = AsyncOpenAI(base_url=endpoint + "/v1", api_key="sk-ZOTLIY4gt9w11")
    chat_completion = await client_ai.embeddings.create(
        model="bge-small-en-v1.5",
        input="write a poem.",
    )
    assert len(chat_completion.data[0].embedding) == 384

    client_ai = AsyncOpenAI(base_url=endpoint + "/v1", api_key="sk-3sjLbdwqAhhAF")
    await client.terminate_model(model_uid)
    assert len(await client.list_models()) == 0
    assert len((await client_ai.models.list()).data) == 0
    await client.close()
