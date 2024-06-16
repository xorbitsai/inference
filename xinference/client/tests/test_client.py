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

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import psutil
import pytest
import requests

from ...constants import XINFERENCE_ENV_MODEL_SRC
from ..restful.restful_client import Client as RESTfulClient
from ..restful.restful_client import (
    RESTfulChatModelHandle,
    RESTfulEmbeddingModelHandle,
    _get_error_string,
)


@pytest.mark.skipif(os.name == "nt", reason="Skip windows")
def test_RESTful_client(setup):
    endpoint, _ = setup
    client = RESTfulClient(endpoint)
    assert len(client.list_models()) == 0

    model_uid = client.launch_model(
        model_name="orca",
        model_engine="llama.cpp",
        model_size_in_billions=3,
        quantization="q4_0",
    )
    assert len(client.list_models()) == 1

    model = client.get_model(model_uid=model_uid)
    assert isinstance(model, RESTfulChatModelHandle)

    with pytest.raises(RuntimeError):
        model = client.get_model(model_uid="test")
        assert isinstance(model, RESTfulChatModelHandle)

    with pytest.raises(RuntimeError):
        completion = model.generate({"max_tokens": 64})

    completion = model.generate("Once upon a time, there was a very old computer")
    assert "text" in completion["choices"][0]

    completion = model.generate(
        "Once upon a time, there was a very old computer", {"max_tokens": 64}
    )
    assert "text" in completion["choices"][0]

    streaming_response = model.generate(
        "Once upon a time, there was a very old computer",
        {"max_tokens": 64, "stream": True},
    )

    for chunk in streaming_response:
        assert "text" in chunk["choices"][0]

    with pytest.raises(RuntimeError):
        completion = model.chat({"max_tokens": 64})

    completion = model.chat("What is the capital of France?")
    assert "content" in completion["choices"][0]["message"]

    def _check_stream():
        streaming_response = model.chat(
            prompt="What is the capital of France?",
            generate_config={"stream": True, "max_tokens": 5},
        )
        for chunk in streaming_response:
            assert ("content" in chunk["choices"][0]["delta"]) or (
                "role" in chunk["choices"][0]["delta"]
            )

    _check_stream()

    results = []
    with ThreadPoolExecutor() as executor:
        for _ in range(2):
            r = executor.submit(_check_stream)
            results.append(r)
    # Parallel generation is not supported by ggml.
    error_count = 0
    for r in results:
        try:
            r.result()
        except Exception as ex:
            assert "Parallel generation" in str(ex)
            error_count += 1
    assert error_count == 1

    # After iteration finish, we can iterate again.
    _check_stream()

    client.terminate_model(model_uid=model_uid)
    assert len(client.list_models()) == 0

    model_uid = client.launch_model(
        model_name="tiny-llama",
        model_engine="llama.cpp",
        model_size_in_billions=1,
        model_format="ggufv2",
        quantization="q2_K",
    )
    assert len(client.list_models()) == 1

    # Test concurrent chat is OK.
    model = client.get_model(model_uid=model_uid)

    def _check(stream=False):
        completion = model.generate(
            "AI is going to", generate_config={"stream": stream, "max_tokens": 5}
        )
        if stream:
            for chunk in completion:
                assert "text" in chunk["choices"][0]
                assert (
                    chunk["choices"][0]["text"] or chunk["choices"][0]["finish_reason"]
                )
        else:
            assert "text" in completion["choices"][0]
            assert len(completion["choices"][0]["text"]) > 0

    for stream in [True, False]:
        results = []
        error_count = 0
        with ThreadPoolExecutor() as executor:
            for _ in range(3):
                r = executor.submit(_check, stream=stream)
                results.append(r)
        for r in results:
            try:
                r.result()
            except Exception as ex:
                assert "Parallel generation" in str(ex)
                error_count += 1
        assert error_count == (2 if stream else 0)

    client.terminate_model(model_uid=model_uid)
    assert len(client.list_models()) == 0

    with pytest.raises(RuntimeError):
        client.terminate_model(model_uid=model_uid)

    model_uid2 = client.launch_model(
        model_name="orca",
        model_engine="llama.cpp",
        model_size_in_billions=3,
        quantization="q4_0",
    )

    client.terminate_model(model_uid=model_uid2)
    assert len(client.list_models()) == 0


@pytest.mark.skipif(os.name == "nt", reason="Skip windows")
def test_list_cached_models(setup):
    endpoint, _ = setup
    client = RESTfulClient(endpoint)
    res = client.list_cached_models()
    assert len(res) > 0


@pytest.mark.skipif(os.name == "nt", reason="Skip windows")
def test_list_deletable_models(setup):
    endpoint, local_host = setup
    client = RESTfulClient(endpoint)
    response = client.list_deletable_models("orca--3B--ggmlv3--q4_0")
    paths = response.get("paths", [])

    expected_path = os.path.join(
        os.environ["HOME"],
        ".xinference",
        "cache",
        "orca-ggmlv3-3b",
        "orca-mini-3b.ggmlv3.q4_0.bin",
    )

    normalized_expected_path = os.path.normpath(expected_path)

    assert normalized_expected_path in paths


@pytest.mark.skipif(os.name == "nt", reason="Skip windows")
def test_remove_cached_models(setup):
    endpoint, local_host = setup
    client = RESTfulClient(endpoint)
    responses = client.confirm_and_remove_model("orca--3B--ggmlv3--q4_0")
    assert responses


def test_RESTful_client_for_embedding(setup):
    endpoint, _ = setup
    client = RESTfulClient(endpoint)
    assert len(client.list_models()) == 0

    model_uid = client.launch_model(model_name="gte-base", model_type="embedding")
    assert len(client.list_models()) == 1

    model = client.get_model(model_uid=model_uid)
    assert isinstance(model, RESTfulEmbeddingModelHandle)

    completion = model.create_embedding("write a poem.")
    assert len(completion["data"][0]["embedding"]) == 768

    kwargs = {
        "invalid": "invalid",
    }
    with pytest.raises(RuntimeError) as err:
        completion = model.create_embedding("write a poem.", **kwargs)
    assert "unexpected" in str(err.value)

    client.terminate_model(model_uid=model_uid)
    assert len(client.list_models()) == 0


def test_RESTful_client_custom_model(setup):
    endpoint, _ = setup
    client = RESTfulClient(endpoint)

    model_regs = client.list_model_registrations(model_type="LLM")
    assert len(model_regs) > 0
    for model_reg in model_regs:
        assert model_reg["is_builtin"]

    model = """{
  "version": 1,
  "context_length":2048,
  "model_name": "custom_model",
  "model_lang": [
    "en", "zh"
  ],
  "model_ability": [
    "embed",
    "chat"
  ],
  "model_family": "other",
  "model_specs": [
    {
      "model_format": "pytorch",
      "model_size_in_billions": 7,
      "quantizations": [
        "4-bit",
        "8-bit",
        "none"
      ],
      "model_id": "ziqingyang/chinese-alpaca-2-7b"
    }
  ],
  "prompt_style": {
    "style_name": "ADD_COLON_SINGLE",
    "system_prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
    "roles": [
      "Instruction",
      "Response"
    ],
    "intra_message_sep": "\\n\\n### "
  }
}"""
    client.register_model(model_type="LLM", model=model, persist=False)

    data = client.get_model_registration(model_type="LLM", model_name="custom_model")
    assert "custom_model" in data["model_name"]

    new_model_regs = client.list_model_registrations(model_type="LLM")
    assert len(new_model_regs) == len(model_regs) + 1
    custom_model_reg = None
    for model_reg in new_model_regs:
        if model_reg["model_name"] == "custom_model":
            custom_model_reg = model_reg
    assert custom_model_reg is not None

    client.unregister_model(model_type="LLM", model_name="custom_model")
    new_model_regs = client.list_model_registrations(model_type="LLM")
    assert len(new_model_regs) == len(model_regs)
    custom_model_reg = None
    for model_reg in new_model_regs:
        if model_reg["model_name"] == "custom_model":
            custom_model_reg = model_reg
    assert custom_model_reg is None

    # test register with string prompt style name
    model_with_prompt = """{
  "version": 1,
  "context_length":2048,
  "model_name": "custom_model",
  "model_lang": [
    "en", "zh"
  ],
  "model_ability": [
    "embed",
    "chat"
  ],
  "model_family": "other",
  "model_specs": [
    {
      "model_format": "pytorch",
      "model_size_in_billions": 7,
      "quantizations": [
        "4-bit",
        "8-bit",
        "none"
      ],
      "model_id": "ziqingyang/chinese-alpaca-2-7b"
    }
  ],
  "prompt_style": "qwen-chat"
}"""
    client.register_model(model_type="LLM", model=model_with_prompt, persist=False)
    client.unregister_model(model_type="LLM", model_name="custom_model")

    model_with_prompt2 = """{
      "version": 1,
      "context_length":2048,
      "model_name": "custom_model",
      "model_lang": [
        "en", "zh"
      ],
      "model_ability": [
        "embed",
        "chat"
      ],
      "model_family": "other",
      "model_specs": [
        {
          "model_format": "pytorch",
          "model_size_in_billions": 7,
          "quantizations": [
            "4-bit",
            "8-bit",
            "none"
          ],
          "model_id": "ziqingyang/chinese-alpaca-2-7b"
        }
      ],
      "prompt_style": "xyz123"
    }"""
    with pytest.raises(RuntimeError):
        client.register_model(model_type="LLM", model=model_with_prompt2, persist=False)


def test_client_from_modelscope(setup):
    try:
        os.environ[XINFERENCE_ENV_MODEL_SRC] = "modelscope"

        endpoint, _ = setup
        client = RESTfulClient(endpoint)
        assert len(client.list_models()) == 0

        model_uid = client.launch_model(
            model_name="tiny-llama", model_engine="llama.cpp"
        )
        assert len(client.list_models()) == 1
        model = client.get_model(model_uid=model_uid)
        completion = model.generate("write a poem.", generate_config={"max_tokens": 5})
        assert "text" in completion["choices"][0]
        assert len(completion["choices"][0]["text"]) > 0
    finally:
        os.environ.pop(XINFERENCE_ENV_MODEL_SRC)


def test_client_error():
    r = requests.Response()
    r.url = "0.0.0.0:1234"
    r.status_code = 502
    r.reason = "Bad Gateway"
    err = _get_error_string(r)
    assert "502 Server Error: Bad Gateway for url: 0.0.0.0:1234" == err
    r._content = json.dumps({"detail": "Test error"}).encode("utf-8")
    err = _get_error_string(r)
    assert "Test error" == err


def test_client_custom_embedding_model(setup):
    endpoint, _ = setup
    client = RESTfulClient(endpoint)

    model_regs = client.list_model_registrations(model_type="embedding")
    assert len(model_regs) > 0
    for model_reg in model_regs:
        assert model_reg["is_builtin"]

    model = """{
  "model_name": "custom-bge-small-en",
  "dimensions": 1024,
  "max_tokens": 512,
  "language": ["en"],
  "model_id": "Xorbits/bge-small-en"
}"""
    client.register_model(model_type="embedding", model=model, persist=False)

    data = client.get_model_registration(
        model_type="embedding", model_name="custom-bge-small-en"
    )
    assert "custom-bge-small-en" in data["model_name"]

    new_model_regs = client.list_model_registrations(model_type="embedding")
    assert len(new_model_regs) == len(model_regs) + 1
    custom_model_reg = None
    for model_reg in new_model_regs:
        if model_reg["model_name"] == "custom-bge-small-en":
            custom_model_reg = model_reg
    assert custom_model_reg is not None
    assert not custom_model_reg["is_builtin"]

    # unregister
    client.unregister_model(model_type="embedding", model_name="custom-bge-small-en")
    new_model_regs = client.list_model_registrations(model_type="embedding")
    assert len(new_model_regs) == len(model_regs)
    custom_model_reg = None
    for model_reg in new_model_regs:
        if model_reg["model_name"] == "custom-bge-small-en":
            custom_model_reg = model_reg
    assert custom_model_reg is None


@pytest.fixture
def set_auto_recover_limit():
    os.environ["XINFERENCE_MODEL_ACTOR_AUTO_RECOVER_LIMIT"] = "1"
    yield
    del os.environ["XINFERENCE_MODEL_ACTOR_AUTO_RECOVER_LIMIT"]


@pytest.fixture
def setup_cluster():
    import xoscar as xo

    from ...api.restful_api import run_in_subprocess as restful_api_run_in_subprocess
    from ...conftest import TEST_FILE_LOGGING_CONF, TEST_LOGGING_CONF, api_health_check
    from ...deploy.local import health_check
    from ...deploy.local import run_in_subprocess as supervisor_run_in_subprocess

    supervisor_address = f"localhost:{xo.utils.get_next_port()}"
    local_cluster = supervisor_run_in_subprocess(
        supervisor_address, None, None, TEST_LOGGING_CONF
    )

    if not health_check(address=supervisor_address, max_attempts=20, sleep_interval=1):
        raise RuntimeError("Supervisor is not available after multiple attempts")

    try:
        port = xo.utils.get_next_port()
        restful_api_proc = restful_api_run_in_subprocess(
            supervisor_address,
            host="localhost",
            port=port,
            logging_conf=TEST_FILE_LOGGING_CONF,
        )
        endpoint = f"http://localhost:{port}"
        if not api_health_check(endpoint, max_attempts=10, sleep_interval=5):
            raise RuntimeError("Endpoint is not available after multiple attempts")

        yield f"http://localhost:{port}", supervisor_address
        restful_api_proc.kill()
    finally:
        local_cluster.kill()


def test_auto_recover(set_auto_recover_limit, setup_cluster):
    endpoint, _ = setup_cluster
    current_proc = psutil.Process()
    chilren_proc = set(current_proc.children(recursive=True))
    client = RESTfulClient(endpoint)

    model_uid = client.launch_model(
        model_name="orca",
        model_engine="llama.cpp",
        model_size_in_billions=3,
        quantization="q4_0",
    )
    new_children_proc = set(current_proc.children(recursive=True))
    model_proc = next(iter(new_children_proc - chilren_proc))
    assert len(client.list_models()) == 1

    model = client.get_model(model_uid=model_uid)
    assert isinstance(model, RESTfulChatModelHandle)

    completion = model.generate("Once upon a time, there was a very old computer")
    assert "text" in completion["choices"][0]

    model_proc.kill()

    for _ in range(60):
        try:
            completion = model.generate(
                "Once upon a time, there was a very old computer", {"max_tokens": 64}
            )
            assert "text" in completion["choices"][0]
            break
        except Exception:
            time.sleep(1)
    else:
        assert False

    new_children_proc = set(current_proc.children(recursive=True))
    model_proc = next(iter(new_children_proc - chilren_proc))
    assert len(client.list_models()) == 1

    model_proc.kill()

    expect_failed = False
    for _ in range(5):
        try:
            completion = model.generate(
                "Once upon a time, there was a very old computer", {"max_tokens": 64}
            )
            assert "text" in completion["choices"][0]
            break
        except Exception:
            time.sleep(1)
    else:
        expect_failed = True
    assert expect_failed
