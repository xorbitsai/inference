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
import os
import tempfile

import pytest
from click.testing import CliRunner

from ...client import Client
from ..cmdline import (
    list_cached_models,
    list_model_registrations,
    model_chat,
    model_generate,
    model_list,
    model_terminate,
    register_model,
    remove_cache,
    unregister_model,
)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("model_uid", [None, "my_model_uid"])
def test_cmdline(setup, stream, model_uid):
    endpoint, _ = setup
    runner = CliRunner()

    # launch model
    """
    result = runner.invoke(
        model_launch,
        [
            "--endpoint",
            endpoint,
            "--model-name",
            "tiny-llama",
            "--size-in-billions",
            1,
            "--model-format",
            "ggufv2",
            "--quantization",
            "Q2_K",
        ],
    )
    assert result.exit_code == 0
    assert "Model uid: " in result.stdout

    model_uid = result.stdout.split("Model uid: ")[1].strip()
    """
    # if use `model_launch` command to launch model, CI will fail.
    # So use client to launch model in temporary
    client = Client(endpoint)
    # CI has limited resources, use replica 1
    replica = 1
    original_model_uid = model_uid
    model_uid = client.launch_model(
        model_name="orca",
        model_engine="llama.cpp",
        model_uid=model_uid,
        model_size_in_billions=3,
        quantization="q4_0",
        replica=replica,
    )
    if original_model_uid == "my_model_uid":
        assert model_uid == "my_model_uid"
    assert len(model_uid) != 0

    # list model
    result = runner.invoke(
        model_list,
        [
            "--endpoint",
            endpoint,
        ],
    )
    assert result.exit_code == 0
    assert model_uid in result.stdout

    # model generate
    result = runner.invoke(
        model_generate,
        [
            "--endpoint",
            endpoint,
            "--model-uid",
            model_uid,
            "--stream",
            stream,
        ],
        input="Once upon a time, there was a very old computer.\n\n",
    )
    assert result.exit_code == 0
    assert len(result.stdout) != 0
    print(result.stdout)

    # model chat
    result = runner.invoke(
        model_chat,
        [
            "--endpoint",
            endpoint,
            "--model-uid",
            model_uid,
            "--stream",
            stream,
        ],
        input="Write a poem.\n\n",
    )
    assert result.exit_code == 0
    assert len(result.stdout) != 0
    print(result.stdout)

    # terminate model
    result = runner.invoke(
        model_terminate,
        [
            "--endpoint",
            endpoint,
            "--model-uid",
            model_uid,
        ],
    )
    assert result.exit_code == 0

    # list model again
    result = runner.invoke(
        model_list,
        [
            "--endpoint",
            endpoint,
        ],
    )
    assert result.exit_code == 0
    assert model_uid not in result.stdout


def test_cmdline_of_custom_model(setup):
    endpoint, _ = setup
    runner = CliRunner()

    # register custom model
    custom_model_desc = """{
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
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(custom_model_desc.encode("utf-8"))
    result = runner.invoke(
        register_model,
        [
            "--endpoint",
            endpoint,
            "--model-type",
            "LLM",
            "--file",
            temp_filename,
        ],
    )
    assert result.exit_code == 0
    os.unlink(temp_filename)

    # list model registrations
    result = runner.invoke(
        list_model_registrations,
        [
            "--endpoint",
            endpoint,
            "--model-type",
            "LLM",
        ],
    )
    assert result.exit_code == 0
    assert "custom_model" in result.stdout

    # unregister custom model
    result = runner.invoke(
        unregister_model,
        [
            "--endpoint",
            endpoint,
            "--model-type",
            "LLM",
            "--model-name",
            "custom_model",
        ],
    )
    assert result.exit_code == 0

    # list model registrations again
    result = runner.invoke(
        list_model_registrations,
        [
            "--endpoint",
            endpoint,
            "--model-type",
            "LLM",
        ],
    )
    assert result.exit_code == 0
    assert "custom_model" not in result.stdout


def test_rotate_logs(setup_with_file_logging):
    endpoint, _, log_file = setup_with_file_logging
    client = Client(endpoint)
    runner = CliRunner()
    replica = 1 if os.name == "nt" else 2
    model_uid = client.launch_model(
        model_name="orca",
        model_engine="llama.cpp",
        model_uid=None,
        model_size_in_billions=3,
        quantization="q4_0",
        replica=replica,
    )
    assert model_uid is not None
    assert len(model_uid) != 0

    # model generate
    result = runner.invoke(
        model_generate,
        [
            "--endpoint",
            endpoint,
            "--model-uid",
            model_uid,
            "--stream",
            False,
        ],
        input="Once upon a time, there was a very old computer.\n\n",
    )
    assert result.exit_code == 0
    assert len(result.stdout) != 0

    # test logs
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        content = f.read()
        assert len(content) > 0


def test_list_cached_models(setup):
    endpoint, _ = setup
    runner = CliRunner()

    result = runner.invoke(
        list_cached_models,
        ["--endpoint", endpoint, "--model_name", "orca"],
    )
    assert "model_name" in result.stdout
    assert "model_format" in result.stdout
    assert "model_size_in_billions" in result.stdout
    assert "quantization" in result.stdout
    assert "model_version" in result.stdout
    assert "path" in result.stdout
    assert "actor_ip_address" in result.stdout


def test_remove_cache(setup):
    endpoint, _ = setup
    runner = CliRunner()

    result = runner.invoke(
        remove_cache,
        ["--endpoint", endpoint, "--model_version", "orca"],
        input="y\n",
    )

    assert result.exit_code == 0
    assert "Cache directory orca has been deleted."
