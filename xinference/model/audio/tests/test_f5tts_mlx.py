# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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


@pytest.fixture
def setup_real_cluster():
    """Use a real actor pool so the model process can enter its virtualenv.

    The shared ``setup`` fixture uses xoscar's in-process ``test://`` backend,
    which accepts ``start_python`` but cannot switch interpreters for subpools.
    """
    import xoscar as xo

    from ....api.restful_api import run_in_subprocess as run_restful_api
    from ....conftest import TEST_LOGGING_CONF, api_health_check
    from ....deploy.local import health_check
    from ....deploy.local import run_in_subprocess as run_local_cluster

    os.environ["XINFERENCE_AUTH_ADVANCED"] = "false"
    supervisor_address = f"localhost:{xo.utils.get_next_port()}"
    local_cluster = run_local_cluster(supervisor_address, None, None, TEST_LOGGING_CONF)
    restful_api = None
    try:
        if not health_check(
            address=supervisor_address, max_attempts=20, sleep_interval=1
        ):
            raise RuntimeError("Supervisor is not available after multiple attempts")

        port = xo.utils.get_next_port()
        endpoint = f"http://localhost:{port}"
        restful_api = run_restful_api(
            supervisor_address,
            host="localhost",
            port=port,
            logging_conf=TEST_LOGGING_CONF,
        )
        if not api_health_check(endpoint, max_attempts=10, sleep_interval=5):
            raise RuntimeError("Endpoint is not available after multiple attempts")

        yield endpoint, supervisor_address
    finally:
        if restful_api is not None:
            restful_api.kill()
        local_cluster.kill()


def test_f5tts_mlx(setup_real_cluster):
    endpoint, _ = setup_real_cluster
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="F5-TTS",
        model_type="audio",
        model_engine="MLX",
        download_hub="huggingface",
        enable_virtual_env=True,
    )
    model = client.get_model(model_uid)
    input_string = (
        "chat T T S is a text to speech model designed for dialogue applications."
    )
    response = model.speech(input_string)
    assert type(response) is bytes
    assert len(response) > 0

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
        f.write(response)

    # Test openai API
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    with client.audio.speech.with_streaming_response.create(
        model=model_uid, input=input_string, voice="echo"
    ) as response:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
            response.stream_to_file(f.name)
            assert os.stat(f.name).st_size > 0
