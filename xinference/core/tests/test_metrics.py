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
import requests


@pytest.fixture
def setup_cluster():
    import xoscar as xo

    from ...api.restful_api import run_in_subprocess as restful_api_run_in_subprocess
    from ...conftest import TEST_FILE_LOGGING_CONF, TEST_LOGGING_CONF, api_health_check
    from ...deploy.local import health_check
    from ...deploy.local import run_in_subprocess as supervisor_run_in_subprocess

    metrics_port = xo.utils.get_next_port()
    supervisor_address = f"localhost:{xo.utils.get_next_port()}"
    local_cluster = supervisor_run_in_subprocess(
        supervisor_address, "localhost", metrics_port, TEST_LOGGING_CONF
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

        yield f"http://localhost:{port}", f"http://localhost:{metrics_port}/metrics", supervisor_address
        restful_api_proc.kill()
    finally:
        local_cluster.kill()


@pytest.mark.asyncio
async def test_metrics_exporter_server(setup_cluster):
    endpoint, metrics_exporter_address, supervisor_address = setup_cluster

    import xoscar as xo

    from ...client import Client
    from ..supervisor import SupervisorActor

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="orca",
        model_engine="llama-cpp-python",
        model_size_in_billions=3,
        quantization="q4_0",
    )

    # Check the supervisor metrics collected the RESTful API.
    supervisor_ref = await xo.actor_ref(supervisor_address, SupervisorActor.uid())
    response = requests.get(f"{endpoint}/metrics")
    assert response.ok
    assert "/v1/models" in response.text

    # Check the worker metrics collected model metrics.
    model_ref = await supervisor_ref.get_model(model_uid)
    await model_ref.record_metrics(
        "input_tokens_total_counter", "inc", {"labels": {"model": model_uid}}
    )
    response = requests.get(metrics_exporter_address)
    assert response.ok
    assert 'xinference:input_tokens_total_counter{model="orca"} 1' in response.text


async def test_metrics_exporter_data(setup_cluster):
    endpoint, metrics_exporter_address, supervisor_address = setup_cluster

    from ...client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="orca",
        model_size_in_billions=3,
        model_format="ggmlv3",
        quantization="q4_0",
    )

    model = client.get_model(model_uid)
    response = model.chat("write a poem.")

    response = requests.get(metrics_exporter_address)
    assert response.ok
    assert 'format="ggmlv3",model="orca"' in response.text
