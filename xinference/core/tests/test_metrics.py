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
        if not api_health_check(endpoint, max_attempts=3, sleep_interval=5):
            raise RuntimeError("Endpoint is not available after multiple attempts")

        yield f"http://localhost:{port}", f"http://localhost:{metrics_port}/metrics", supervisor_address
        restful_api_proc.terminate()
    finally:
        local_cluster.terminate()


@pytest.mark.asyncio
async def test_metrics_exporter(setup_cluster):
    endpoint, metrics_exporter_address, supervisor_address = setup_cluster

    import xoscar as xo
    from ..supervisor import SupervisorActor

    supervisor_ref = await xo.actor_ref(supervisor_address, SupervisorActor.uid())
    await supervisor_ref.record_metrics(
        "requests_throughput",
        "set",
        {"labels": {"node": supervisor_address}, "value": 12357},
    )
    response = requests.get(metrics_exporter_address)
    assert response.ok
    assert "12357" in response.text
    print(response)
