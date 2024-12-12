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

import multiprocessing
import time
from typing import Dict, Optional

import xoscar as xo

from ...api import restful_api
from ...client import Client


def test_restart_supervisor():
    from ...deploy.supervisor import run_in_subprocess as supervisor_run_in_subprocess
    from ...deploy.worker import main as _start_worker

    def worker_run_in_subprocess(
        address: str, supervisor_address: str, logging_conf: Optional[Dict] = None
    ) -> multiprocessing.Process:
        p = multiprocessing.Process(
            target=_start_worker,
            args=(address, supervisor_address, None, None, logging_conf),
        )
        p.start()
        return p

    # start supervisor
    web_port, supervisor_port = xo.utils.get_next_port(), xo.utils.get_next_port()
    supervisor_address = f"127.0.0.1:{supervisor_port}"
    proc_supervisor = supervisor_run_in_subprocess(supervisor_address)
    rest_api_proc = multiprocessing.Process(
        target=restful_api.run,
        kwargs=dict(
            supervisor_address=supervisor_address, host="127.0.0.1", port=web_port
        ),
    )
    rest_api_proc.start()

    time.sleep(5)

    # start worker
    proc_worker = worker_run_in_subprocess(
        address=f"127.0.0.1:{xo.utils.get_next_port()}",
        supervisor_address=supervisor_address,
    )

    time.sleep(10)

    client = Client(f"http://127.0.0.1:{web_port}")

    try:
        model_uid = "qwen1.5-chat"
        client.launch_model(
            model_uid=model_uid,
            model_name="qwen1.5-chat",
            model_size_in_billions="0_5",
            quantization="q4_0",
            model_engine="llama.cpp",
        )

        # query replica info
        model_replica_info = client.describe_model(model_uid)
        assert model_replica_info is not None

        # kill supervisor
        proc_supervisor.terminate()
        proc_supervisor.join()

        # restart supervisor
        supervisor_run_in_subprocess(supervisor_address)

        time.sleep(5)

        # check replica info
        model_replic_info_check = client.describe_model(model_uid)
        assert model_replica_info["replica"] == model_replic_info_check["replica"]

    finally:
        client.abort_cluster()
        proc_supervisor.terminate()
        proc_worker.terminate()
        proc_supervisor.join()
        proc_worker.join()
