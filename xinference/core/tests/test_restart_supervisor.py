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

import asyncio
import pytest
import xoscar as xo
from typing import List, Optional, Union, Dict
import multiprocessing

from ...core.supervisor import SupervisorActor



# test restart supervisor
@pytest.mark.asyncio
async def test_restart_supervisor():
    from ...deploy.supervisor import run_in_subprocess as supervisor_run_in_subprocess
    from ...deploy.worker import main as _start_worker

    def worker_run_in_subprocess(
        address: str, 
        supervisor_address: str,
        logging_conf: Optional[Dict] = None
    ) -> multiprocessing.Process:
        p = multiprocessing.Process(target=_start_worker, args=(address, supervisor_address, None, None, logging_conf))
        p.start()
        return p

    # start supervisor
    supervisor_address = f"localhost:{xo.utils.get_next_port()}"
    proc_supervisor = supervisor_run_in_subprocess(supervisor_address)

    await asyncio.sleep(5)

    # start worker
    worker_run_in_subprocess(
        address=f"localhost:{xo.utils.get_next_port()}", 
        supervisor_address=supervisor_address
    )
    
    await asyncio.sleep(10)

    # load model
    supervisor_ref = await xo.actor_ref(
        supervisor_address, SupervisorActor.default_uid()
    )

    model_uid = "qwen1.5-chat"
    await supervisor_ref.launch_builtin_model(
        model_uid=model_uid,
        model_name="qwen1.5-chat",
        model_size_in_billions="0_5",
        quantization="q4_0",
        model_engine="vLLM"
    )

    # query replica info
    model_replica_info = await supervisor_ref.describe_model(model_uid)

    # kill supervisor
    proc_supervisor.terminate()
    proc_supervisor.join()

    # restart supervisor
    proc_supervisor = supervisor_run_in_subprocess(supervisor_address)

    await asyncio.sleep(5)

    supervisor_ref = await xo.actor_ref(
        supervisor_address, SupervisorActor.default_uid()
    )

    # check replica info
    model_replic_info_check = await supervisor_ref.describe_model(model_uid)

    assert model_replica_info["replica"] == model_replic_info_check["replica"]
