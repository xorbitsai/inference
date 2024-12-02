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
import xoscar as xo

from ...core.supervisor import SupervisorActor


# test restart supervisor
@pytest.mark.asyncio
async def test_restart_supervisor():
    from ...deploy.supervisor import run_in_subprocess as supervisor_run_in_subprocess
    from ...deploy.worker import main as worker_run_in_subprocess

    # start supervisor
    supervisor_address = "localhost:19034"
    proc_supervisor = supervisor_run_in_subprocess(supervisor_address)

    # start worker
    worker_run_in_subprocess(
        address="localhost:9998", supervisor_address=supervisor_address
    )

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
    )

    # query replica info
    bge_m3_info = await supervisor_ref.describe_model(model_uid)

    # kill supervisor
    proc_supervisor.kill()

    # restart supervisor
    proc_supervisor = supervisor_run_in_subprocess(supervisor_address)

    # check replica info
    bge_m3_info_check = await supervisor_ref.describe_model(model_uid)

    assert bge_m3_info["replica"] == bge_m3_info_check["replica"]
