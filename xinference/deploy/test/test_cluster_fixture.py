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
from unittest.mock import Mock

import pytest

from ... import conftest as fixtures
from ...api import restful_api
from .. import utils as deploy_utils


@pytest.mark.parametrize(
    "fixture_name", ["setup", "setup_with_file_logging", "setup_real_actor_pool"]
)
@pytest.mark.parametrize("failure", ["cluster", "api", "api_start", None])
def test_cluster_fixture_reaps_children_on_setup_failure(
    monkeypatch, fixture_name, failure
):
    cluster = Mock()
    api = Mock()
    monkeypatch.setattr(fixtures.logging.config, "dictConfig", lambda config: None)
    start_cluster = Mock(return_value=cluster)
    monkeypatch.setattr(fixtures, "run_test_cluster_in_subprocess", start_cluster)
    monkeypatch.setattr(
        deploy_utils, "health_check", lambda *args, **kwargs: failure != "cluster"
    )
    monkeypatch.setattr(
        fixtures, "api_health_check", lambda *args, **kwargs: failure != "api"
    )
    start_api = Mock(return_value=api)
    if failure == "api_start":
        start_api.side_effect = RuntimeError("API spawn failed")
    monkeypatch.setattr(restful_api, "run_in_subprocess", start_api)
    monkeypatch.setattr(fixtures, "_get_test_port", lambda: 12345)
    monkeypatch.setenv("XINFERENCE_AUTH_ADVANCED", "true")

    generator = getattr(fixtures, fixture_name).__wrapped__()
    if failure:
        with pytest.raises(RuntimeError):
            next(generator)
    else:
        next(generator)
        generator.close()

    if fixture_name != "setup_with_file_logging":
        start_cluster.assert_called_once_with(
            "127.0.0.1:12345",
            fixtures.TEST_LOGGING_CONF,
            fixture_name == "setup",
        )
    cluster.kill.assert_called_once()
    cluster.join.assert_called_once_with(timeout=10)
    if failure not in ("cluster", "api_start"):
        api.kill.assert_called_once()
        api.join.assert_called_once_with(timeout=10)
    else:
        api.kill.assert_not_called()


def test_stop_test_process_reaps_an_already_exited_child():
    process = Mock()
    process.is_alive.return_value = False
    fixtures._stop_test_process(process)
    process.kill.assert_not_called()
    process.join.assert_called_once_with(timeout=10)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_support,reason",
    [
        ("config", "FPQuantConfig is not available"),
        ("cuda", "FPQuant requires a CUDA GPU"),
        ("fp_quant", "fp_quant"),
    ],
)
async def test_unsupported_fp4_skips_before_starting_cluster(
    monkeypatch, missing_support, reason
):
    import sys
    from types import ModuleType

    import torch

    from ...model.llm.transformers.tests.test_opt import test_opt_fp4_model

    transformers = ModuleType("transformers")
    if missing_support != "config":
        transformers.FPQuantConfig = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: missing_support == "fp_quant"
    )
    monkeypatch.setitem(sys.modules, "fp_quant", None)
    request = Mock()
    with pytest.raises(pytest.skip.Exception, match=reason):
        await test_opt_fp4_model(request)
    request.getfixturevalue.assert_not_called()
