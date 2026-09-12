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


def _fail_if_get_event_loop_is_called():
    raise AssertionError("entry point must not rely on an implicit event loop")


def test_test_cluster_entrypoint_creates_event_loop(monkeypatch):
    from ... import conftest

    calls = []

    async def fake_start_test_cluster(address, logging_conf=None):
        calls.append((address, logging_conf))

    monkeypatch.setattr(conftest, "_start_test_cluster", fake_start_test_cluster)
    monkeypatch.setattr(conftest.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        conftest.asyncio, "get_event_loop", _fail_if_get_event_loop_is_called
    )

    conftest.run_test_cluster("localhost:1234", {"version": 1})

    assert calls == [("localhost:1234", {"version": 1})]


def test_supervisor_entrypoint_creates_event_loop(monkeypatch):
    from .. import supervisor

    calls = []

    async def fake_start_supervisor(address, logging_conf=None):
        calls.append((address, logging_conf))

    monkeypatch.setattr(supervisor, "_start_supervisor", fake_start_supervisor)
    monkeypatch.setattr(supervisor.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        supervisor.asyncio, "get_event_loop", _fail_if_get_event_loop_is_called
    )

    supervisor.run("localhost:1234", {"version": 1})

    assert calls == [("localhost:1234", {"version": 1})]


def test_worker_entrypoint_creates_event_loop(monkeypatch):
    from .. import worker

    calls = []

    async def fake_start_worker(*args):
        calls.append(args)

    monkeypatch.setattr(worker, "_start_worker", fake_start_worker)
    monkeypatch.setattr(worker.multiprocessing, "set_start_method", lambda *args: None)
    monkeypatch.setattr(
        worker.asyncio, "get_event_loop", _fail_if_get_event_loop_is_called
    )

    worker.main("worker", "supervisor", "endpoint", "host", 1234, {"version": 1})

    assert calls == [("worker", "supervisor", "endpoint", "host", 1234, {"version": 1})]


def test_local_subprocess_applies_environment_before_start(monkeypatch):
    from .. import local

    calls = []

    def fake_run(*args):
        calls.append((os.environ.get("XINFERENCE_DISABLE_METRICS"), args))

    monkeypatch.delenv("XINFERENCE_DISABLE_METRICS", raising=False)
    monkeypatch.setattr(local, "run", fake_run)

    local._run_with_environment(
        "worker",
        "host",
        1234,
        {"version": 1},
        None,
        {"XINFERENCE_DISABLE_METRICS": "1"},
    )

    assert calls == [("1", ("worker", "host", 1234, {"version": 1}, None))]


def test_restful_api_subprocess_applies_environment_before_start(monkeypatch):
    from ...api import restful_api

    calls = []

    def fake_run(*args):
        calls.append((os.environ.get("XINFERENCE_DISABLE_METRICS"), args))

    monkeypatch.delenv("XINFERENCE_DISABLE_METRICS", raising=False)
    monkeypatch.setattr(restful_api, "run", fake_run)

    restful_api._run_with_environment(
        "supervisor",
        "host",
        1234,
        {"version": 1},
        {"XINFERENCE_DISABLE_METRICS": "1"},
    )

    assert calls == [("1", ("supervisor", "host", 1234, {"version": 1}))]
