# Copyright 2022-2026 XProbe Inc.
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

from ..utils import (
    build_replica_model_uid,
    build_subpool_envs_for_virtual_env,
    iter_replica_model_uid,
    parse_replica_model_uid,
)


def test_replica_model_uid():
    all_gen_ids = []
    for replica_model_uid in iter_replica_model_uid("abc", 5):
        rebuild_replica_model_uid = build_replica_model_uid(
            *parse_replica_model_uid(replica_model_uid)
        )
        assert rebuild_replica_model_uid == replica_model_uid
        all_gen_ids.append(replica_model_uid)
    assert len(all_gen_ids) == 5
    assert len(set(all_gen_ids)) == 5


class DummyVirtualEnvManager:
    def __init__(self, python_path: str):
        self._python_path = python_path

    def get_python_path(self) -> str:
        return self._python_path


def test_build_subpool_envs_for_virtual_env_disabled():
    base_envs = {"PATH": "/usr/bin", "FLASHINFER_NINJA_PATH": "/custom/ninja"}
    result = build_subpool_envs_for_virtual_env(base_envs, False, None)

    assert result == base_envs
    assert result is not base_envs


def test_build_subpool_envs_for_virtual_env_enabled():
    manager = DummyVirtualEnvManager("/venv/bin/python")
    base_envs = {"PATH": "/usr/bin", "FLASHINFER_NINJA_PATH": "/custom/ninja"}

    result = build_subpool_envs_for_virtual_env(base_envs, True, manager)

    assert result["PATH"] == "/venv/bin" + ":" + "/usr/bin"
    assert result["VIRTUAL_ENV"] == "/venv"
    assert result["FLASHINFER_NINJA_PATH"] == "/custom/ninja"
    assert result is not base_envs
