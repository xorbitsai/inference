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

import asyncio
from unittest.mock import MagicMock

import pytest

from xinference.core.supervisor import SupervisorActor


class _DummySupervisor:
    _load_autostart_entries = SupervisorActor._load_autostart_entries

    def __init__(self, entries):
        self._autostart_store_lock = asyncio.Lock()
        self._launch_history_store = MagicMock()
        self._launch_history_store.list_autostart.return_value = entries

    async def _run_in_executor(self, func, *args):
        return func(*args)


@pytest.mark.asyncio
async def test_load_autostart_entries_reads_sqlite_store_and_normalizes():
    supervisor = _DummySupervisor(
        [
            {
                "priority": "5",
                "max_retries": "2",
                "retry_interval_seconds": "9",
                "launch": {"model_name": "llama", "model_uid": "uid-1"},
            }
        ]
    )

    entries = await supervisor._load_autostart_entries()

    assert entries == [
        {
            "enabled": True,
            "priority": 5,
            "max_retries": 2,
            "retry_interval_seconds": 9,
            "launch": {
                "model_name": "llama",
                "model_uid": "uid-1",
                "model_type": "LLM",
            },
        }
    ]
    supervisor._launch_history_store.list_autostart.assert_called_once_with(None)
