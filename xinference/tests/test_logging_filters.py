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

import logging

import pytest

from xinference.deploy import utils as deploy_utils
from xinference.deploy.utils import PollingAccessFilter


def _access_record(method: str, path: str, status: int) -> logging.LogRecord:
    """Build the record uvicorn emits for one request."""
    return logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "",
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("10.0.0.1:1234", method, path, "1.1", status),
        None,
    )


@pytest.mark.parametrize(
    "method,path,status",
    [
        ("GET", "/v1/models/my-model/progress", 200),
        ("GET", "/v1/models/my-model/replicas", 200),
        ("GET", "/v1/models/my-model/progress?request_id=abc", 200),
        ("GET", "/metrics", 200),
        ("GET", "/status", 200),
    ],
)
def test_polling_is_dropped(method, path, status):
    assert PollingAccessFilter().filter(_access_record(method, path, status)) is False


@pytest.mark.parametrize(
    "method,path,status",
    [
        ("GET", "/v1/models", 200),  # real query, not polling
        ("POST", "/v1/models", 200),  # launching a model
        ("DELETE", "/v1/models/my-model", 200),
        ("GET", "/v1/models/my-model/progress", 500),  # failures stay visible
        ("GET", "/v1/models/my-model/progress", 404),
    ],
)
def test_everything_else_is_kept(method, path, status):
    assert PollingAccessFilter().filter(_access_record(method, path, status)) is True


def test_opt_in_keeps_polling(monkeypatch):
    monkeypatch.setattr(deploy_utils, "XINFERENCE_LOG_POLLING_ACCESS", True)
    assert PollingAccessFilter().filter(_access_record("GET", "/metrics", 200)) is True


def test_unexpected_record_shape_is_kept():
    """Records that are not uvicorn access lines must pass through untouched."""
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0, "something else %s", ("x",), None
    )
    assert PollingAccessFilter().filter(record) is True
