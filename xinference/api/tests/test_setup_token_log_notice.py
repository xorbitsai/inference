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
"""Regression test for the first-run setup token startup log notice.

Gemini review on PR #5144: an operator-provided XINFERENCE_AUTH_SETUP_TOKEN
(e.g. sourced from a Kubernetes Secret to keep it out of logs) must never
be printed verbatim -- only an auto-generated token should be logged, since
otherwise any log reader could use it to win the first-admin race.

    pytest xinference/api/tests/test_setup_token_log_notice.py -v
"""

from xinference.api.restful_api import _log_setup_token_notice


def test_generated_token_is_logged(monkeypatch, caplog):
    monkeypatch.delenv("XINFERENCE_AUTH_SETUP_TOKEN", raising=False)
    monkeypatch.setattr(
        "xinference.api.restful_api.get_or_create_setup_token",
        lambda: "the-generated-token",
    )

    with caplog.at_level("WARNING"):
        _log_setup_token_notice()

    assert "the-generated-token" in caplog.text


def test_operator_provided_token_is_not_logged(monkeypatch, caplog):
    monkeypatch.setenv("XINFERENCE_AUTH_SETUP_TOKEN", "super-secret-from-k8s")

    with caplog.at_level("WARNING"):
        _log_setup_token_notice()

    assert "super-secret-from-k8s" not in caplog.text
    assert "XINFERENCE_AUTH_SETUP_TOKEN" in caplog.text
