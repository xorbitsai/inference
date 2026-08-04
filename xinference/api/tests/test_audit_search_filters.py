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

import json
from typing import Any, Dict, List

import pytest

from ..routers.admin import _search_audit_from_file, search_audit_logs

ENTRIES: List[Dict[str, Any]] = [
    {
        "@timestamp": "2026-08-03T16:24:16.000Z",
        "user": "admin",
        "api_key_name": "robot",
        "model_id": "SenseVoiceSmall",
        "model_name": "SenseVoiceSmall",
        "model_type": "audio",
        "endpoint": "/v1/audio/transcriptions",
        "status": "success",
        "category": "inference",
        "auth_type": "api_key",
        "client_ip": "117.61.88.85",
    },
    {
        "@timestamp": "2026-08-03T16:24:15.000Z",
        "user": "alice",
        "api_key_name": "laptop",
        "model_id": "qwen2.5-instruct-abc123",
        "model_name": "qwen2.5-instruct",
        "model_type": "llm",
        "endpoint": "/v1/chat/completions",
        "status": "error",
        "category": "inference",
        "auth_type": "bearer",
        "client_ip": "10.0.0.7",
    },
]


async def _search(audit_path, **kwargs) -> Dict[str, Any]:
    params: Dict[str, Any] = dict(
        time_from="",
        time_to="",
        user="",
        api_key_name="",
        model_id="",
        model_name="",
        model_type="",
        category="",
        auth_type="",
        status="",
        client_ip="",
        page_from=0,
        size=50,
    )
    params.update(kwargs)
    resp = await _search_audit_from_file(**params)
    return json.loads(resp.body)


@pytest.fixture
def audit_log(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    path = log_dir / "audit.log"
    path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in ENTRIES) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("xinference.constants.XINFERENCE_LOG_DIR", str(log_dir))
    monkeypatch.delenv("XINFERENCE_ES_URL", raising=False)
    return path


@pytest.mark.asyncio
async def test_no_filter_returns_all(audit_log):
    data = await _search(audit_log)
    assert data["total"] == 2


@pytest.mark.parametrize(
    "field, needle",
    [
        # The exact case reported in the issue: a single-character prefix.
        ("model_id", "S"),
        ("model_name", "S"),
        ("model_id", "SenseVoice"),
        ("model_name", "sensevoicesmall"),  # case-insensitive
        ("model_id", "voice"),  # mid-string substring
        ("user", "adm"),
        ("api_key_name", "rob"),
        ("client_ip", "117.61"),  # IP prefix
    ],
)
@pytest.mark.asyncio
async def test_partial_text_filters_match(audit_log, field, needle):
    """A partial value must surface the matching record instead of nothing.

    Other records may legitimately also match (e.g. a case-insensitive "S"
    also hits the "s" in "qwen2.5-instruct"), so assert on inclusion.
    """
    data = await _search(audit_log, **{field: needle})
    assert data["total"] >= 1, f"{field}={needle!r} should match SenseVoiceSmall"
    assert "SenseVoiceSmall" in [h["model_id"] for h in data["hits"]]


@pytest.mark.asyncio
async def test_exact_value_still_matches(audit_log):
    data = await _search(audit_log, model_id="SenseVoiceSmall")
    assert data["total"] == 1


@pytest.mark.asyncio
async def test_non_matching_text_filter_returns_nothing(audit_log):
    data = await _search(audit_log, model_id="does-not-exist")
    assert data["total"] == 0
    assert data["hits"] == []


@pytest.mark.asyncio
async def test_text_filters_are_combined_with_and(audit_log):
    data = await _search(audit_log, user="admin", model_id="qwen")
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_enum_filters_remain_exact(audit_log):
    # `llm` must not match via substring against some longer stored value,
    # and the dropdown value is matched case-insensitively.
    assert (await _search(audit_log, model_type="llm"))["total"] == 1
    assert (await _search(audit_log, model_type="LLM"))["total"] == 1
    assert (await _search(audit_log, model_type="l"))["total"] == 0
    assert (await _search(audit_log, status="success"))["total"] == 1
    assert (await _search(audit_log, model_type="llm,audio"))["total"] == 2


@pytest.mark.asyncio
async def test_malformed_entry_does_not_break_search(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    lines = [
        json.dumps(ENTRIES[0]),
        "not json at all",
        # non-string values previously raised AttributeError -> HTTP 500
        json.dumps({"@timestamp": "2026-08-03T16:00:00.000Z", "status": 500}),
        json.dumps({"@timestamp": "2026-08-03T16:00:00.000Z", "model_id": None}),
    ]
    (log_dir / "audit.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr("xinference.constants.XINFERENCE_LOG_DIR", str(log_dir))

    data = await _search(log_dir / "audit.log", model_id="Sense")
    assert data["total"] == 1
    data = await _search(log_dir / "audit.log", status="success")
    assert data["total"] == 1


@pytest.mark.asyncio
async def test_missing_log_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("xinference.constants.XINFERENCE_LOG_DIR", str(tmp_path))
    data = await _search(tmp_path / "audit.log")
    assert data == {"hits": [], "total": 0}


@pytest.mark.asyncio
async def test_es_mode_uses_case_insensitive_wildcard(monkeypatch):
    """ES mode must use substring semantics on the `.keyword` subfield.

    A `term` query against the analyzed `text` field never matches a value
    containing uppercase or `-`/`_`/`/`.
    """
    captured: Dict[str, Any] = {}

    class _FakeResponse:
        status = 200

        async def json(self):
            return {"hits": {"hits": [], "total": {"value": 0}}}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        def post(self, url, json=None, headers=None):
            captured["body"] = json
            return _FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setenv("XINFERENCE_ES_URL", "http://localhost:9200")
    monkeypatch.setattr("aiohttp.ClientSession", _FakeSession)

    await search_audit_logs(model_id="Sense*Voice")

    clauses = captured["body"]["query"]["bool"]["filter"]
    wildcards = [c["wildcard"] for c in clauses if "wildcard" in c]
    assert len(wildcards) == 1
    clause = wildcards[0]["model_id.keyword"]
    assert clause["case_insensitive"] is True
    # user-typed `*` is escaped rather than treated as a wildcard
    assert clause["value"] == r"*Sense\*Voice*"
