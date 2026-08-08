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

"""Regression tests for Elasticsearch pagination beyond 10,000 hits."""

from typing import Any, Optional

import pytest

from xinference.api.routers import admin


class _FakeResponse:
    def __init__(self, data: dict[str, Any], status: int = 200):
        self._data = data
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def json(self) -> dict[str, Any]:
        return self._data

    async def text(self) -> str:
        return ""


class _FakeElasticsearchSession:
    def __init__(self, total: int):
        self._documents = [
            {"_source": {"sequence": value}, "sort": [value, value]}
            for value in range(total)
        ]
        self.search_bodies: list[dict[str, Any]] = []
        self.closed_pit_id: Optional[str] = None

    def post(self, url: str, *, json=None, params=None, headers=None):
        if url.endswith("/_pit"):
            assert params == {"keep_alive": admin._ES_PIT_KEEP_ALIVE}
            return _FakeResponse({"id": "pit-1"})

        assert url.endswith("/_search")
        self.search_bodies.append(json)
        if json["size"] == 0:
            return _FakeResponse(
                {
                    "pit_id": "pit-2",
                    "hits": {
                        "hits": [],
                        "total": {"value": len(self._documents), "relation": "eq"},
                    },
                }
            )

        order = json["sort"][0]["@timestamp"]["order"]
        documents = sorted(
            self._documents,
            key=lambda hit: tuple(hit["sort"]),
            reverse=order == "desc",
        )
        search_after = json.get("search_after")
        if search_after is not None:
            start = next(
                index
                for index, hit in enumerate(documents)
                if hit["sort"] == search_after
            )
            documents = documents[start + 1 :]
        return _FakeResponse(
            {
                "pit_id": "pit-2",
                "hits": {"hits": documents[: json["size"]]},
            }
        )

    def delete(self, url: str, *, json=None, headers=None):
        assert url.endswith("/_pit")
        self.closed_pit_id = json["id"]
        return _FakeResponse({"succeeded": True, "num_freed": 1})


@pytest.mark.asyncio
async def test_search_es_page_can_cross_default_result_window():
    session = _FakeElasticsearchSession(total=12050)

    hits, total = await admin._search_es_page(
        session,  # type: ignore[arg-type]
        es_url="http://elasticsearch:9200",
        es_index="xinference-logs-*",
        headers={"Content-Type": "application/json"},
        query={"match_all": {}},
        page_from=10000,
        size=50,
        source={"excludes": ["@version"]},
    )

    assert total == 12050
    assert [hit["sequence"] for hit in hits] == list(range(2049, 1999, -1))
    assert all("from" not in body for body in session.search_bodies)
    assert any("search_after" in body for body in session.search_bodies)
    assert all(
        body.get("_source") == {"excludes": ["@version"]}
        for body in session.search_bodies
    )
    assert session.search_bodies[0]["pit"]["id"] == "pit-1"
    assert all(body["pit"]["id"] == "pit-2" for body in session.search_bodies[1:])
    assert session.closed_pit_id == "pit-2"


@pytest.mark.asyncio
async def test_search_es_page_reads_partial_last_page_from_nearest_edge():
    session = _FakeElasticsearchSession(total=12025)

    hits, total = await admin._search_es_page(
        session,  # type: ignore[arg-type]
        es_url="http://elasticsearch:9200",
        es_index="xinference-audit-*",
        headers={"Content-Type": "application/json"},
        query={"match_all": {}},
        page_from=12000,
        size=50,
    )

    assert total == 12025
    assert [hit["sequence"] for hit in hits] == list(range(24, -1, -1))
    page_searches = [body for body in session.search_bodies if body["size"] > 0]
    assert len(page_searches) == 1
    assert page_searches[0]["sort"][0]["@timestamp"]["order"] == "asc"
    assert session.closed_pit_id == "pit-2"
