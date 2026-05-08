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

"""
Tests for Jina embeddings v3/v4 task parameter mapping.

Verifies that the Jina-style ``task`` parameter is correctly resolved:
- v3: maps to SentenceTransformer ``prompt_name`` (dot-notation keys)
- v4: passes task through to ``model.forward()``
"""

import pytest

from ..core import JINA_V3_TASK_TO_PROMPT_NAME, JINA_V4_VALID_TASKS, _resolve_jina_task

# ---------------------------------------------------------------------------
# _resolve_jina_task unit tests
# ---------------------------------------------------------------------------


class TestResolveJinaTaskV4:
    """Tests for v4: returns (None, task) — task passthrough to forward()."""

    def test_retrieval_passage(self):
        prompt_name, task = _resolve_jina_task(
            "jina-embeddings-v4", "retrieval.passage"
        )
        assert prompt_name is None
        assert task == "retrieval.passage"

    def test_retrieval_query(self):
        prompt_name, task = _resolve_jina_task("jina-embeddings-v4", "retrieval.query")
        assert prompt_name is None
        assert task == "retrieval.query"

    def test_text_matching(self):
        prompt_name, task = _resolve_jina_task("jina-embeddings-v4", "text-matching")
        assert prompt_name is None
        assert task == "text-matching"

    def test_code(self):
        prompt_name, task = _resolve_jina_task("jina-embeddings-v4", "code")
        assert prompt_name is None
        assert task == "code"

    def test_retrieval_passthrough(self):
        """Plain 'retrieval' is passed through as-is for v4."""
        prompt_name, task = _resolve_jina_task("jina-embeddings-v4", "retrieval")
        assert prompt_name is None
        assert task == "retrieval"

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Invalid task"):
            _resolve_jina_task("jina-embeddings-v4", "nonexistent-task")


class TestResolveJinaTaskV3:
    """Tests for v3: returns (prompt_name, None) — uses prompt_name mechanism."""

    def test_retrieval_passage(self):
        prompt_name, task = _resolve_jina_task(
            "jina-embeddings-v3", "retrieval.passage"
        )
        assert prompt_name == "retrieval.passage"
        assert task is None

    def test_retrieval_query(self):
        prompt_name, task = _resolve_jina_task("jina-embeddings-v3", "retrieval.query")
        assert prompt_name == "retrieval.query"
        assert task is None

    def test_retrieval_backward_compat(self):
        """Plain 'retrieval' maps to 'retrieval.passage' for v3."""
        prompt_name, task = _resolve_jina_task("jina-embeddings-v3", "retrieval")
        assert prompt_name == "retrieval.passage"
        assert task is None

    def test_passage_alias(self):
        prompt_name, task = _resolve_jina_task("jina-embeddings-v3", "passage")
        assert prompt_name == "retrieval.passage"
        assert task is None

    def test_query_alias(self):
        prompt_name, task = _resolve_jina_task("jina-embeddings-v3", "query")
        assert prompt_name == "retrieval.query"
        assert task is None

    def test_document(self):
        prompt_name, task = _resolve_jina_task("jina-embeddings-v3", "document")
        assert prompt_name == "document"
        assert task is None

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Invalid task"):
            _resolve_jina_task("jina-embeddings-v3", "nonexistent-task")


class TestResolveJinaTaskGeneral:
    """Tests for general behavior: None task, non-Jina models, case sensitivity."""

    def test_none_task_returns_none_none(self):
        prompt_name, task = _resolve_jina_task("jina-embeddings-v4", None)
        assert prompt_name is None
        assert task is None

    def test_non_jina_model_returns_none_none(self):
        prompt_name, task = _resolve_jina_task("bge-small-en-v1.5", "retrieval.passage")
        assert prompt_name is None
        assert task is None

    def test_non_jina_model_with_none_task(self):
        prompt_name, task = _resolve_jina_task("bge-small-en-v1.5", None)
        assert prompt_name is None
        assert task is None

    def test_case_insensitive_model_name_v4(self):
        prompt_name, task = _resolve_jina_task("Jina-Embeddings-V4", "retrieval.query")
        assert prompt_name is None
        assert task == "retrieval.query"

    def test_case_insensitive_model_name_v3(self):
        prompt_name, task = _resolve_jina_task("Jina-Embeddings-V3", "retrieval.query")
        assert prompt_name == "retrieval.query"
        assert task is None


# ---------------------------------------------------------------------------
# Mapping table completeness
# ---------------------------------------------------------------------------


class TestJinaMappingTables:
    """Verify mapping tables are complete and correct."""

    def test_v3_table_has_expected_keys(self):
        expected = {
            "retrieval.passage",
            "retrieval.query",
            "retrieval",
            "passage",
            "query",
            "document",
        }
        assert expected == set(JINA_V3_TASK_TO_PROMPT_NAME.keys())

    def test_v4_valid_tasks_has_expected_values(self):
        expected = {
            "retrieval.passage",
            "retrieval.query",
            "retrieval",
            "text-matching",
            "code",
            "passage",
            "query",
            "document",
        }
        assert expected == JINA_V4_VALID_TASKS

    def test_v3_retrieval_alias_maps_to_retrieval_passage(self):
        assert JINA_V3_TASK_TO_PROMPT_NAME["retrieval"] == "retrieval.passage"

    def test_v3_dotted_tasks_map_to_themselves(self):
        for task in ("retrieval.passage", "retrieval.query"):
            assert JINA_V3_TASK_TO_PROMPT_NAME[task] == task
