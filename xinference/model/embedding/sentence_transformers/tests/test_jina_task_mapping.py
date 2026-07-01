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
Tests for Jina embeddings v3/v4/v5 task parameter mapping.

Verifies that the Jina-style ``task`` parameter is correctly resolved:
- v3: maps to SentenceTransformer ``prompt_name`` (dot-notation keys)
- v4: passes task through to ``model.forward()``
- v5: passes task through to ``model.encode()`` and additionally maps
      query/document aliases to v5 ``prompt_name``
"""

import pytest

from ..core import (
    JINA_V3_TASK_TO_PROMPT_NAME,
    JINA_V4_VALID_TASKS,
    JINA_V5_PROMPT_NAME_ALIASES,
    JINA_V5_VALID_TASKS,
    _resolve_jina_task,
)

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

    def test_v5_valid_tasks(self):
        assert JINA_V5_VALID_TASKS == {
            "retrieval",
            "text-matching",
            "classification",
            "clustering",
        }

    def test_v5_alias_keys(self):
        assert set(JINA_V5_PROMPT_NAME_ALIASES.keys()) == {
            "query",
            "document",
            "passage",
            "retrieval.query",
            "retrieval.passage",
        }


class TestResolveJinaTaskV5:
    """Tests for v5: returns (prompt_name_or_None, task)."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "jina-embeddings-v5-text-nano",
            "jina-embeddings-v5-text-small",
            "jina-embeddings-v5-omni-nano",
            "jina-embeddings-v5-omni-small",
        ],
    )
    def test_retrieval_passthrough(self, model_name):
        prompt_name, task = _resolve_jina_task(model_name, "retrieval")
        assert prompt_name is None
        assert task == "retrieval"

    @pytest.mark.parametrize(
        "task_name",
        ["text-matching", "classification", "clustering"],
    )
    def test_non_retrieval_tasks_passthrough(self, task_name):
        prompt_name, task = _resolve_jina_task(
            "jina-embeddings-v5-text-small", task_name
        )
        assert prompt_name is None
        assert task == task_name

    def test_query_alias_maps_to_retrieval(self):
        prompt_name, task = _resolve_jina_task("jina-embeddings-v5-text-small", "query")
        assert prompt_name == "query"
        assert task == "retrieval"

    def test_document_alias_maps_to_retrieval(self):
        prompt_name, task = _resolve_jina_task(
            "jina-embeddings-v5-text-small", "document"
        )
        assert prompt_name == "document"
        assert task == "retrieval"

    def test_passage_alias_maps_to_document(self):
        prompt_name, task = _resolve_jina_task(
            "jina-embeddings-v5-text-small", "passage"
        )
        assert prompt_name == "document"
        assert task == "retrieval"

    def test_retrieval_dot_query_alias(self):
        prompt_name, task = _resolve_jina_task(
            "jina-embeddings-v5-omni-small", "retrieval.query"
        )
        assert prompt_name == "query"
        assert task == "retrieval"

    def test_retrieval_dot_passage_alias(self):
        prompt_name, task = _resolve_jina_task(
            "jina-embeddings-v5-omni-small", "retrieval.passage"
        )
        assert prompt_name == "document"
        assert task == "retrieval"

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Invalid task"):
            _resolve_jina_task("jina-embeddings-v5-text-nano", "nonexistent")

    def test_none_task_defaults_to_retrieval(self):
        # v5 mandates a task; missing task should default to ``retrieval`` so
        # OpenAI-compatible clients (that have no notion of a task) still work.
        prompt_name, task = _resolve_jina_task("jina-embeddings-v5-omni-nano", None)
        assert prompt_name is None
        assert task == "retrieval"
