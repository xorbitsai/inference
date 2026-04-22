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
Tests for Jina embeddings v4 task parameter mapping.

Verifies that the Jina-style ``task`` parameter (e.g. ``retrieval.passage``,
``retrieval.query``) is correctly resolved to SentenceTransformer
``prompt_name`` values, ensuring API compatibility with the official Jina AI
API.
"""

import pytest

from ..core import (
    JINA_TASK_SUPPORTED_MODELS,
    JINA_V4_TASK_TO_PROMPT_NAME,
    _resolve_jina_task,
)

# ---------------------------------------------------------------------------
# _resolve_jina_task unit tests
# ---------------------------------------------------------------------------


class TestResolveJinaTask:
    """Tests for the _resolve_jina_task helper."""

    # -- Valid task values for jina-embeddings-v4 --

    def test_retrieval_passage(self):
        result = _resolve_jina_task("jina-embeddings-v4", "retrieval.passage")
        assert result == "retrieval.passage"

    def test_retrieval_query(self):
        result = _resolve_jina_task("jina-embeddings-v4", "retrieval.query")
        assert result == "retrieval.query"

    def test_text_matching(self):
        result = _resolve_jina_task("jina-embeddings-v4", "text-matching")
        assert result == "text-matching"

    def test_code(self):
        result = _resolve_jina_task("jina-embeddings-v4", "code")
        assert result == "code"

    # -- Backward compatibility: plain "retrieval" maps to "retrieval.passage" --

    def test_retrieval_backward_compat(self):
        result = _resolve_jina_task("jina-embeddings-v4", "retrieval")
        assert result == "retrieval.passage"

    # -- jina-embeddings-v3 is also supported --

    def test_jina_v3_retrieval_passage(self):
        result = _resolve_jina_task("jina-embeddings-v3", "retrieval.passage")
        assert result == "retrieval.passage"

    def test_jina_v3_retrieval_query(self):
        result = _resolve_jina_task("jina-embeddings-v3", "retrieval.query")
        assert result == "retrieval.query"

    def test_jina_v3_text_matching(self):
        result = _resolve_jina_task("jina-embeddings-v3", "text-matching")
        assert result == "text-matching"

    def test_jina_v3_code(self):
        result = _resolve_jina_task("jina-embeddings-v3", "code")
        assert result == "code"

    # -- None task returns None (no-op) --

    def test_none_task_returns_none(self):
        result = _resolve_jina_task("jina-embeddings-v4", None)
        assert result is None

    # -- Non-Jina model returns None regardless of task value --

    def test_non_jina_model_returns_none(self):
        result = _resolve_jina_task("bge-small-en-v1.5", "retrieval.passage")
        assert result is None

    def test_non_jina_model_with_none_task(self):
        result = _resolve_jina_task("bge-small-en-v1.5", None)
        assert result is None

    # -- Invalid task raises ValueError --

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Invalid task"):
            _resolve_jina_task("jina-embeddings-v4", "nonexistent-task")

    def test_invalid_task_message_contains_valid_options(self):
        with pytest.raises(ValueError, match="retrieval.passage"):
            _resolve_jina_task("jina-embeddings-v4", "bad")

    # -- Case sensitivity: model name matching is case-insensitive --

    def test_case_insensitive_model_name(self):
        result = _resolve_jina_task("Jina-Embeddings-V4", "retrieval.query")
        assert result == "retrieval.query"

    def test_uppercase_model_name(self):
        result = _resolve_jina_task("JINA-EMBEDDINGS-V4", "code")
        assert result == "code"


# ---------------------------------------------------------------------------
# Mapping table completeness
# ---------------------------------------------------------------------------


class TestJinaTaskMappingTable:
    """Verify the JINA_V4_TASK_TO_PROMPT_NAME mapping is complete."""

    EXPECTED_TASKS = {
        "retrieval.passage",
        "retrieval.query",
        "text-matching",
        "code",
        "retrieval",
    }

    def test_all_expected_tasks_present(self):
        assert self.EXPECTED_TASKS == set(JINA_V4_TASK_TO_PROMPT_NAME.keys())

    def test_supported_models_contains_v4(self):
        assert "jina-embeddings-v4" in JINA_TASK_SUPPORTED_MODELS

    def test_supported_models_contains_v3(self):
        assert "jina-embeddings-v3" in JINA_TASK_SUPPORTED_MODELS

    def test_retrieval_alias_maps_to_passage(self):
        """Plain 'retrieval' should default to 'retrieval.passage' for backward compat."""
        assert JINA_V4_TASK_TO_PROMPT_NAME["retrieval"] == "retrieval.passage"

    def test_each_dotted_task_maps_to_itself(self):
        """retrieval.passage -> retrieval.passage, retrieval.query -> retrieval.query, etc."""
        for task in ("retrieval.passage", "retrieval.query", "text-matching", "code"):
            assert JINA_V4_TASK_TO_PROMPT_NAME[task] == task
