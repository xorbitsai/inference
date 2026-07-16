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

"""Tests for the ``truncate_prompt_tokens`` field on CreateEmbeddingRequest.

Validates the schema contract: the field exists, defaults to None, and
accepts valid integer values (positive, negative, and None).
"""

from ..schemas.requests import CreateEmbeddingRequest


class TestCreateEmbeddingRequestTruncateField:
    """Schema-level validation for the truncate_prompt_tokens field."""

    def test_default_is_none(self):
        """Field must default to None so existing clients are unaffected."""
        req = CreateEmbeddingRequest(model="m", input="hello")
        assert req.truncate_prompt_tokens is None

    def test_accepts_positive_int(self):
        req = CreateEmbeddingRequest(
            model="m", input="hello", truncate_prompt_tokens=512
        )
        assert req.truncate_prompt_tokens == 512

    def test_accepts_negative_int(self):
        req = CreateEmbeddingRequest(
            model="m", input="hello", truncate_prompt_tokens=-1
        )
        assert req.truncate_prompt_tokens == -1

    def test_accepts_none_explicit(self):
        req = CreateEmbeddingRequest(
            model="m", input="hello", truncate_prompt_tokens=None
        )
        assert req.truncate_prompt_tokens is None

    def test_accepts_zero(self):
        """Zero is a valid (though unusual) token limit."""
        req = CreateEmbeddingRequest(model="m", input="hello", truncate_prompt_tokens=0)
        assert req.truncate_prompt_tokens == 0

    def test_accepts_str_int_coerced(self):
        """pydantic coerces the string ``"512"`` to int for Optional[int]
        fields — this is normal pydantic lenient-coercion behaviour and must
        not raise."""
        req = CreateEmbeddingRequest(
            model="m", input="hello", truncate_prompt_tokens="512"
        )
        assert req.truncate_prompt_tokens == 512

    def test_passes_through_to_kwargs(self):
        """The field must be present in the request body when serialised, so
        the ``create_embedding`` handler can pop it from kwargs."""
        import json

        req = CreateEmbeddingRequest(
            model="m", input="hello", truncate_prompt_tokens=128
        )
        body = json.loads(req.json())
        assert body.get("truncate_prompt_tokens") == 128
