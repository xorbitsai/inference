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
Unit tests for deepseek_ocr coordinate parsing.

The two parsing helpers in `xinference/model/image/ocr/deepseek_ocr.py`
used to call `eval()` directly on substrings extracted from OCR model
output. Because OCR output is LLM-generated (and thus can be steered by
crafted documents/prompts), `eval()` was a remote-code-execution path:
a model that emitted `<|det|>[[__import__('os').system('id')]]<|/det|>`
would have its payload executed by the host process.

Both call sites now use `ast.literal_eval`. These tests pin that
behaviour: legitimate coordinate strings still parse, while anything
that isn't a Python literal is rejected without side-effects.

This is the same fix shape as the merged tool-parser PR #4786 — that one
covered `xinference/model/llm/`; this one covers the image OCR path.
"""

import pytest

# The OCR module pulls in torch, PIL, and torchvision. Skip the whole
# module if any of those are missing in the test environment.
pytest.importorskip("torch")
pytest.importorskip("PIL")
pytest.importorskip("torchvision")

from xinference.model.image.ocr import deepseek_ocr  # noqa: E402


class TestExtractCoordinatesAndLabel:
    """`extract_coordinates_and_label` parses the third element of the
    re_match tuple (coordinate list literal) into a Python list."""

    def test_parses_valid_coordinate_list(self):
        ref = (
            "<|ref|>logo<|/ref|><|det|>[[10,20,30,40]]<|/det|>",
            "logo",
            "[[10,20,30,40]]",
        )
        result = deepseek_ocr.extract_coordinates_and_label(ref, 100, 100)
        assert result is not None
        label, coords = result
        assert label == "logo"
        assert coords == [[10, 20, 30, 40]]

    def test_parses_multiple_coordinate_lists(self):
        ref = ("raw", "title", "[[1, 2, 3, 4], [5, 6, 7, 8]]")
        result = deepseek_ocr.extract_coordinates_and_label(ref, 100, 100)
        assert result is not None
        label, coords = result
        assert label == "title"
        assert coords == [[1, 2, 3, 4], [5, 6, 7, 8]]

    def test_rejects_arbitrary_python_call(self, tmp_path):
        """The fix's reason for being: a payload that would have been
        executed by the old eval() must now be rejected with no side
        effect, and the function must return None instead of crashing."""
        sentinel = tmp_path / "pwned"
        payload = f"__import__('pathlib').Path({str(sentinel)!r}).write_text('x')"
        ref = ("raw", "label", payload)
        result = deepseek_ocr.extract_coordinates_and_label(ref, 100, 100)
        assert result is None
        assert not sentinel.exists(), (
            "ast.literal_eval must not execute __import__; "
            "if this file exists, the parser is unsafe again."
        )

    def test_rejects_attribute_access_payload(self):
        ref = ("raw", "label", "().__class__.__bases__[0].__subclasses__()")
        result = deepseek_ocr.extract_coordinates_and_label(ref, 100, 100)
        assert result is None

    def test_rejects_malformed_input_gracefully(self):
        ref = ("raw", "label", "[[1, 2,")  # truncated
        result = deepseek_ocr.extract_coordinates_and_label(ref, 100, 100)
        assert result is None


class TestExtractTextBlocks:
    """`extract_text_blocks` walks an annotated OCR string and collects
    `(label_type, coordinates, text)` blocks."""

    def test_parses_well_formed_block(self):
        text = "<|ref|>title<|/ref|><|det|>[[1, 2, 3, 4]]<|/det|>Hello world"
        blocks = deepseek_ocr.extract_text_blocks(text)
        assert len(blocks) == 1
        b = blocks[0]
        assert b["label_type"] == "title"
        assert b["coordinates"] == [[1, 2, 3, 4]]
        assert b["text"] == "Hello world"

    def test_parses_multiple_blocks(self):
        text = (
            "<|ref|>title<|/ref|><|det|>[[1, 2, 3, 4]]<|/det|>First "
            "<|ref|>body<|/ref|><|det|>[[5, 6, 7, 8]]<|/det|>Second"
        )
        blocks = deepseek_ocr.extract_text_blocks(text)
        assert [b["label_type"] for b in blocks] == ["title", "body"]
        assert blocks[0]["coordinates"] == [[1, 2, 3, 4]]
        assert blocks[1]["coordinates"] == [[5, 6, 7, 8]]

    def test_rejects_code_payload_in_coordinates(self, tmp_path):
        """An OCR model that emits a Python expression in place of a
        numeric coordinate must NOT have it executed. The block is
        skipped and parsing continues for any well-formed siblings."""
        sentinel = tmp_path / "pwned"
        payload = f"__import__('pathlib').Path({str(sentinel)!r}).write_text('x')"
        # The malicious block is followed by a valid one to verify the
        # parser does not abort on a bad block.
        text = (
            f"<|ref|>evil<|/ref|><|det|>[[{payload}]]<|/det|>bad "
            "<|ref|>safe<|/ref|><|det|>[[9, 9, 9, 9]]<|/det|>good"
        )
        blocks = deepseek_ocr.extract_text_blocks(text)

        assert not sentinel.exists(), (
            "ast.literal_eval must not execute __import__ inside the OCR "
            "coordinate parser; if this file exists, the parser is unsafe "
            "again."
        )
        # Only the safe block survives.
        assert len(blocks) == 1
        assert blocks[0]["label_type"] == "safe"
        assert blocks[0]["coordinates"] == [[9, 9, 9, 9]]

    def test_returns_empty_for_non_string_input(self):
        assert deepseek_ocr.extract_text_blocks(None) == []  # type: ignore[arg-type]
        assert deepseek_ocr.extract_text_blocks(123) == []  # type: ignore[arg-type]

    def test_returns_empty_when_no_annotations(self):
        assert deepseek_ocr.extract_text_blocks("plain text, no markers") == []
