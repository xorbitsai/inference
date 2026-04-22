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

import pytest

from ..utils import handle_click_args_type


class TestHandleClickArgsType:
    """Tests for handle_click_args_type, including JSON parameter parsing.

    Covers the fix for https://github.com/xorbitsai/inference/issues/4760
    where JSON object parameters (e.g. compilation_config, kv_transfer_config)
    passed via the CLI were incorrectly treated as plain strings.
    """

    # --- Original type conversions (backwards compatibility) ---

    def test_none_string(self):
        assert handle_click_args_type("None") is None

    def test_true_string(self):
        assert handle_click_args_type("True") is True
        assert handle_click_args_type("true") is True

    def test_false_string(self):
        assert handle_click_args_type("False") is False
        assert handle_click_args_type("false") is False

    def test_integer(self):
        assert handle_click_args_type("42") == 42
        assert isinstance(handle_click_args_type("42"), int)

    def test_negative_integer(self):
        assert handle_click_args_type("-7") == -7
        assert isinstance(handle_click_args_type("-7"), int)

    def test_float(self):
        result = handle_click_args_type("3.14")
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_negative_float(self):
        result = handle_click_args_type("-0.5")
        assert result == pytest.approx(-0.5)
        assert isinstance(result, float)

    def test_plain_string(self):
        assert handle_click_args_type("hello") == "hello"
        assert isinstance(handle_click_args_type("hello"), str)

    def test_plain_string_with_spaces(self):
        assert handle_click_args_type("hello world") == "hello world"

    def test_empty_string(self):
        assert handle_click_args_type("") == ""

    # --- JSON object parsing (issue #4760) ---

    def test_json_object_simple(self):
        """Basic JSON object — the core bug scenario."""
        result = handle_click_args_type('{"key": "value"}')
        assert result == {"key": "value"}
        assert isinstance(result, dict)

    def test_json_object_compilation_config(self):
        """Real-world example: vLLM compilation_config."""
        arg = '{"mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}'
        result = handle_click_args_type(arg)
        assert isinstance(result, dict)
        assert result["mode"] == 3
        assert result["cudagraph_capture_sizes"] == [1, 2, 4, 8]

    def test_json_object_kv_transfer_config(self):
        """Real-world example: vLLM kv_transfer_config."""
        arg = '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'
        result = handle_click_args_type(arg)
        assert isinstance(result, dict)
        assert result["kv_connector"] == "FlexKVConnectorV1"
        assert result["kv_role"] == "kv_both"

    def test_json_object_speculative_config(self):
        """Real-world example: vLLM speculative_config."""
        arg = '{"method": "mtp", "num_speculative_tokens": 1}'
        result = handle_click_args_type(arg)
        assert isinstance(result, dict)
        assert result["method"] == "mtp"
        assert result["num_speculative_tokens"] == 1

    def test_json_nested_object(self):
        """Nested JSON objects should be parsed correctly."""
        arg = '{"outer": {"inner_key": "inner_value"}, "flag": true}'
        result = handle_click_args_type(arg)
        assert isinstance(result, dict)
        assert result["outer"] == {"inner_key": "inner_value"}
        assert result["flag"] is True

    def test_json_object_with_numeric_values(self):
        """JSON objects containing integers, floats, and booleans."""
        arg = '{"int_val": 10, "float_val": 1.5, "bool_val": false, "null_val": null}'
        result = handle_click_args_type(arg)
        assert isinstance(result, dict)
        assert result["int_val"] == 10
        assert result["float_val"] == 1.5
        assert result["bool_val"] is False
        assert result["null_val"] is None

    # --- JSON array parsing ---

    def test_json_array_of_integers(self):
        result = handle_click_args_type("[1, 2, 3]")
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_json_array_of_strings(self):
        result = handle_click_args_type('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_json_array_of_objects(self):
        arg = '[{"name": "lora1"}, {"name": "lora2"}]'
        result = handle_click_args_type(arg)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "lora1"

    def test_json_empty_array(self):
        result = handle_click_args_type("[]")
        assert result == []

    def test_json_empty_object(self):
        result = handle_click_args_type("{}")
        assert result == {}

    # --- Malformed JSON (should fall back to string) ---

    def test_malformed_json_object(self):
        """Invalid JSON starting with '{' should fall back to string."""
        arg = "{not valid json}"
        result = handle_click_args_type(arg)
        assert result == arg
        assert isinstance(result, str)

    def test_malformed_json_array(self):
        """Invalid JSON starting with '[' should fall back to string."""
        arg = "[not, valid, json"
        result = handle_click_args_type(arg)
        assert result == arg
        assert isinstance(result, str)

    def test_malformed_json_trailing_comma(self):
        """Trailing comma is invalid JSON — should fall back to string."""
        arg = '{"key": "value",}'
        result = handle_click_args_type(arg)
        assert result == arg
        assert isinstance(result, str)

    def test_malformed_json_single_quotes(self):
        """Single quotes are not valid JSON — should fall back to string."""
        arg = "{'key': 'value'}"
        result = handle_click_args_type(arg)
        assert result == arg
        assert isinstance(result, str)

    # --- Edge cases: ensure priority order is correct ---

    def test_integer_not_parsed_as_json(self):
        """'42' is valid JSON too, but should be parsed as int (higher priority)."""
        result = handle_click_args_type("42")
        assert isinstance(result, int)
        assert result == 42

    def test_float_not_parsed_as_json(self):
        """'3.14' is valid JSON too, but should be parsed as float (higher priority)."""
        result = handle_click_args_type("3.14")
        assert isinstance(result, float)

    def test_string_starting_with_brace_but_not_json(self):
        """{abc is not JSON and does not start with valid JSON syntax."""
        result = handle_click_args_type("{abc")
        assert result == "{abc"
        assert isinstance(result, str)

    def test_string_starting_with_bracket_but_not_json(self):
        result = handle_click_args_type("[abc")
        assert result == "[abc"
        assert isinstance(result, str)

    # --- Additional edge cases ---

    def test_json_with_whitespace(self):
        """JSON with internal whitespace should parse correctly."""
        result = handle_click_args_type('{ "key": "value", "number": 42 }')
        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["number"] == 42

    def test_json_with_unicode(self):
        """JSON containing Unicode characters should parse correctly."""
        result = handle_click_args_type('{"name": "测试", "emoji": "🚀"}')
        assert isinstance(result, dict)
        assert result["name"] == "测试"
        assert result["emoji"] == "🚀"

    def test_json_array_with_mixed_types(self):
        """JSON arrays with mixed element types should parse correctly."""
        result = handle_click_args_type('[1, "two", null, true, {"key": "value"}]')
        assert isinstance(result, list)
        assert len(result) == 5
        assert result[0] == 1
        assert result[1] == "two"
        assert result[2] is None
        assert result[3] is True
        assert result[4] == {"key": "value"}
