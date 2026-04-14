import pytest

from ..llama3_tool_parser import Llama3ToolParser


class TestLlama3ToolParserSafeParsing:
    """Tests for Llama3 tool parser with safe eval replacement."""

    def setup_method(self):
        self.parser = Llama3ToolParser()

    # ── Normal parsing (JSON format) ──────────────────────────────────

    def test_parse_valid_json_tool_call(self):
        """Standard JSON tool call should parse correctly."""
        model_output = '{"name": "get_weather", "parameters": {"location": "Beijing"}}'
        result = self.parser.extract_tool_calls(model_output)
        assert result == [(None, "get_weather", {"location": "Beijing"})]

    def test_parse_json_with_nested_parameters(self):
        """JSON with nested dict parameters should parse correctly."""
        model_output = (
            '{"name": "search", "parameters": '
            '{"query": "test", "options": {"limit": 10, "offset": 0}}}'
        )
        result = self.parser.extract_tool_calls(model_output)
        assert result == [
            (None, "search", {"query": "test", "options": {"limit": 10, "offset": 0}})
        ]

    def test_parse_json_with_list_parameter(self):
        """JSON with list parameter should parse correctly."""
        model_output = (
            '{"name": "multi_search", "parameters": {"queries": ["a", "b", "c"]}}'
        )
        result = self.parser.extract_tool_calls(model_output)
        assert result == [(None, "multi_search", {"queries": ["a", "b", "c"]})]

    def test_parse_json_with_null_value(self):
        """JSON with null values should parse correctly."""
        model_output = '{"name": "test_func", "parameters": {"value": null}}'
        result = self.parser.extract_tool_calls(model_output)
        assert result == [(None, "test_func", {"value": None})]

    def test_parse_json_with_boolean_values(self):
        """JSON with boolean values should parse correctly."""
        model_output = (
            '{"name": "toggle", "parameters": {"enabled": true, "verbose": false}}'
        )
        result = self.parser.extract_tool_calls(model_output)
        assert result == [(None, "toggle", {"enabled": True, "verbose": False})]

    def test_parse_json_with_numeric_values(self):
        """JSON with integer and float parameters should parse correctly."""
        model_output = '{"name": "calculate", "parameters": {"x": 42, "y": 3.14}}'
        result = self.parser.extract_tool_calls(model_output)
        assert result == [(None, "calculate", {"x": 42, "y": 3.14})]

    def test_parse_json_empty_parameters(self):
        """JSON with empty parameters dict should parse correctly."""
        model_output = '{"name": "no_args_func", "parameters": {}}'
        result = self.parser.extract_tool_calls(model_output)
        assert result == [(None, "no_args_func", {})]

    # ── Normal parsing (Python literal format) ────────────────────────

    def test_parse_python_literal_with_true_false_none(self):
        """Python dict with True/False/None should parse via ast.literal_eval."""
        model_output = "{'name': 'check', 'parameters': {'flag': True, 'value': None}}"
        result = self.parser.extract_tool_calls(model_output)
        assert result == [(None, "check", {"flag": True, "value": None})]

    def test_parse_python_literal_with_single_quotes(self):
        """Python dict with single quotes should parse correctly."""
        model_output = "{'name': 'get_weather', 'parameters': {'location': 'Shanghai'}}"
        result = self.parser.extract_tool_calls(model_output)
        assert result == [(None, "get_weather", {"location": "Shanghai"})]

    # ── Security: RCE prevention ──────────────────────────────────────

    def test_reject_os_system_call(self):
        """Malicious os.system() call must not execute."""
        malicious = "__import__('os').system('echo PWNED')"
        result = self.parser.extract_tool_calls(malicious)
        assert result == [(malicious, None, None)]

    def test_reject_subprocess_call(self):
        """Malicious subprocess call must not execute."""
        malicious = "__import__('subprocess').check_output(['id'])"
        result = self.parser.extract_tool_calls(malicious)
        assert result == [(malicious, None, None)]

    def test_reject_builtins_exploit_via_class(self):
        """Class-based builtins exploit must not execute (CVE PoC from issue)."""
        malicious = "().__class__.__bases__[0].__subclasses__()"
        result = self.parser.extract_tool_calls(malicious)
        assert result == [(malicious, None, None)]

    def test_reject_eval_within_eval(self):
        """Nested eval must not execute."""
        malicious = 'eval(\'__import__("os").system("id")\')'
        result = self.parser.extract_tool_calls(malicious)
        assert result == [(malicious, None, None)]

    def test_reject_exec_call(self):
        """exec() call must not execute."""
        malicious = "exec('import os; os.system(\"whoami\")')"
        result = self.parser.extract_tool_calls(malicious)
        assert result == [(malicious, None, None)]

    def test_reject_lambda(self):
        """Lambda expressions must not execute."""
        malicious = "(lambda: __import__('os').system('id'))()"
        result = self.parser.extract_tool_calls(malicious)
        assert result == [(malicious, None, None)]

    def test_reject_compile_exec(self):
        """compile() + exec() chain must not execute."""
        malicious = "exec(compile('import os; os.system(\"id\")', '<string>', 'exec'))"
        result = self.parser.extract_tool_calls(malicious)
        assert result == [(malicious, None, None)]

    # ── Edge cases: malformed input ───────────────────────────────────

    def test_empty_string(self):
        """Empty string should return as content."""
        result = self.parser.extract_tool_calls("")
        assert result == [("", None, None)]

    def test_plain_text(self):
        """Plain text should return as content."""
        result = self.parser.extract_tool_calls("Hello, I cannot help with that.")
        assert result == [("Hello, I cannot help with that.", None, None)]

    def test_incomplete_json(self):
        """Truncated JSON should return as content."""
        result = self.parser.extract_tool_calls('{"name": "func", "parameters": {')
        assert result == [('{"name": "func", "parameters": {', None, None)]

    def test_missing_name_key(self):
        """Dict missing 'name' key should return as content."""
        model_output = '{"function": "test", "parameters": {}}'
        result = self.parser.extract_tool_calls(model_output)
        assert result == [(model_output, None, None)]

    def test_missing_parameters_key(self):
        """Dict missing 'parameters' key should return as content."""
        model_output = '{"name": "test", "args": {}}'
        result = self.parser.extract_tool_calls(model_output)
        assert result == [(model_output, None, None)]

    def test_non_dict_json(self):
        """JSON array (not dict) should return as content."""
        model_output = '[{"name": "test"}]'
        result = self.parser.extract_tool_calls(model_output)
        assert result == [(model_output, None, None)]

    def test_json_string_value(self):
        """A plain JSON string should return as content."""
        model_output = '"just a string"'
        result = self.parser.extract_tool_calls(model_output)
        assert result == [(model_output, None, None)]


class TestLlama3ToolParserStreaming:
    """Verify streaming raises NotImplementedError as expected."""

    def test_streaming_not_supported(self):
        parser = Llama3ToolParser()
        with pytest.raises(NotImplementedError):
            parser.extract_tool_calls_streaming(["prev"], "current", "delta")
