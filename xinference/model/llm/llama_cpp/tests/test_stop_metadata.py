# CI: linux

from xinference.model.llm.llama_cpp.core import _annotate_stop_sequence


def test_annotate_stop_sequence_preserves_backend_marker():
    response = {"choices": [{"finish_reason": "stop", "stop": "HALT"}]}
    _annotate_stop_sequence(response, ["HALT"])
    assert response["choices"][0]["stop_sequence"] == "HALT"


def test_annotate_stop_sequence_uses_matching_text_as_fallback():
    response = {"choices": [{"finish_reason": "stop", "text": "answer HALT"}]}
    _annotate_stop_sequence(response, ["HALT"])
    assert response["choices"][0]["stop_sequence"] == "HALT"
