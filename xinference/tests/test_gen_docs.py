import ast
import os
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("preceding_hub", [None, "huggingface", "modelscope"])
@pytest.mark.parametrize("fallback", [None, ["int4"]])
def test_llm_documentation_quantizations_are_local_to_each_spec(
    tmp_path, monkeypatch, preceding_hub, fallback
):
    # Compile the real entry point without executing the module-level engine
    # mocks, which otherwise modify global runtime registries during collection.
    path = Path(__file__).parents[2] / "doc/source/gen_docs.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    spec = {
        "model_format": "pytorch",
        "model_size_in_billions": 7,
        "model_src": {"openmind_hub": {"model_id": "example/model"}},
    }
    if fallback is not None:
        spec["quantizations"] = fallback
    specs = []
    if preceding_hub is not None:
        specs.append(
            {
                "model_format": "pytorch",
                "model_size_in_billions": 3,
                "model_src": {
                    preceding_hub: {
                        "model_id": "example/previous",
                        "quantizations": ["previous-quantization"],
                    }
                },
            }
        )
    specs.append(spec)

    class LlmDocsComplete(Exception):
        pass

    def load_catalog(filename):
        if Path(filename).parent.name != "llm":
            raise LlmDocsComplete
        return [{"model_name": "example", "model_specs": specs}]

    template = Mock()
    template.render.return_value = "generated"
    environment = Mock()
    environment.get_template.return_value = template
    check_engine = Mock()
    namespace = {
        "os": os,
        "Environment": lambda **kwargs: environment,
        "FileSystemLoader": lambda path: path,
        "load_model_catalog": load_catalog,
        "SUPPORTED_ENGINES": ["test-engine"],
        "MODEL_HUB_HUGGING_FACE": "Hugging Face",
        "MODEL_HUB_MODELSCOPE": "ModelScope",
        "check_engine_by_spec_parameters": check_engine,
    }
    exec(
        compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec"), namespace
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(LlmDocsComplete):
        namespace["main"]()

    expected = []
    if preceding_hub is not None:
        expected.append(
            ("test-engine", "example", "pytorch", 3, "previous-quantization")
        )
    expected.extend(("test-engine", "example", "pytorch", 7, q) for q in fallback or [])
    assert [call.args for call in check_engine.call_args_list] == expected
    assert spec["engines"] == (["test-engine"] if fallback else [])
