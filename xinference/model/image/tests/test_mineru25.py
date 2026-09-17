import ast
import asyncio
import json
import logging
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, mock_open

import pytest
from fastapi import HTTPException, UploadFile
from packaging.version import parse
from pydantic import ValidationError

from ..docanalyze import mineru25
from ..docanalyze.schemas import DocAnalyzeCodeResponse


def mock_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)


def test_docanalyze_response_preserves_serialized_json(monkeypatch):
    from xinference.api import restful_api
    from xinference.core.model import ModelActor

    result = [{"type": "text", "text": "document content"}]
    actor = SimpleNamespace(_lock=None, _add_running_task=Mock())
    payload = asyncio.run(
        ModelActor._call_wrapper(actor, "json", AsyncMock(return_value=result))
    )
    model_ref = SimpleNamespace(docanalyze=AsyncMock(return_value=payload))
    monkeypatch.setattr(restful_api, "require_model", AsyncMock(return_value=model_ref))
    api = SimpleNamespace(
        _get_supervisor_ref=Mock(),
        _report_error_event=AsyncMock(),
        _add_running_task=Mock(),
    )
    file = UploadFile(file=BytesIO(b"pdf"), filename="document.pdf", size=3)
    response = asyncio.run(
        restful_api.RESTfulAPI.create_doc_analyze(
            api, model="mineru", file=file, kwargs=None
        )
    )
    assert response.status_code == 200
    assert response.media_type == "application/json"
    assert json.loads(response.body) == result


@pytest.mark.parametrize("invalid_json", [False, True])
def test_download_config_closes_file(invalid_json):
    # Load only the helper to avoid importing optional download dependencies.
    source = (
        Path(__file__).resolve().parents[3] / "thirdparty/mineru/cli/models_download.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "download_and_modify_json"
    )
    file_open = mock_open(
        read_data=(
            "invalid"
            if invalid_json
            else '{"config_version": "1.3.0", "models-dir": {}}'
        )
    )
    namespace = {
        "json": json,
        "os": SimpleNamespace(path=SimpleNamespace(exists=lambda path: True)),
        "open": file_open,
        "download_json": Mock(),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    if invalid_json:
        with pytest.raises(json.JSONDecodeError):
            namespace[function.name]("url", "config.json", {})
    else:
        namespace[function.name]("url", "config.json", {"models-dir": {"vlm": "model"}})
    assert file_open.return_value.__exit__.call_count == (1 if invalid_json else 2)
    namespace["download_json"].assert_not_called()


@pytest.mark.parametrize(
    "stage", [None, "extract", "middle", "content", "validate", "cancel"]
)
def test_document_is_closed_on_success_failure_and_cancellation(monkeypatch, stage):
    pdf_doc = Mock()
    model = mineru25.Mineru2_5Model(
        "mineru", model_spec=SimpleNamespace(model_ability=["docanalyze"])
    )
    extract = AsyncMock(return_value=[])
    if stage == "extract":
        extract.side_effect = RuntimeError("extract failed")
    elif stage == "cancel":
        extract.side_effect = asyncio.CancelledError()
    model._model = SimpleNamespace(aio_batch_two_step_extract=extract)
    mock_module(
        monkeypatch,
        "xinference.thirdparty.mineru.cli.common",
        convert_pdf_bytes_to_bytes_by_pypdfium2=lambda data: data,
    )
    mock_module(
        monkeypatch,
        "xinference.thirdparty.mineru.utils.enum_class",
        ImageType=SimpleNamespace(PIL="pil"),
        MakeMode=SimpleNamespace(CONTENT_LIST="content_list"),
    )
    mock_module(
        monkeypatch,
        "xinference.thirdparty.mineru.utils.pdf_image_tools",
        load_images_from_pdf=lambda *args, **kwargs: ([{"img_pil": "image"}], pdf_doc),
    )
    monkeypatch.setattr(mineru25, "read_fn", lambda data, name: data)
    for name, failure_stage, result in [
        ("result_to_middle_json", "middle", {"pdf_info": []}),
        ("vlm_union_make", "content", []),
        ("check_json_structure", "validate", None),
    ]:
        function = Mock(return_value=result)
        if stage == failure_stage:
            function.side_effect = RuntimeError(f"{stage} failed")
        monkeypatch.setattr(mineru25, name, function)

    if stage:
        expected_error = asyncio.CancelledError if stage == "cancel" else RuntimeError
        with pytest.raises(expected_error):
            asyncio.run(model.docanalyze(b"pdf", "document.pdf"))
    else:
        assert asyncio.run(model.docanalyze(b"pdf", "document.pdf")) == []
    pdf_doc.close.assert_called_once_with()


@pytest.mark.parametrize("batch_size", [None, 2])
def test_batch_size_is_not_forwarded_to_vllm(monkeypatch, batch_size):
    client = Mock()
    engine_args = Mock(return_value="engine_args")
    mock_module(
        monkeypatch,
        "xinference.thirdparty.mineru_vl_utils",
        MinerUClient=client,
        MinerULogitsProcessor=Mock(),
    )
    mock_module(
        monkeypatch,
        "xinference.thirdparty.mineru.backend.vlm.custom_logits_processors",
        enable_custom_logits_processors=lambda: False,
    )
    mock_module(monkeypatch, "vllm.engine.arg_utils", AsyncEngineArgs=engine_args)
    mock_module(
        monkeypatch,
        "vllm.v1.engine.async_llm",
        AsyncLLM=SimpleNamespace(from_engine_args=Mock()),
    )
    monkeypatch.setattr(mineru25, "VLLM_INSTALLED", False)
    monkeypatch.setenv("VLLM_USE_V1", "0")
    kwargs = {} if batch_size is None else {"batch_size": batch_size}
    model = mineru25.Mineru2_5Model(
        "mineru",
        model_path="model",
        model_spec=SimpleNamespace(model_ability=[]),
        **kwargs,
    )
    model.load()
    assert "batch_size" not in engine_args.call_args.kwargs
    assert client.call_args.kwargs["batch_size"] == (
        8 if batch_size is None else batch_size
    )


def test_code_schema_retains_and_validates_code_fields():
    block = dict(
        type="code",
        bbox=[0, 0, 1, 1],
        page_idx=0,
        sub_type="code",
        code_body="print('hello')",
        code_caption=["Example"],
        guess_lang="python",
    )
    response = DocAnalyzeCodeResponse(**block)
    assert response.dict()["code_body"] == block["code_body"]
    assert response.dict()["code_caption"] == block["code_caption"]
    assert "table_caption" not in response.dict()
    with pytest.raises(ValidationError):
        DocAnalyzeCodeResponse(**{**block, "code_body": ["invalid"]})
    with pytest.raises(ValidationError):
        DocAnalyzeCodeResponse(**{**block, "code_caption": "invalid"})


@pytest.mark.parametrize(
    "filename,size",
    [(None, None), ("", None), ("document.pdf", 0), ("document.pdf", None)],
)
def test_empty_uploads_return_bad_request(monkeypatch, filename, size):
    from xinference.api import restful_api

    model_ref = SimpleNamespace(docanalyze=AsyncMock())
    monkeypatch.setattr(restful_api, "require_model", AsyncMock(return_value=model_ref))
    api = SimpleNamespace(
        _get_supervisor_ref=Mock(),
        _report_error_event=AsyncMock(),
        _add_running_task=Mock(),
    )
    file = UploadFile(file=BytesIO(b""), filename=filename, size=size)
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            restful_api.RESTfulAPI.create_doc_analyze(
                api, model="mineru", file=file, kwargs=None
            )
        )
    assert error.value.status_code == 400
    model_ref.docanalyze.assert_not_called()


@pytest.mark.parametrize(
    "backend,error_type",
    [("vllm-engine", NotImplementedError), ("vllm-async-engine", RuntimeError)],
)
def test_backend_and_version_errors_use_specific_types(
    monkeypatch, backend, error_type
):
    mock_module(
        monkeypatch,
        "xinference.thirdparty.mineru_vl_utils",
        MinerUClient=Mock(),
        MinerULogitsProcessor=Mock(),
    )
    mock_module(
        monkeypatch,
        "xinference.thirdparty.mineru.backend.vlm.custom_logits_processors",
        enable_custom_logits_processors=lambda: False,
    )
    monkeypatch.setattr(mineru25, "VLLM_INSTALLED", True)
    monkeypatch.setattr(mineru25, "VLLM_VERSION", parse("0.10.0"))
    model = mineru25.Mineru2_5Model(
        "mineru", model_spec=SimpleNamespace(model_ability=[]), backend=backend
    )
    with pytest.raises(error_type):
        model.load()


def test_unsupported_file_suffix_raises_value_error(monkeypatch):
    mock_module(
        monkeypatch,
        "xinference.thirdparty.mineru.cli.common",
        image_suffixes=["png"],
        pdf_suffixes=["pdf"],
    )
    mock_module(
        monkeypatch,
        "xinference.thirdparty.mineru.utils.pdf_image_tools",
        images_bytes_to_pdf_bytes=Mock(),
    )
    with pytest.raises(ValueError, match="Unknown file suffix: txt"):
        mineru25.read_fn(b"text", "document.txt")


def test_docanalyze_logging_omits_file_bytes(caplog):
    from xinference.core.model import ModelActor

    actor = SimpleNamespace(
        _require_ready=Mock(),
        _model=SimpleNamespace(docanalyze=AsyncMock()),
        _call_wrapper_json=AsyncMock(return_value="[]"),
    )
    with caplog.at_level(logging.DEBUG, logger="xinference.core.model"):
        result = asyncio.run(
            ModelActor.docanalyze(
                actor, file_bytes=b"private-document-payload", file_name="document.pdf"
            )
        )
    assert result == "[]"
    assert "Enter docanalyze" in caplog.text
    assert "document.pdf" in caplog.text
    assert "private-document-payload" not in caplog.text
    assert "file_bytes=" not in caplog.text
