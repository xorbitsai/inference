import asyncio
import sys
from io import BytesIO
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException, UploadFile
from pydantic import ValidationError

from ..docanalyze import mineru25
from ..docanalyze.schemas import DocAnalyzeCodeResponse


def mock_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)


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
