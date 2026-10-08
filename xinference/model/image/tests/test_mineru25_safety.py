import ast
import asyncio
import glob
import os
import re
import shutil
import tempfile
import time
import zipfile
from base64 import b64encode
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from starlette.requests import Request


def load_vendor_nodes(relative_path, names, namespace):
    """Exercise vendor helpers without loading optional GPU dependencies."""
    source = Path(__file__).resolve().parents[3] / "thirdparty/mineru" / relative_path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace
    )
    return namespace


def cache_namespace():
    return load_vendor_nodes(
        "utils/cache_key.py", {"_IdentityKey", "config_cache_key"}, {}
    )


def test_cache_key_handles_nested_configuration_and_identity():
    key = cache_namespace()["config_cache_key"]
    options = {"limits": {"image": 2}, "processors": ["a", "b"]}
    assert key(options) == key(dict(reversed(list(options.items()))))
    original = key(options)
    options["limits"]["image"] = 3
    assert original != key(options)
    assert key([1]) != key((1,))
    opaque = SimpleNamespace(value=1)
    assert key(opaque) == key(opaque)
    assert key(opaque) != key(SimpleNamespace(value=1))
    assert isinstance(hash(key({"opaque": opaque})), int)


def test_atomic_cache_respects_device_and_weights():
    init = Mock(side_effect=lambda **kwargs: object())
    namespace = load_vendor_nodes(
        "backend/pipeline/model_init.py",
        {"AtomModelSingleton"},
        {
            "config_cache_key": cache_namespace()["config_cache_key"],
            "AtomicModel": SimpleNamespace(
                WiredTable="wired", WirelessTable="wireless", OCR="ocr"
            ),
            "atom_model_init": init,
        },
    )
    manager = namespace["AtomModelSingleton"]()
    first = manager.get_atom_model("layout", device="cpu", weight="a")
    assert manager.get_atom_model("layout", weight="a", device="cpu") is first
    assert manager.get_atom_model("layout", device="cuda", weight="a") is not first
    assert manager.get_atom_model("layout", device="cpu", weight="b") is not first
    assert init.call_count == 3


def test_atomic_initialization_raises_instead_of_exiting():
    atomic = SimpleNamespace(
        **{
            name: name
            for name in [
                "Layout",
                "MFD",
                "MFR",
                "OCR",
                "WirelessTable",
                "WiredTable",
                "TableCls",
                "ImgOrientationCls",
            ]
        }
    )
    namespace = load_vendor_nodes(
        "backend/pipeline/model_init.py",
        {"atom_model_init"},
        {"AtomicModel": atomic, "table_cls_model_init": lambda: None},
    )
    with pytest.raises(ValueError, match="not allowed"):
        namespace["atom_model_init"]("unknown")
    with pytest.raises(RuntimeError, match="Failed to initialize"):
        namespace["atom_model_init"]("TableCls")


def test_vlm_cache_respects_nested_options():
    client = Mock(side_effect=lambda **kwargs: object())
    namespace = load_vendor_nodes(
        "backend/vlm/vlm_analyze.py",
        {"ModelSingleton"},
        {
            "config_cache_key": cache_namespace()["config_cache_key"],
            "MinerUClient": client,
            "time": time,
            "logger": Mock(),
        },
    )
    manager = namespace["ModelSingleton"]()
    first = manager.get_model("http-client", None, "url", limits={"image": 2})
    assert manager.get_model("http-client", None, "url", limits={"image": 2}) is first
    assert (
        manager.get_model("http-client", None, "url", limits={"image": 3}) is not first
    )
    assert client.call_count == 2


def test_null_title_configuration_is_disabled():
    source = (
        Path(__file__).resolve().parents[3]
        / "thirdparty/mineru/backend/vlm/model_output_to_middle_json.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    conditional = next(node for node in tree.body if isinstance(node, ast.If))
    namespace = {"llm_aided_config": {"title_aided": None}}
    exec(
        compile(ast.Module(body=[conditional], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    assert namespace["title_aided_config"] == {}


@pytest.mark.parametrize("kwargs", ["[]", '"string"', "null", "1", "true", "{invalid"])
def test_invalid_api_kwargs_are_bad_requests(monkeypatch, kwargs):
    from xinference.api import restful_api

    model_ref = SimpleNamespace(docanalyze=AsyncMock())
    monkeypatch.setattr(restful_api, "require_model", AsyncMock(return_value=model_ref))
    api = SimpleNamespace(
        _get_supervisor_ref=Mock(),
        _report_error_event=AsyncMock(),
        _add_running_task=Mock(),
        _set_trace_model=Mock(),
        _set_trace_model_type=Mock(),
        _check_model_access=Mock(),
    )
    file = UploadFile(file=BytesIO(b"pdf"), filename="document.pdf", size=3)
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            restful_api.RESTfulAPI.create_doc_analyze(
                api,
                request=Request({"type": "http", "headers": []}),
                model="mineru",
                file=file,
                kwargs=kwargs,
            )
        )
    assert error.value.status_code == 400
    model_ref.docanalyze.assert_not_called()
    api._add_running_task.assert_not_called()


@pytest.mark.parametrize(
    "stage", ["json", "zip", "unsupported", "read", "parse", "cancel", "zip_error"]
)
def test_vendor_api_cleans_request_files(tmp_path, monkeypatch, stage):
    output = tmp_path / "output"
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(archive_dir))

    async def parse(**kwargs):
        if stage == "parse":
            raise RuntimeError("parse failure")
        if stage == "cancel":
            raise asyncio.CancelledError()
        directory = Path(kwargs["output_dir"]) / "document" / "vlm"
        directory.mkdir(parents=True)
        (directory / "document.md").write_text("parsed", encoding="utf-8")

    namespace = {
        "os": os,
        "re": re,
        "shutil": shutil,
        "tempfile": tempfile,
        "zipfile": zipfile,
        "glob": glob,
        "b64encode": b64encode,
        "Path": Path,
        "List": List,
        "Optional": Optional,
        "File": File,
        "Form": Form,
        "UploadFile": UploadFile,
        "FileResponse": FileResponse,
        "JSONResponse": JSONResponse,
        "BackgroundTask": BackgroundTask,
    }
    namespace.update(
        app=SimpleNamespace(state=SimpleNamespace(config={})),
        logger=Mock(),
        aio_do_parse=parse,
        read_fn=Mock(
            side_effect=RuntimeError("read failure") if stage == "read" else None,
            return_value=b"pdf",
        ),
        pdf_suffixes=["pdf"],
        image_suffixes=["png"],
        guess_suffix_by_path=lambda path: "txt" if stage == "unsupported" else "pdf",
        __version__="test",
    )
    load_vendor_nodes(
        "cli/fast_api.py",
        {
            "parse_pdf",
            "cleanup_file",
            "sanitize_filename",
            "encode_image",
            "get_infer_result",
        },
        namespace,
    )
    if stage == "zip_error":
        monkeypatch.setattr(
            zipfile, "ZipFile", Mock(side_effect=RuntimeError("zip failure"))
        )
    file = UploadFile(file=BytesIO(b"pdf"), filename="document.pdf")
    coroutine = namespace["parse_pdf"](
        files=[file],
        output_dir=str(output),
        lang_list=["ch"],
        backend="vlm",
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        server_url=None,
        return_md=True,
        return_middle_json=False,
        return_model_output=False,
        return_content_list=False,
        return_images=False,
        response_format_zip=stage in ["zip", "zip_error"],
        start_page_id=0,
        end_page_id=99999,
    )
    if stage == "cancel":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(coroutine)
    else:
        response = asyncio.run(coroutine)
        expected = (
            400
            if stage in ["unsupported", "read"]
            else 500 if stage in ["parse", "zip_error"] else 200
        )
        assert response.status_code == expected
        if stage == "zip":
            assert Path(response.path).exists()
            with zipfile.ZipFile(response.path) as archive:
                assert archive.read("document/document.md") == b"parsed"
            asyncio.run(response.background())
    assert output.exists()
    assert list(output.iterdir()) == []
    assert list(archive_dir.iterdir()) == []
