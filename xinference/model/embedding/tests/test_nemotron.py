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

# CI: linux

import asyncio
import copy
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from threading import Thread
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import torch
from fastapi import FastAPI
from PIL import Image

from xinference.core.exceptions import InvalidModelRequestError
from xinference.core.utils import filter_virtualenv_packages_by_markers
from xinference.model.utils import virtualenv_discovery_var

from .. import _install, generate_engine_config_by_model_name, nemotron
from ..embed_family import SENTENCE_TRANSFORMER_CLASSES, VLLM_CLASSES, match_embedding
from ..sentence_transformers.core import SentenceTransformerEmbeddingModel
from ..vllm.core import VLLMEmbeddingModel

CLASSES = (
    nemotron.TransformersNemotronEmbeddingModel,
    nemotron.SentenceTransformerNemotronEmbeddingModel,
    nemotron.VLLMNemotronEmbeddingModel,
)


def make_model(
    cls=nemotron.TransformersNemotronEmbeddingModel,
    name="Nemotron-3.5-Embed-1B-BF16",
    model_path="/unused",
    **kwargs,
):
    model = cls(
        "replica", model_path, match_embedding(name, "pytorch", "none"), **kwargs
    )
    model._clean_cache_if_needed = MagicMock()
    return model


@pytest.mark.parametrize("name", sorted(nemotron.MODEL_NAMES))
def test_catalog_and_engine_discovery(name):
    for hub, org, revision in (
        ("huggingface", "nvidia", "main"),
        ("modelscope", "nv-community", "master"),
    ):
        family = match_embedding(name, "pytorch", "none", hub)
        assert family.model_specs[0].model_id == f"{org}/{name}"
        assert family.model_specs[0].model_revision == revision
        assert family.model_ability == ["embed_vision"]
        assert family.dimensions == (4096 if "8B" in name else 2048)
        assert family.max_tokens == 32768
    token = virtualenv_discovery_var.set(True)
    try:
        engines = {}
        generate_engine_config_by_model_name(family, engines)
        expected = (
            {"vllm"}
            if name.endswith("NVFP4")
            else {"vllm", "transformers", "sentence_transformers"}
        )
        # Discovery uses the Hugging Face spec, as does the registry refresh.
        family = match_embedding(name, "pytorch", "none")
        generate_engine_config_by_model_name(family, engines)
        assert set(engines[name]) == expected
        for cls in CLASSES:
            assert (
                cls.match_json(family, SimpleNamespace(model_format="ggufv2"), "none")
                is False
            )
        assert (
            SentenceTransformerEmbeddingModel.match_json(
                family, family.model_specs[0], "none"
            )
            is not True
        )
        assert (
            VLLMEmbeddingModel.match_json(family, family.model_specs[0], "none")
            is not True
        )
    finally:
        virtualenv_discovery_var.reset(token)


def test_registration_is_idempotent():
    _install()
    _install()
    assert SENTENCE_TRANSFORMER_CLASSES.count(CLASSES[1]) == 1
    assert VLLM_CLASSES.count(CLASSES[2]) == 1


@pytest.mark.parametrize("engine", ["transformers", "sentence_transformers", "vllm"])
@pytest.mark.parametrize("size", ["1B", "8B"])
def test_virtualenv_dependencies(engine, size):
    family = match_embedding(f"Nemotron-3.5-Embed-{size}-BF16", "pytorch", "none")
    packages = filter_virtualenv_packages_by_markers(
        family.virtualenv.packages, engine, "13.0"
    )
    assert "pillow" in packages
    assert "#system_torch#" not in packages
    assert "#system_torchvision#" not in packages
    assert ("torch>=2.8.0" in packages) == (engine != "vllm")
    assert ("torchvision>=0.23.0" in packages) == (engine != "vllm")
    # Transformers imports installed torchaudio even when processing images.
    assert ("torchaudio>=2.8.0" in packages) == (engine != "vllm")
    assert ("sentence-transformers[image]>5.4" in packages) == (
        engine == "sentence_transformers"
    )
    assert ("vllm>=0.32.0" in packages) == (engine == "vllm")
    assert ("transformers>=5.17.0,<5.20.0" in packages) == (engine == "vllm")


@pytest.mark.parametrize(
    "engine,package,installed",
    [
        ("transformers", "transformers", "5.19.0"),
        ("sentence_transformers", "sentence-transformers", "5.4"),
        ("vllm", "vllm", "0.31.0"),
        ("vllm", "transformers", "5.16.1"),
        ("vllm", "transformers", "5.20.0"),
    ],
)
def test_old_dependencies_are_rejected(monkeypatch, engine, package, installed):
    monkeypatch.setattr(nemotron, "check_dependency_available", lambda *_: True)
    monkeypatch.setattr(nemotron, "requires", lambda _: ["torch==99.0"])
    monkeypatch.setattr(
        nemotron, "version", lambda name: installed if name == package else "99.0"
    )
    result = nemotron.check_dependencies(engine)
    assert result[0] is False
    assert package in result[1]


@pytest.mark.parametrize("cls", CLASSES[:2])
@pytest.mark.parametrize(
    "torch_version,vision_version,torch_requirement,expected",
    [
        ("2.7.1", "0.22.1", "2.7.1", "requires torch>=2.8.0"),
        ("2.8.0", "0.22.1", "2.7.1", "requires torchvision>=0.23.0"),
        ("2.8.0", "0.24.0", "2.9.0", "requires matching Torch and torchvision"),
    ],
)
def test_native_load_rejects_incompatible_torch(
    monkeypatch, cls, torch_version, vision_version, torch_requirement, expected
):
    versions = {"torch": torch_version, "torchvision": vision_version}
    check_import = MagicMock(return_value=True)
    monkeypatch.setattr(nemotron, "check_dependency_available", check_import)
    monkeypatch.setattr(nemotron, "version", versions.__getitem__)
    monkeypatch.setattr(nemotron, "requires", lambda _: [f"torch=={torch_requirement}"])
    with pytest.raises(ImportError, match=expected):
        make_model(cls, device="cpu").load()
    assert all(call.args[0] == "torch" for call in check_import.call_args_list)


def test_input_roles_and_image_text_order():
    sample = {"text": "caption", "image_url": {"url": "data:image/png;base64,abc"}}
    original = copy.deepcopy(sample)
    samples, prefix = nemotron.normalize_inputs(["text", sample])
    assert prefix == "passage:"
    assert sample == original
    messages = nemotron.make_messages(samples[1], prefix)
    assert messages[0] == {"role": "system", "content": "passage:"}
    assert [part["type"] for part in messages[1]["content"]] == ["image", "text"]
    assert nemotron.normalize_inputs("question", input_type="query")[1] == "query:"
    assert nemotron.normalize_inputs("question", prompt_name="query")[1] == "query:"
    assert (
        nemotron.normalize_inputs("document", prompt_name="document")[1] == "passage:"
    )


@pytest.mark.parametrize(
    "value,kwargs",
    [
        ({}, {}),
        ({"text": 1}, {}),
        ({"image": []}, {}),
        ({"image": ""}, {}),
        ({"audio": "a.wav"}, {}),
        ([1], {}),
        (None, {}),
        ({"image_url": {}}, {}),
        ({"image": "a.png", "image_url": "b.png"}, {}),
        ({"image": "a.png"}, {"input_type": "query"}),
        ("text", {"input_type": "bad"}),
        ("text", {"prompt_name": "bad"}),
        ("text", {"input_type": "query", "prompt_name": "document"}),
    ],
)
def test_invalid_inputs(value, kwargs):
    with pytest.raises(InvalidModelRequestError):
        nemotron.normalize_inputs(value, **kwargs)


@pytest.mark.parametrize("dimensions", [0, -1, 2049, True, 1.5])
def test_invalid_dimensions(dimensions):
    with pytest.raises(InvalidModelRequestError, match="dimensions must"):
        make_model()._create_embedding("text", dimensions=dimensions)


def test_transformers_pooling_ignores_padding_and_renormalizes_after_truncation():
    model = make_model()
    model._processor = MagicMock()

    class Features(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    features = Features(
        input_ids=torch.tensor([[1, 2, 0]]),
        attention_mask=torch.tensor([[1, 1, 0]]),
        pixel_values=torch.ones(1, 3, 2, 2),
    )
    model._processor.apply_chat_template.return_value = features
    model._model = MagicMock(device="cpu", dtype=torch.bfloat16)
    model._model.return_value.last_hidden_state = torch.tensor(
        [[[1.0, 2.0, 4.0], [3.0, 2.0, 2.0], [100.0, 100.0, 100.0]]]
    )
    image = Image.new("RGB", (2, 2))
    result = model._create_embedding(
        {"image": image, "text": "caption"}, dimensions=2, model_uid="public"
    )
    torch.testing.assert_close(
        torch.tensor(result["data"][0]["embedding"]), torch.tensor([2.0, 2.0]) / 8**0.5
    )
    assert result["usage"] == {"prompt_tokens": 2, "total_tokens": 2}
    assert result.token_counts == [2]
    assert result["model"] == "public"
    assert result["model_replica"] == "replica"
    assert features["pixel_values"].dtype == torch.bfloat16
    assert features["input_ids"].dtype == torch.long


def test_transformers_rejects_expanded_input_over_context_limit():
    model = make_model()
    model._processor = MagicMock()
    model._processor.apply_chat_template.return_value = {
        "input_ids": torch.zeros(1, 32769, dtype=torch.long)
    }
    model._model = MagicMock()
    with pytest.raises(InvalidModelRequestError, match="exceeds 32768"):
        model._create_embedding("long")
    model._model.assert_not_called()


def make_embedding_app(model, monkeypatch):
    from xinference.api.restful_api import RESTfulAPI

    async def create_embedding(*args, **kwargs):
        return json.dumps(await model.create_embedding(*args, **kwargs))

    monkeypatch.setattr(
        "xinference.api.restful_api.require_model",
        AsyncMock(
            return_value=SimpleNamespace(
                uid=b"replica", create_embedding=create_embedding
            )
        ),
    )
    api = object.__new__(RESTfulAPI)
    api._set_trace_model = MagicMock()
    api._set_trace_model_type = MagicMock()
    api._check_model_access = MagicMock()
    api._get_supervisor_ref = AsyncMock()
    api._get_model_last_error = AsyncMock(side_effect=lambda _, error: error)
    api._report_error_event = AsyncMock()
    app = FastAPI()
    app.add_api_route("/v1/embeddings", api.create_embedding, methods=["POST"])
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True], ids=["special-first", "valid-first"])
@pytest.mark.parametrize(
    "special_input,options,valid_options,error",
    [
        (["valid document", "overlong"], {}, {}, "exceeds 32768"),
        ("short", {"dimensions": 0}, {}, "dimensions must"),
        ("short", {"dimensions": True}, {"dimensions": 1}, "dimensions must"),
        ("short", {"processing_kwargs": None}, {}, None),
        ("short", {"processing_kwargs": []}, {}, "processing_kwargs must be an object"),
        (
            "short",
            {"processing_kwargs": False},
            {},
            "processing_kwargs must be an object",
        ),
        (
            "short",
            {"processing_kwargs": "padding"},
            {},
            "processing_kwargs must be an object",
        ),
        *[
            (
                "short",
                {"processing_kwargs": {key: value}},
                {},
                f"reserved processing_kwargs: {key}",
            )
            for key, value in [
                ("return_tensors", "pt"),
                ("return_dict", True),
                ("tokenize", True),
                ("conversation", []),
                ("self", None),
            ]
        ],
    ],
    ids=[
        "context",
        "zero-dimensions",
        "boolean-dimensions",
        "null-options",
        "list-options",
        "boolean-options",
        "string-options",
        "reserved-return-tensors",
        "reserved-return-dict",
        "reserved-tokenize",
        "reserved-conversation",
        "reserved-self",
    ],
)
async def test_rest_embedding_validation_is_request_local(
    monkeypatch, special_input, options, valid_options, error, reverse
):
    model = make_model(batch_size=3, batch_interval=0.05)
    model._processor = MagicMock()
    model._model = MagicMock(device="cpu", dtype=torch.float32)
    model._model.return_value.last_hidden_state = torch.tensor([[[3.0, 4.0]]])

    class Features(dict):
        def to(self, device):
            return self

    def preprocess(conversation, **kwargs):
        text = conversation[-1]["content"][-1]["text"]
        length = 32769 if text == "overlong" else 1
        return Features(
            input_ids=torch.zeros(1, length, dtype=torch.long),
            attention_mask=torch.ones(1, length),
        )

    model._processor.apply_chat_template.side_effect = preprocess
    app = make_embedding_app(model, monkeypatch)

    payloads = [
        {"model": "nemotron", "input": special_input, **options},
        {"model": "nemotron", "input": "short", **valid_options},
    ]
    if reverse:
        payloads.reverse()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        responses = await asyncio.gather(
            *(client.post("/v1/embeddings", json=payload) for payload in payloads)
        )

    special, valid = reversed(responses) if reverse else responses
    if error is None:
        assert special.status_code == 200, special.text
        assert special.json() == valid.json()
    else:
        assert special.status_code == 400, special.text
        assert error in special.json()["detail"]
    assert valid.status_code == 200, valid.text
    result = valid.json()
    assert result["model"] == "nemotron"
    assert result["data"][0]["embedding"] == pytest.approx(
        [1.0] if valid_options else [0.6, 0.8]
    )
    assert result["data"][0]["index"] == 0
    assert result["usage"]["total_tokens"] == 1


def test_empty_input_and_sparse_rejection():
    model = make_model()
    model._encode_samples = MagicMock()
    assert model._create_embedding([])["data"] == []
    model._encode_samples.assert_not_called()
    with pytest.raises(InvalidModelRequestError, match="sparse"):
        model._create_embedding("text", return_sparse=True)


@pytest.mark.parametrize("cls", CLASSES)
def test_image_loading_obeys_local_file_policy(tmp_path, monkeypatch, cls):
    image_path = tmp_path / "image with spaces.png"
    Image.new("RGBA", (2, 3)).save(image_path)
    model = make_model(cls)
    model._encode_samples = MagicMock(return_value=([torch.ones(3)], [1]))
    monkeypatch.delenv("XINFERENCE_MEDIA_ALLOW_LOCAL_PATH", raising=False)
    for source in (str(image_path), image_path.as_uri()):
        with pytest.raises(InvalidModelRequestError, match="Local file media"):
            model._create_embedding({"image": source})
    model._encode_samples.assert_not_called()

    monkeypatch.setenv("XINFERENCE_MEDIA_ALLOW_LOCAL_PATH", "true")
    for source in (str(image_path), image_path.as_uri()):
        model._create_embedding({"image": source})
        image = model._encode_samples.call_args.args[0][0]["image"]
        assert image.mode == "RGB"
        assert image.size == (2, 3)


def test_image_loading_obeys_remote_media_policy(monkeypatch):
    png = BytesIO()
    Image.new("RGB", (2, 3)).save(png, format="PNG")
    body = png.getvalue()
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        source = f"http://127.0.0.1:{server.server_port}/image.png"
        model = make_model()
        model._encode_samples = MagicMock(return_value=([torch.ones(3)], [1]))
        monkeypatch.delenv("XINFERENCE_MEDIA_BLOCK_PRIVATE_ADDRESS", raising=False)
        with pytest.raises(InvalidModelRequestError, match="public|private"):
            model._create_embedding({"image": source})
        assert received == []

        monkeypatch.setenv("XINFERENCE_MEDIA_BLOCK_PRIVATE_ADDRESS", "false")
        monkeypatch.setenv("XINFERENCE_MEDIA_MAX_BYTES", "16")
        with pytest.raises(InvalidModelRequestError, match="Media exceeds 16 bytes"):
            model._create_embedding({"image": source})
        model._encode_samples.assert_not_called()

        monkeypatch.setenv("XINFERENCE_MEDIA_MAX_BYTES", str(len(body)))
        model._create_embedding({"image": source})
        image = model._encode_samples.call_args.args[0][0]["image"]
        assert image.size == (2, 3)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", CLASSES)
async def test_batch_image_failure_is_request_local(monkeypatch, cls):
    png = BytesIO()
    Image.new("RGB", (2, 3)).save(png, format="PNG")
    load = MagicMock(side_effect=[png.getvalue(), OSError("download failed")])
    monkeypatch.setattr(nemotron, "load_media_bytes", load)
    model = make_model(cls, batch_size=4, batch_interval=0.05)
    model._encode_samples = MagicMock(
        return_value=([torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])], [2, 3])
    )
    failed, first, second = await asyncio.gather(
        model.create_embedding(
            [
                {"image": "https://example.com/good.png"},
                {"image": "https://example.com/bad.png"},
            ]
        ),
        model.create_embedding("first text"),
        model.create_embedding("second text"),
        return_exceptions=True,
    )
    assert isinstance(failed, InvalidModelRequestError)
    assert "download failed" in failed.client_message
    assert load.call_count == 2
    model._encode_samples.assert_called_once_with(
        [{"text": "first text"}, {"text": "second text"}], "passage:"
    )
    assert first["data"][0]["embedding"] == [1.0, 0.0]
    assert second["data"][0]["embedding"] == [0.0, 1.0]
    assert first["data"][0]["index"] == second["data"][0]["index"] == 0
    assert first["usage"]["total_tokens"] == 2
    assert second["usage"]["total_tokens"] == 3


def test_sentence_transformers_preserves_input_order_and_counts_and_removes_hook():
    model = make_model(CLASSES[1])
    encoder = MagicMock()

    def encode(samples, **kwargs):
        assert kwargs["prompt_name"] == "document"
        assert kwargs["batch_size"] == 1
        text = samples[0].get("text", "")
        length = 32769 if text == "overlong" else len(text) + 1
        encoder.register_forward_pre_hook.call_args.args[0](
            None, ({"attention_mask": torch.ones(1, length)},)
        )
        return torch.tensor([[float(len(text)), 2.0, 9.0]])

    encoder.encode.side_effect = encode
    model._model = encoder
    result = model._create_embedding(
        ["longer", {"image": Image.new("RGB", (2, 2))}, "x"], dimensions=2
    )
    assert result.token_counts == [7, 1, 2]
    assert result["usage"]["total_tokens"] == 10
    assert result["data"][1]["embedding"] == [0.0, 1.0]
    encoder.register_forward_pre_hook.return_value.remove.assert_called_once()
    with pytest.raises(InvalidModelRequestError, match="exceeds 32768"):
        model._create_embedding("overlong")
    assert encoder.register_forward_pre_hook.return_value.remove.call_count == 2
    encoder.encode.side_effect = RuntimeError("encode failure")
    with pytest.raises(RuntimeError, match="encode failure"):
        model._create_embedding("x")
    assert encoder.register_forward_pre_hook.return_value.remove.call_count == 3

    forward_error = ValueError("forward failure")

    def fail_forward(samples, **kwargs):
        encode(samples, **kwargs)
        raise forward_error

    encoder.encode.side_effect = fail_forward
    with pytest.raises(ValueError) as exc:
        model._create_embedding("x", processing_kwargs={"text": {"padding": True}})
    assert exc.value is forward_error
    assert encoder.register_forward_pre_hook.return_value.remove.call_count == 4


def mock_vllm_pooling(monkeypatch):
    class VLLMValidationError(Exception):
        pass

    vllm = ModuleType("vllm")
    vllm.PoolingParams = lambda **kwargs: kwargs
    exceptions = ModuleType("vllm.exceptions")
    exceptions.VLLMValidationError = VLLMValidationError
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.exceptions", exceptions)
    return VLLMValidationError


def test_vllm_prompts_preserve_template_tokens_and_multimodal_data(monkeypatch):
    model = make_model(CLASSES[2])
    mock_vllm_pooling(monkeypatch)
    model._processor = MagicMock()
    model._processor.apply_chat_template.side_effect = [
        "<bos>text",
        "<bos>[IMG]caption",
    ]
    model._model = MagicMock()
    model._model.embed.return_value = [
        SimpleNamespace(
            outputs=SimpleNamespace(embedding=[3.0, 4.0, 9.0]), prompt_token_ids=[1, 2]
        ),
        SimpleNamespace(
            outputs=SimpleNamespace(embedding=[0.0, 2.0, 9.0]),
            prompt_token_ids=[1, 2, 3],
        ),
    ]
    image = Image.new("RGB", (2, 2))
    result = model._create_embedding(
        ["text", {"image": image, "text": "caption"}], dimensions=2
    )
    args, kwargs = model._model.embed.call_args
    assert args[0][0] == {"prompt": "<bos>text"}
    assert args[0][1]["multi_modal_data"]["image"].size == (2, 2)
    assert kwargs["tokenization_kwargs"] == {"add_special_tokens": False}
    assert kwargs["pooling_params"] == {"use_activation": False}
    assert result.token_counts == [2, 3]
    assert result["data"][0]["embedding"] == pytest.approx([0.6, 0.8])


@pytest.mark.asyncio
async def test_vllm_context_error_is_request_local(monkeypatch):
    error_class = mock_vllm_pooling(monkeypatch)
    model = make_model(CLASSES[2], batch_size=2, batch_interval=0.05)
    model._processor = MagicMock()
    model._processor.apply_chat_template.side_effect = (
        lambda messages, **kwargs: messages[-1]["content"][-1]["text"]
    )
    model._model = MagicMock()

    def embed(prompts, **kwargs):
        if any(item["prompt"] == "overlong" for item in prompts):
            raise error_class("Prompt exceeds engine context")
        return [
            SimpleNamespace(
                outputs=SimpleNamespace(embedding=[3.0, 4.0]), prompt_token_ids=[1, 2]
            )
        ]

    model._model.embed.side_effect = embed
    valid, invalid = await asyncio.gather(
        model.create_embedding("valid"),
        model.create_embedding("overlong"),
        return_exceptions=True,
    )
    first_prompts = model._model.embed.call_args_list[0].args[0]
    assert first_prompts == [{"prompt": "valid"}, {"prompt": "overlong"}]
    assert valid["data"][0]["embedding"] == pytest.approx([0.6, 0.8])
    assert valid["usage"]["total_tokens"] == 2
    assert isinstance(invalid, InvalidModelRequestError)
    assert invalid.client_message == "Prompt exceeds engine context"


@pytest.mark.parametrize(
    "cls,start_method",
    [
        (CLASSES[0], None),
        (CLASSES[1], None),
        (CLASSES[2], None),
        (CLASSES[2], "forkserver"),
    ],
)
def test_load_uses_official_entry_points(monkeypatch, cls, start_method):
    monkeypatch.setattr(nemotron, "check_dependencies", lambda _: True)
    transformers = ModuleType("transformers")
    transformers.AutoModel = MagicMock()
    transformers.AutoProcessor = MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    model = make_model(cls, device="cpu")
    if cls is CLASSES[1]:
        st = ModuleType("sentence_transformers")
        st.SentenceTransformer = MagicMock()
        modules = ModuleType("sentence_transformers.models")
        modules.Normalize = torch.nn.Identity
        monkeypatch.setitem(sys.modules, "sentence_transformers", st)
        monkeypatch.setitem(sys.modules, "sentence_transformers.models", modules)
    elif cls is CLASSES[2]:
        if start_method is None:
            monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
        else:
            monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", start_method)
        vllm = ModuleType("vllm")
        vllm.LLM = MagicMock()
        config = ModuleType("vllm.config")
        config.PoolerConfig = lambda **kwargs: kwargs
        monkeypatch.setitem(sys.modules, "vllm", vllm)
        monkeypatch.setitem(sys.modules, "vllm.config", config)
    model.load()
    if cls is CLASSES[0]:
        assert (
            transformers.AutoModel.from_pretrained.call_args.kwargs["device_map"]
            == "cpu"
        )
    elif cls is CLASSES[1]:
        assert st.SentenceTransformer.call_args.kwargs["device"] == "cpu"
    else:
        assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == (start_method or "spawn")
        assert vllm.LLM.call_args.kwargs["max_model_len"] == 32768
        assert vllm.LLM.call_args.kwargs["pooler_config"]["pooling_type"] == "MEAN"


@pytest.fixture(scope="module")
def tiny_nemotron_checkpoint(tmp_path_factory):
    transformers = pytest.importorskip("transformers", minversion="5.20.0")
    from tokenizers import Tokenizer, models, pre_tokenizers

    tmp_path = tmp_path_factory.mktemp("tiny-nemotron")
    vocab = {
        token: index
        for index, token in enumerate(
            [
                "<unk>",
                "<s>",
                "</s>",
                "query",
                ":",
                "passage",
                "caption",
                "document",
                "[IMG_BREAK]",
                "[IMG_END]",
                "[IMG]",
                "<pad>",
            ]
        )
    }
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        additional_special_tokens=["[IMG]", "[IMG_BREAK]", "[IMG_END]"],
    )
    template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{% if message['content'] is string %}{{ message['content'].rstrip(' ') }} "
        "{% else %}{{ message['content'][0]['text'].rstrip(' ') }} {% endif %}"
        "{% else %}{% for part in message['content'] %}"
        "{% if part['type'] == 'image' %}[IMG] "
        "{% else %}{{ part['text'] }}{% endif %}{% endfor %}{% endif %}{% endfor %}"
    )
    processor = transformers.PixtralProcessor(
        image_processor=transformers.PixtralImageProcessor(size={"longest_edge": 56}),
        tokenizer=tokenizer,
        spatial_merge_size=2,
        chat_template=template,
    )
    processor.save_pretrained(tmp_path)
    config = transformers.Mistral3Config(
        text_config={
            "model_type": "ministral3",
            "vocab_size": len(vocab),
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "is_causal": False,
            "pad_token_id": 11,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "head_dim": 8,
            "image_size": 56,
        },
        image_token_index=10,
        spatial_merge_size=2,
    )
    transformers.Mistral3Model(config).save_pretrained(tmp_path)
    records = {
        "modules.json": [
            {
                "idx": 0,
                "name": "0",
                "path": "",
                "type": "sentence_transformers.models.Transformer",
            },
            {
                "idx": 1,
                "name": "1",
                "path": "1_Pooling",
                "type": "sentence_transformers.models.Pooling",
            },
            {
                "idx": 2,
                "name": "2",
                "path": "2_Normalize",
                "type": "sentence_transformers.models.Normalize",
            },
        ],
        "config_sentence_transformers.json": {
            "prompts": {"query": "query: ", "document": "passage: "}
        },
        "sentence_bert_config.json": {
            "max_seq_length": 32768,
            "module_output_name": "token_embeddings",
            "modality_config": {
                modality: {
                    "method": "forward",
                    "method_output_name": "last_hidden_state",
                    **({"format": "structured"} if modality == "message" else {}),
                }
                for modality in ["text", "image", "message"]
            },
        },
        "1_Pooling/config.json": {
            "word_embedding_dimension": 32,
            "pooling_mode_mean_tokens": True,
        },
    }
    for name, value in records.items():
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(value))
    return tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True], ids=["invalid-first", "valid-first"])
@pytest.mark.parametrize(
    "processing_kwargs,error",
    [
        ({"return_attention_mask": False}, "return_attention_mask"),
        ({"text_kwargs": {"return_attention_mask": False}}, "return_attention_mask"),
        ({"text_kwargs": {"padding": "invalid"}}, "PaddingStrategy"),
        ({"text_kwargs": None}, "processing_kwargs.text_kwargs must be an object"),
        ({"images_kwargs": []}, "processing_kwargs.images_kwargs must be an object"),
        (
            {"audio_kwargs": "invalid"},
            "processing_kwargs.audio_kwargs must be an object",
        ),
        ({"videos_kwargs": False}, "processing_kwargs.videos_kwargs must be an object"),
        ({"common_kwargs": None}, "processing_kwargs.common_kwargs must be an object"),
        (
            {"processor_kwargs": True},
            "processing_kwargs.processor_kwargs must be an object",
        ),
        (
            {"processor_kwargs": {"text_kwargs": None}},
            "processing_kwargs.processor_kwargs.text_kwargs must be an object",
        ),
    ],
    ids=[
        "flat-mask",
        "nested-mask",
        "invalid-padding",
        "null-text-options",
        "list-image-options",
        "string-audio-options",
        "boolean-video-options",
        "null-common-options",
        "boolean-processor-options",
        "wrapped-null-text-options",
    ],
)
async def test_rest_transformers_processor_options_are_request_local(
    tiny_nemotron_checkpoint, monkeypatch, processing_kwargs, error, reverse
):
    await check_rest_processor_options(
        tiny_nemotron_checkpoint,
        monkeypatch,
        CLASSES[0],
        processing_kwargs,
        error,
        reverse,
    )


async def check_rest_processor_options(
    tiny_nemotron_checkpoint, monkeypatch, model_cls, processing_kwargs, error, reverse
):
    model = make_model(
        model_cls,
        model_path=str(tiny_nemotron_checkpoint),
        device="cpu",
        torch_dtype="float32",
        batch_size=3,
        batch_interval=0.05,
    )
    model.load()
    app = make_embedding_app(model, monkeypatch)
    payloads = [
        {
            "model": "nemotron",
            "input": "caption",
            "processing_kwargs": processing_kwargs,
        },
        {"model": "nemotron", "input": "document"},
    ]
    if reverse:
        payloads.reverse()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        responses = await asyncio.gather(
            *(client.post("/v1/embeddings", json=payload) for payload in payloads)
        )

    special, plain = reversed(responses) if reverse else responses
    assert (special.status_code, plain.status_code) == (400 if error else 200, 200), (
        special.text,
        plain.text,
    )
    if error:
        assert error in special.json()["detail"]
    successful = [plain] if error else [special, plain]
    for response in successful:
        result = response.json()
        assert len(result["data"][0]["embedding"]) == 32
        assert result["usage"]["total_tokens"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True], ids=["special-first", "plain-first"])
@pytest.mark.parametrize(
    "processing_kwargs,error",
    [
        ({"text": True}, "processing_kwargs.text must be an object"),
        ({"text": {"padding": "invalid"}}, "PaddingStrategy"),
        ({"text": {"return_attention_mask": False}}, "return_attention_mask"),
        ({"text": {"padding": "longest"}}, None),
        (
            {"chat_template": {"text_kwargs": None}},
            "processing_kwargs.chat_template.text_kwargs must be an object",
        ),
        ({"common": {"return_tensors": "np"}}, "return_tensors"),
    ],
    ids=[
        "boolean-text",
        "invalid-padding",
        "no-mask",
        "valid-padding",
        "null-template-text",
        "numpy-mask",
    ],
)
async def test_rest_sentence_transformers_processor_options_are_request_local(
    tiny_nemotron_checkpoint, monkeypatch, processing_kwargs, error, reverse
):
    pytest.importorskip("sentence_transformers", minversion="6.1.0")
    await check_rest_processor_options(
        tiny_nemotron_checkpoint,
        monkeypatch,
        CLASSES[1],
        processing_kwargs,
        error,
        reverse,
    )


def test_native_transformers_and_sentence_transformers_agree(tiny_nemotron_checkpoint):
    pytest.importorskip("sentence_transformers", minversion="6.1.0")
    inputs = [
        "document",
        {"image": Image.new("RGB", (56, 56))},
        {"image": Image.new("RGB", (56, 56)), "text": "caption"},
    ]
    results = []
    raw_results = []
    for cls in CLASSES[:2]:
        family = match_embedding("Nemotron-3.5-Embed-1B-BF16", "pytorch", "none")
        family.dimensions = 32
        model = cls(
            "tiny",
            str(tiny_nemotron_checkpoint),
            family,
            device="cpu",
            torch_dtype="float32",
        )
        model.load()
        result = model._create_embedding(inputs, dimensions=16)
        assert len(result["data"]) == 3
        assert (
            len(
                model._create_embedding("query", input_type="query")["data"][0][
                    "embedding"
                ]
            )
            == 32
        )
        results.append(result)
        raw = model._create_embedding(inputs, dimensions=16, normalize_embeddings=False)
        assert raw.token_counts == result.token_counts
        for normalized, unnormalized in zip(result["data"], raw["data"]):
            vector = torch.tensor(unnormalized["embedding"])
            assert not torch.isclose(vector.norm(), torch.tensor(1.0))
            torch.testing.assert_close(
                torch.nn.functional.normalize(vector, dim=0),
                torch.tensor(normalized["embedding"]),
            )
        raw_results.append(raw)
    assert results[0].token_counts == results[1].token_counts
    for left, right in zip(results[0]["data"], results[1]["data"]):
        vector = torch.tensor(left["embedding"])
        torch.testing.assert_close(
            vector, torch.tensor(right["embedding"]), atol=1e-5, rtol=1e-4
        )
        torch.testing.assert_close(vector.norm(), torch.tensor(1.0))
    for left, right in zip(raw_results[0]["data"], raw_results[1]["data"]):
        torch.testing.assert_close(
            torch.tensor(left["embedding"]),
            torch.tensor(right["embedding"]),
            atol=1e-5,
            rtol=1e-4,
        )
