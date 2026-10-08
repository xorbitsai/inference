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

import asyncio
import dataclasses
import queue
import sys
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

from .. import BUILTIN_VIDEO_MODELS, _install, native
from ..core import create_video_model_instance, match_diffusion
from ..engine_family import VIDEO_ENGINES
from ..native import NATIVE_VIDEO_MODELS, SGLangVideoModel, VLLMVideoModel


@pytest.fixture
def native_registry(monkeypatch):
    monkeypatch.setattr(native.platform, "system", lambda: "Linux")
    monkeypatch.setattr(native, "has_cuda_device", lambda: True)
    _install()
    yield
    monkeypatch.undo()
    _install()


def make_model(cls):
    spec = next(
        s for s in BUILTIN_VIDEO_MODELS["Wan2.1-1.3B"] if s.engine == cls.engine_name
    )
    return cls("uid", "/model", spec)


@pytest.mark.parametrize("engine", ["SGLang", "vLLM"])
def test_native_video_registry_and_factory(native_registry, engine):
    for name in NATIVE_VIDEO_MODELS:
        assert engine in VIDEO_ENGINES[name]
        spec = match_diffusion(name, "huggingface", model_engine=engine)
        assert spec.model_ability == ["text2video"]
        model = create_video_model_instance(
            "uid",
            name,
            download_hub="huggingface",
            model_engine=engine,
            model_path="/model",
            enable_virtual_env=False,
        )
        assert model.allow_batch is True
        assert model.model_ability == ["text2video"]
    # Keep the previous default and do not advertise unimplemented I2V.
    assert next(iter(VIDEO_ENGINES["Wan2.1-1.3B"])) == "diffusers"
    assert engine not in VIDEO_ENGINES["Wan2.2-i2v-A14B"]


@pytest.mark.parametrize("cls", [SGLangVideoModel, VLLMVideoModel])
def test_host_and_unsupported_adapters(native_registry, monkeypatch, cls):
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    assert cls.check_host() is not True
    spec = make_model(cls)._model_spec
    with pytest.raises(ValueError, match="GGUF"):
        cls("uid", "/model", spec, gguf_model_path="/gguf")
    with pytest.raises(ValueError, match="Lightning"):
        cls("uid", "/model", spec, lightning_version="4-step")


@pytest.mark.asyncio
async def test_sglang_concurrent_video_results(native_registry, monkeypatch):
    model = make_model(SGLangVideoModel)
    barrier = threading.Barrier(2, timeout=5)
    received = []

    def sampling(prompt, n, width, height, config):
        return dict(prompt=prompt, n=n, width=width, height=height, **config)

    def generate(sampling_params_kwargs):
        received.append(sampling_params_kwargs)
        barrier.wait()
        return [
            SimpleNamespace(frames=[sampling_params_kwargs["prompt"]])
            for _ in range(sampling_params_kwargs["n"])
        ]

    monkeypatch.setattr(model, "_build_sampling_params", sampling)
    model._model = SimpleNamespace(generate=generate)
    monkeypatch.setattr(
        native, "_export_videos", lambda videos, fps, fmt: (videos, fps, fmt)
    )
    results = await asyncio.gather(
        model.text_to_video("first", n=2, width=640, num_inference_steps=4, fps=20),
        model.text_to_video("second", response_format="url"),
    )
    assert results == [
        ([["first"], ["first"]], 20, "b64_json"),
        ([["second"]], 16, "url"),
    ]
    first = next(r for r in received if r["prompt"] == "first")
    assert first["width"] == 640
    assert first["num_frames"] == 81
    assert first["num_inference_steps"] == 4
    assert model._model_spec.default_generate_config["width"] == 832


@pytest.mark.asyncio
async def test_vllm_video_dispatch_routes_reverse_order(native_registry, monkeypatch):
    @dataclasses.dataclass
    class OutputMessage:
        request_id: str
        engine_outputs: object
        finished: bool = True

    class ErrorMessage:
        pass

    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.engine.messages",
        SimpleNamespace(OutputMessage=OutputMessage, ErrorMessage=ErrorMessage),
    )
    model = make_model(VLLMVideoModel)
    requests = []
    outputs = queue.Queue()

    def add_request(**request):
        requests.append(request)
        if len(requests) == 2:
            for index, req in reversed(list(enumerate(requests))):
                frames = [Image.new("RGB", (2, 2), (index, 0, 0))]
                # Both output layouts used by the native Wan postprocessor.
                result = (
                    SimpleNamespace(images=[frames])
                    if index == 0
                    else SimpleNamespace(multimodal_output={"video": [frames]})
                )
                outputs.put(OutputMessage(req["request_id"], result))

    def get_output(timeout):
        try:
            return outputs.get(timeout=timeout)
        except queue.Empty:
            return None

    model._model = SimpleNamespace(
        engine=SimpleNamespace(add_request=add_request, try_get_output=get_output),
        close=lambda: None,
    )
    monkeypatch.setattr(
        model,
        "_build_sampling_params",
        lambda n, w, h, c: dict(n=n, width=w, height=h, **c),
    )
    monkeypatch.setattr(native, "_export_videos", lambda videos, fps, fmt: videos)
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                model.text_to_video("first", seed=10, negative_prompt="bad"),
                model.text_to_video("second", seed=20),
            ),
            timeout=5,
        )
        assert [r[0][0].getpixel((0, 0))[0] for r in results] == [0, 1]
        assert requests[0]["prompt"] == {
            "prompt": "first",
            "negative_prompt": "bad",
            "modalities": ["video"],
        }
        assert requests[0]["sampling_params_list"][0]["seed"] == 10
        assert requests[0]["sampling_params_list"][0]["output_type"] == "pil"
    finally:
        model.stop()


@pytest.mark.asyncio
async def test_missing_outputs_and_invalid_arguments(native_registry, monkeypatch):
    model = make_model(SGLangVideoModel)

    async def no_videos(*args):
        return []

    monkeypatch.setattr(model, "_generate_videos", no_videos)
    with pytest.raises(RuntimeError, match="Expected 1 videos"):
        await model.text_to_video("prompt")
    with pytest.raises(ValueError, match="response format"):
        await model.text_to_video("prompt", response_format="invalid")
    with pytest.raises(ValueError, match="n must"):
        await model.text_to_video("prompt", n=0)


@pytest.mark.parametrize("cls", [SGLangVideoModel, VLLMVideoModel])
def test_actor_allows_native_concurrency(native_registry, cls):
    from ....core.model import ModelActor

    model = make_model(cls)
    actor = ModelActor("supervisor", "worker", model, "uid-0")
    assert actor._lock is None


@pytest.mark.parametrize("response_format", ["url", "b64_json"])
def test_video_export_response_and_cleanup(tmp_path, monkeypatch, response_format):
    import base64
    from pathlib import Path

    monkeypatch.setattr(native, "XINFERENCE_VIDEO_DIR", str(tmp_path))

    def export(frames, path, fps):
        assert fps == 16
        Path(path).write_bytes(frames)

    monkeypatch.setitem(
        sys.modules, "diffusers.utils", SimpleNamespace(export_to_video=export)
    )
    response = native._export_videos([b"first", b"second"], 16, response_format)
    assert len(response["data"]) == 2
    if response_format == "b64_json":
        assert [base64.b64decode(item["b64_json"]) for item in response["data"]] == [
            b"first",
            b"second",
        ]
        assert not list(tmp_path.iterdir())
    else:
        assert [Path(item["url"]).read_bytes() for item in response["data"]] == [
            b"first",
            b"second",
        ]
        assert len(list(tmp_path.iterdir())) == 2
