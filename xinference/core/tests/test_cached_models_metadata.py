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

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from .. import worker as worker_module
from ..worker import WorkerActor


async def cached_models(monkeypatch, entries, engines):
    worker = SimpleNamespace(
        address="worker:9978",
        _cache_tracker_ref=SimpleNamespace(
            list_cached_models=AsyncMock(return_value=deepcopy(entries))
        ),
        discover_engines=(
            MagicMock(side_effect=engines)
            if callable(engines)
            else MagicMock(return_value=engines)
        ),
    )
    monkeypatch.setattr(
        worker_module, "get_engine_params_by_name", worker.discover_engines
    )
    monkeypatch.setattr(
        worker_module,
        "get_engine_params_by_name_with_virtual_env",
        worker.discover_engines,
    )
    monkeypatch.setattr(worker_module, "get_real_path", lambda path: None)
    monkeypatch.setattr(worker_module, "get_path_size", lambda *args, **kwargs: 42)
    result = await WorkerActor.list_cached_models(worker)
    return result, worker


def entry(model_type="LLM", version="demo--7B--pytorch--none"):
    return {
        "model_name": "demo",
        "model_type": model_type,
        "model_version": version,
        "model_file_location": "/models/demo",
    }


@pytest.mark.asyncio
async def test_llm_engines_match_cached_spec_on_each_worker(monkeypatch):
    entries = [entry(), entry(version="demo--7B--ggufv2--Q4_K_M")]
    params = {
        "model_name": "demo",
        "model_format": "pytorch",
        "model_size_in_billions": 7,
        "quantizations": ["none"],
    }
    first, first_worker = await cached_models(
        monkeypatch,
        entries,
        {"Transformers": [params, params], "vLLM": "Not installed"},
    )
    second, second_worker = await cached_models(
        monkeypatch,
        entries,
        {
            "vLLM": [params],
            "llama.cpp": [
                {**params, "model_format": "ggufv2", "quantizations": ["Q4_K_M"]}
            ],
        },
    )
    assert [item["model_engine"] for item in first] == [["Transformers"], []]
    assert [item["model_engine"] for item in second] == [["vLLM"], ["llama.cpp"]]
    for worker in (first_worker, second_worker):
        worker.discover_engines.assert_called_once_with(
            "LLM",
            "demo",
            enable_virtual_env=worker_module.XINFERENCE_ENABLE_VIRTUAL_ENV,
        )
    assert first[0]["model_type"] == "LLM"
    assert first[0]["path"] == "/models/demo"
    assert first[0]["actor_ip_address"] == "worker:9978"
    assert first[0]["size_bytes"] == 42
    for worker in (first_worker, second_worker):
        returned = worker._cache_tracker_ref.list_cached_models.return_value
        assert all("model_engine" not in item for item in returned)


@pytest.mark.asyncio
async def test_embedding_engines_match_format_and_quantization(monkeypatch):
    entries = [
        entry("embedding", "demo--512--768--pytorch--none"),
        entry("embedding", "demo--512--768--ggufv2--Q8_0"),
    ]
    result, _ = await cached_models(
        monkeypatch,
        entries,
        {
            "sentence_transformers": [
                {
                    "model_name": "demo",
                    "model_format": "pytorch",
                    "quantization": "none",
                }
            ],
            "llama.cpp": [
                {"model_name": "demo", "model_format": "ggufv2", "quantization": "Q8_0"}
            ],
        },
    )
    assert [item["model_engine"] for item in result] == [
        ["sentence_transformers"],
        ["llama.cpp"],
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_type,version",
    [("rerank", "demo"), ("image", "demo--controlnet"), ("world", "demo")],
)
async def test_non_llm_engines(monkeypatch, model_type, version):
    result, _ = await cached_models(
        monkeypatch,
        [entry(model_type, version)],
        {"available": [{"model_name": "demo"}], "unavailable": "Missing dependencies"},
    )
    assert result[0]["model_engine"] == ["available"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", ["audio", "video"])
async def test_engine_specific_cache_names(monkeypatch, model_type):
    module = __import__(f"xinference.model.{model_type}", fromlist=[model_type])
    families = [
        SimpleNamespace(
            model_name="demo",
            cache_name=cache_name,
            engine=engine,
            quantization=None,
            model_hub="huggingface",
        )
        for engine, cache_name in [("PyTorch", None), ("MLX", "demo-mlx")]
    ]
    monkeypatch.setattr(
        module, f"BUILTIN_{model_type.upper()}_MODELS", {"demo": families}
    )
    result, _ = await cached_models(
        monkeypatch,
        [entry(model_type, "demo"), entry(model_type, "demo-mlx")],
        {"PyTorch": [{"model_name": "demo"}], "MLX": [{"model_name": "demo"}]},
    )
    assert [item["model_engine"] for item in result] == [["PyTorch"], ["MLX"]]


@pytest.mark.asyncio
async def test_unknown_metadata_and_discovery_failure_keep_cache_visible(monkeypatch):
    entries = [entry(), entry(None, "old"), entry("flexible", "custom")]
    result, worker = await cached_models(monkeypatch, entries, None)
    assert [item["model_engine"] for item in result] == [None, None, None]
    assert [item["model_type"] for item in result] == ["LLM", None, "flexible"]
    worker.discover_engines.side_effect = RuntimeError("Cannot discover")
    result = await WorkerActor.list_cached_models(worker)
    assert len(result) == 3
    assert result[0]["model_engine"] is None


@pytest.mark.asyncio
async def test_tracker_engine_metadata_does_not_override_worker_discovery(monkeypatch):
    entries = [{**entry(), "model_engine": ["known"]}]
    result, worker = await cached_models(monkeypatch, entries, {})
    assert result[0]["model_engine"] == []
    worker.discover_engines.assert_called_once()


@pytest.mark.asyncio
async def test_cache_api_adds_unknown_fields_for_older_workers():
    import json

    from ...api.routers import admin

    api = SimpleNamespace(
        _get_supervisor_ref=AsyncMock(
            return_value=SimpleNamespace(
                list_cached_models=AsyncMock(return_value=[{"model_name": "legacy"}])
            )
        )
    )
    response = await admin.list_cached_models(api=api, model_name=None, worker_ip=None)
    assert json.loads(response.body) == {
        "list": [{"model_name": "legacy", "model_type": None, "model_engine": None}]
    }


@pytest.fixture
def model_version_catalog(monkeypatch):
    categories = {
        "llm": "get_llm_version_infos",
        "embedding": "get_embedding_model_descriptions",
        "rerank": "get_rerank_model_descriptions",
        "image": "get_image_model_descriptions",
        "audio": "get_audio_model_descriptions",
        "video": "get_video_model_descriptions",
        "world": "get_world_model_descriptions",
        "flexible": "get_flexible_model_descriptions",
    }
    for category, getter in categories.items():
        module = __import__(f"xinference.model.{category}", fromlist=[category])
        monkeypatch.setattr(
            module,
            getter,
            lambda category=category: {
                category: [
                    {
                        "model_version": category,
                        "cache_status": False,
                        "model_file_location": None,
                    }
                ]
            },
        )
    return {
        category: "LLM" if category == "llm" else category for category in categories
    }


@pytest.mark.asyncio
async def test_worker_discovery_records_types_for_every_model_category(
    model_version_catalog,
):
    tracker = SimpleNamespace(record_model_version=AsyncMock())
    await WorkerActor._record_model_versions_best_effort(
        SimpleNamespace(address="worker"), tracker
    )
    versions, address = tracker.record_model_version.call_args.args
    assert address == "worker"
    assert {
        key: value[0]["model_type"] for key, value in versions.items()
    } == model_version_catalog


@pytest.mark.asyncio
async def test_supervisor_initial_discovery_records_every_category(
    monkeypatch, model_version_catalog
):
    import asyncio

    from ... import extensions
    from .. import supervisor as supervisor_module
    from ..cache_tracker import CacheTrackerActor

    tracker = SimpleNamespace(record_model_version=AsyncMock())

    async def create_actor(cls, *args, **kwargs):
        return tracker if cls is CacheTrackerActor else SimpleNamespace()

    monkeypatch.setattr(extensions, "get_extensions", lambda: ())
    monkeypatch.setattr(supervisor_module, "XINFERENCE_DISABLE_HEALTH_CHECK", True)
    monkeypatch.setattr(supervisor_module, "XINFERENCE_ENABLE_OTEL", False)
    monkeypatch.setattr(supervisor_module.xo, "create_actor", create_actor)
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", MagicMock())
    supervisor = SimpleNamespace(
        address="supervisor",
        _initialize_worker_ownership=MagicMock(),
        _schedule_autostart=MagicMock(),
        _relaunch_manager=MagicMock(),
    )
    await supervisor_module.SupervisorActor.__post_create__(supervisor)
    versions, address = tracker.record_model_version.call_args.args
    assert address == "supervisor"
    assert {
        key: value[0]["model_type"] for key, value in versions.items()
    } == model_version_catalog


@pytest.mark.asyncio
async def test_audio_quantized_cached_variants_use_registered_cache_names(monkeypatch):
    model_name = "Irodori-TTS-v4.1-Anime"
    versions = [model_name, model_name + "-INT4-Weight-Only"]
    entries = [
        {**entry("audio", version), "model_name": model_name} for version in versions
    ]
    result, _ = await cached_models(
        monkeypatch, entries, {"PyTorch": [{"model_name": model_name}]}
    )
    assert [item["model_engine"] for item in result] == [["PyTorch"], ["PyTorch"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse_engines", [False, True])
async def test_audio_engine_quantization_mismatch_keeps_other_matches(
    monkeypatch, reverse_engines
):
    from ...model import audio

    # Include MLX variants even when the host catalog filters them out.
    models = {}
    with monkeypatch.context() as catalog_patch:
        catalog_patch.setattr(audio, "_need_filter", lambda spec: False)
        audio.load_model_family_from_json("models", models)

    name = "Irodori-TTS-v4.1-Small"
    monkeypatch.setitem(audio.BUILTIN_AUDIO_MODELS, name, models[name])
    versions = [name + "-INT8-Weight-Only", name + "-MLX", name]
    entries = [{**entry("audio", version), "model_name": name} for version in versions]
    engines = {
        engine: [{"model_name": name}]
        for engine in (["MLX", "PyTorch"] if reverse_engines else ["PyTorch", "MLX"])
    }
    result, worker = await cached_models(monkeypatch, entries, engines)
    assert [item["model_engine"] for item in result] == [
        ["PyTorch"],
        ["MLX"],
        ["PyTorch"],
    ]
    worker.discover_engines.assert_called_once_with(
        "audio", name, enable_virtual_env=worker_module.XINFERENCE_ENABLE_VIRTUAL_ENV
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", ["rerank", "world", "image"])
async def test_unknown_cached_version_does_not_get_an_engine(monkeypatch, model_type):
    result, _ = await cached_models(
        monkeypatch,
        [entry(model_type, "different-model")],
        {"available": [{"model_name": "demo"}]},
    )
    assert result[0]["model_engine"] == []


@pytest.mark.asyncio
async def test_matching_failure_does_not_poison_later_cached_variants(monkeypatch):
    original = worker_module.get_cached_model_engines

    def match(name, model_type, version, params):
        if version == "bad":
            raise RuntimeError("Invalid cached version metadata")
        return original(name, model_type, version, params)

    monkeypatch.setattr(worker_module, "get_cached_model_engines", match)
    result, worker = await cached_models(
        monkeypatch,
        [entry(version="bad"), entry()],
        {
            "Transformers": [
                {
                    "model_name": "demo",
                    "model_format": "pytorch",
                    "model_size_in_billions": 7,
                    "quantizations": ["none"],
                }
            ]
        },
    )
    assert [item["model_engine"] for item in result] == [None, ["Transformers"]]
    worker.discover_engines.assert_called_once()


@pytest.mark.asyncio
async def test_cache_engine_discovery_runs_off_the_actor_loop(monkeypatch):
    import threading

    loop_thread = threading.get_ident()
    threads = []

    def discover(*args, **kwargs):
        threads.append(threading.get_ident())
        return {}

    await cached_models(monkeypatch, [entry()], discover)
    assert len(threads) == 1
    assert threads[0] != loop_thread
