from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xinference._compat import ValidationError
from xinference.core.model_recommendation import (
    ModelRecommendationRequest,
    select_recommendation,
    spec_key,
)


def snapshot(
    address="host:1",
    engine="transformers",
    sizes=(4,),
    device="cuda",
    platform="Linux",
    cached=False,
    installed=True,
    venv=False,
):
    params = [
        {
            "model_format": "pytorch",
            "model_size_in_billions": size,
            "quantizations": ["none"],
        }
        for size in sizes
    ]
    keys = {spec_key(p, "none") for p in params}
    return {
        "model_exists": True,
        "worker_ip": address,
        "platform": platform,
        "device": device,
        "gpu_indices": [0, 1],
        "gpu_count": 2,
        "enable_virtual_env": venv,
        "engines": {engine: params},
        "installed_engines": {engine: params} if installed else {},
        "launch_specs": keys,
        "cached_specs": keys if cached else set(),
    }


def recommend(workers, **constraints):
    return select_recommendation(
        ModelRecommendationRequest(model_name="test", constraints=constraints),
        workers,
        [],
    )


@pytest.mark.parametrize(
    "constraints",
    [
        {"model_size_in_billions": True},
        {"model_size_in_billions": 1.7},
        {"model_size_in_billions": "nan"},
        {"model_size_in_billions": 0},
        {"worker_ip": ""},
        {"enable_virtual_env": "false"},
        {"enable_virtual_env": None},
        {"n_gpu": 0},
        {"n_gpu": True},
        {"n_gpu": "2"},
        {"gpu_idx": []},
        {"gpu_idx": [-1]},
        {"gpu_idx": [0, 0]},
        {"gpu_idx": [False]},
        {"other": 1},
    ],
)
def test_strict_constraints(constraints):
    with pytest.raises(ValidationError):
        ModelRecommendationRequest(model_name="test", constraints=constraints)


def test_smallest_size_not_registry_order_and_hard_size():
    workers = [snapshot(sizes=(397, 122, 35, "1_7", "0_6"))]
    assert recommend(workers)["config"]["model_size_in_billions"] == "0_6"
    assert (
        recommend(workers, model_size_in_billions="1.7")["config"][
            "model_size_in_billions"
        ]
        == "1_7"
    )
    assert recommend(workers, model_size_in_billions=7)["config"] is None


def test_heterogeneous_workers_do_not_mix_engines_cache_or_platform():
    workers = [
        snapshot("linux:1", "vLLM"),
        snapshot("mac:1", "MLX", device="mps", platform="Darwin", cached=True),
    ]
    result = recommend(workers)
    assert result["config"]["worker_ip"] == "mac:1"
    assert result["config"]["model_engine"] == "MLX"
    assert recommend(list(reversed(workers))) == result
    # A cache entry for a different quantization must not bias selection.
    workers[1]["cached_specs"] = {("pytorch", 4, "int4")}
    assert not any(
        r["code"] == "exact_spec_cached" for r in recommend(workers)["reasons"]
    )


def test_platform_order_and_readiness():
    linux = snapshot(engine="transformers")
    linux["engines"]["vLLM"] = linux["engines"]["transformers"]
    linux["installed_engines"]["vLLM"] = linux["engines"]["vLLM"]
    assert recommend([linux])["config"]["model_engine"] == "vLLM"
    venv = snapshot(engine="vLLM", installed=False, venv=True)
    result = recommend([venv])
    assert any(r["code"] == "virtual_env_candidate" for r in result["reasons"])
    assert any(r["code"] == "virtual_env_not_verified" for r in result["warnings"])
    venv["enable_virtual_env"] = False
    assert recommend([venv])["config"] is None


def test_gpu_constraints_and_null_are_preserved():
    worker = snapshot()
    assert "n_gpu" not in recommend([worker])["config"]
    assert recommend([worker], n_gpu=None)["config"]["n_gpu"] is None
    assert recommend([worker], n_gpu=3)["config"] is None
    assert recommend([worker], n_gpu=99, gpu_idx=[0])["config"] is None
    assert recommend([worker], gpu_idx=[2])["config"] is None
    result = recommend([worker], n_gpu=None, gpu_idx=[1])
    assert result["config"]["gpu_idx"] == [1]
    assert result["config"]["n_gpu"] is None
    assert any(w["code"] == "gpu_idx_overrides_n_gpu" for w in result["warnings"])
    assert any(w["code"] == "memory_not_verified" for w in result["warnings"])


@pytest.mark.parametrize("device,cpu", [("cuda", True), ("mps", False)])
@pytest.mark.parametrize("model_format", ["awq", "gptq", "bnb", "fp4"])
def test_cpu_and_mps_exclude_cuda_transformer_quantization(device, cpu, model_format):
    worker = snapshot(device=device)
    param = worker["engines"]["transformers"][0]
    param["model_format"] = model_format
    param["quantizations"] = ["int4"]
    worker["launch_specs"] = {spec_key(param, "int4")}
    assert recommend([worker], **({"n_gpu": None} if cpu else {}))["config"] is None


def test_zero_allocatable_devices_and_other_device_engine_fallback():
    worker = snapshot(engine="vLLM")
    worker["gpu_indices"] = []
    assert recommend([worker])["config"] is None
    worker = snapshot()
    worker["gpu_indices"] = []
    assert recommend([worker])["config"] is None
    assert recommend([worker], n_gpu=None)["config"]["n_gpu"] is None
    worker = snapshot(engine="MindSpore", device="npu")
    assert recommend([worker])["config"]["model_engine"] == "MindSpore"
    assert recommend([snapshot(engine="MLX", device="cuda")])["config"] is None


def test_mps_auto_with_zero_gpu_indices_preserves_mlx():
    worker = snapshot(engine="MLX", device="mps", platform="Darwin")
    worker["gpu_indices"] = []
    worker["gpu_count"] = 0
    worker["engines"]["llama.cpp"] = worker["engines"]["MLX"]
    worker["installed_engines"]["llama.cpp"] = worker["engines"]["MLX"]
    assert recommend([worker])["config"]["model_engine"] == "MLX"
    assert recommend([worker], n_gpu=1)["config"] is None
    assert recommend([worker], n_gpu=None)["config"] is None


def test_cache_is_tiebreak_after_engine_and_quantization():
    preferred = snapshot("a:1", "vLLM")
    cached = snapshot("b:1", "transformers", cached=True)
    assert recommend([cached, preferred])["config"]["worker_ip"] == "a:1"
    worker = snapshot(engine="llama.cpp")
    param = worker["engines"]["llama.cpp"][0]
    param["model_format"] = "ggufv2"
    param["quantizations"] = ["Q2_K", "Q4_K_M"]
    worker["launch_specs"] = {spec_key(param, q) for q in param["quantizations"]}
    worker["cached_specs"] = {spec_key(param, "Q2_K")}
    assert recommend([worker])["config"]["quantization"] == "Q4_K_M"


@pytest.mark.asyncio
async def test_supervisor_no_workers_is_not_false_404():
    from xinference.core.supervisor import SupervisorActor

    supervisor = SimpleNamespace(_worker_address_to_worker={})
    result = await SupervisorActor.recommend_model(supervisor, {"model_name": "test"})
    assert result["status"] == "no_recommendation"
    assert result["warnings"][0]["code"] == "no_workers"


@pytest.mark.asyncio
async def test_supervisor_failures_missing_model_and_worker_constraint():
    from xinference.core.supervisor import SupervisorActor

    good = AsyncMock()
    good.get_model_recommendation_info.return_value = snapshot()
    bad = AsyncMock()
    bad.get_model_recommendation_info.side_effect = RuntimeError("worker offline")
    supervisor = SimpleNamespace(
        _worker_address_to_worker={"host:1": good, "other:1": bad},
        _get_worker_host=SupervisorActor._get_worker_host,
    )
    request = {
        "model_name": "test",
        "constraints": {"worker_ip": "host", "enable_virtual_env": False},
    }
    result = await SupervisorActor.recommend_model(supervisor, request)
    assert result["config"]["worker_ip"] == "host:1"
    assert result["warnings"][0]["code"] == "worker_discovery_failed"
    good.get_model_recommendation_info.assert_awaited_once_with("test", False)
    request["constraints"]["worker_ip"] = "missing"
    assert (await SupervisorActor.recommend_model(supervisor, request))[
        "config"
    ] is None
    good.get_model_recommendation_info.return_value = {"model_exists": False}
    assert (await SupervisorActor.recommend_model(supervisor, request))[
        "status"
    ] == "no_recommendation"
    del supervisor._worker_address_to_worker["other:1"]
    with pytest.raises(LookupError):
        await SupervisorActor.recommend_model(supervisor, request)


@pytest.mark.asyncio
async def test_worker_read_only_default_hub_and_exact_cache(monkeypatch, tmp_path):
    from xinference.core.worker import WorkerActor
    from xinference.model.llm import llm_family
    from xinference.model.llm.cache_manager import LLMCacheManager

    data = snapshot()
    worker = SimpleNamespace(
        _total_gpu_devices=[0],
        get_model_registration=AsyncMock(return_value=object()),
        query_engines_by_model_name=AsyncMock(return_value=data["engines"]),
    )
    spec = SimpleNamespace(
        model_format="pytorch", model_size_in_billions=4, quantization="none"
    )
    calls = []

    def match(*args):
        calls.append(args)
        return SimpleNamespace(model_specs=[spec])

    monkeypatch.setattr(llm_family, "match_llm", match)
    monkeypatch.setattr("xinference.constants.XINFERENCE_CACHE_DIR", str(tmp_path))
    path = LLMCacheManager.get_cache_dir_for_spec("test", spec)

    def no_mkdir(*args, **kwargs):
        raise AssertionError("recommendation must not create directories")

    with monkeypatch.context() as readonly:
        readonly.setattr("os.makedirs", no_mkdir)
        result = await WorkerActor.get_model_recommendation_info(worker, "test", False)
    assert not result["cached_specs"]
    assert calls == [("test", "pytorch", 4, "none")]
    from pathlib import Path

    Path(path).mkdir(parents=True)
    result = await WorkerActor.get_model_recommendation_info(worker, "test", False)
    assert result["cached_specs"] == data["launch_specs"]
    monkeypatch.setattr(llm_family, "match_llm", lambda *args: None)
    result = await WorkerActor.get_model_recommendation_info(worker, "test", False)
    assert not result["launch_specs"] and not result["cached_specs"]
