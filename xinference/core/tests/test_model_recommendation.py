# CI: linux

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
        "model_context_lengths": dict.fromkeys(keys, 32768),
    }


def recommend(workers, **constraints):
    return select_recommendation(
        ModelRecommendationRequest(model_name="test", constraints=constraints),
        workers,
        [],
    )


def memory_worker(device="cuda"):
    worker = snapshot(sizes=(4, 8, 14), device=device)
    worker["memory"] = {
        "host_available_mib": 20000,
        "gpu_available_mib": {"gpu-0": 20000, "gpu-1": 20000},
    }
    worker["memory_estimates"] = {
        spec_key(p, "none"): size * 1500
        for size, p in zip((4, 8, 14), worker["engines"]["transformers"])
    }
    return worker


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps"])
def test_largest_estimated_fit(device):
    worker = memory_worker(device)
    result = recommend([worker], **({"n_gpu": None} if device == "cpu" else {}))
    assert result["config"]["model_size_in_billions"] == 8
    assert any(r["code"] == "memory_estimate" for r in result["reasons"])
    assert result["config"].get("gpu_idx") is None
    assert (
        recommend([worker], model_size_in_billions=4)["config"][
            "model_size_in_billions"
        ]
        == 4
    )
    assert recommend([worker], model_size_in_billions=14)["config"] is None


def test_engine_specific_estimates_are_not_shared():
    worker = memory_worker()
    worker["engines"]["vLLM"] = worker["engines"]["transformers"]
    worker["installed_engines"]["vLLM"] = worker["engines"]["vLLM"]
    worker["engine_memory_estimates"] = {
        ("vLLM", spec_key(param, "none")): 9000 for param in worker["engines"]["vLLM"]
    }
    result = recommend([worker])
    assert result["config"]["model_engine"] == "vLLM"
    assert result["config"]["model_size_in_billions"] == 14


@pytest.mark.parametrize("engine,fraction", [("vLLM", 0.90)])
def test_default_engine_reservation_must_fit_free_memory(engine, fraction):
    worker = memory_worker()
    worker["engines"][engine] = worker["engines"]["transformers"]
    worker["installed_engines"][engine] = worker["engines"][engine]
    worker["memory"]["gpu_total_mib"] = {"gpu-0": 24000, "gpu-1": 24000}
    result = recommend([worker])
    assert result["config"]["model_engine"] == "transformers"
    assert any(w["code"] == "vllm_reservation_exceeds_free" for w in result["warnings"])
    worker["memory"]["gpu_available_mib"] = {"gpu-0": 24000, "gpu-1": 24000}
    assert recommend([worker])["config"]["model_engine"] == engine
    worker["memory"]["gpu_available_mib"]["gpu-1"] = 24000 * fraction - 1
    assert recommend([worker])["config"]["model_engine"] == "transformers"
    assert recommend([worker], gpu_idx=[0])["config"]["model_engine"] == engine


@pytest.mark.parametrize(
    "constraints",
    [
        {"context_length": 0},
        {"concurrency": True},
        {"vision_tokens": -1},
        {"prefix_cache_bytes": "1000"},
    ],
)
def test_memory_workload_constraints_are_strict(constraints):
    with pytest.raises(ValidationError):
        ModelRecommendationRequest(model_name="test", constraints=constraints)


def test_non_llm_rejects_memory_workload_constraints():
    with pytest.raises(ValidationError, match="only supported for LLM"):
        ModelRecommendationRequest(
            model_name="test", model_type="audio", constraints={"context_length": 4096}
        )


@pytest.mark.asyncio
async def test_supervisor_passes_memory_workload_to_worker():
    from xinference.core.supervisor import SupervisorActor

    worker = SimpleNamespace(
        get_model_recommendation_info=AsyncMock(return_value=snapshot())
    )
    supervisor = SimpleNamespace(_worker_address_to_worker={"test:1": worker})
    await SupervisorActor.recommend_model(
        supervisor,
        {
            "model_name": "test",
            "constraints": {"context_length": 8192, "concurrency": 2},
        },
    )
    options = worker.get_model_recommendation_info.call_args.kwargs["memory_options"]
    assert options == {
        "context_length": 8192,
        "concurrency": 2,
        "vision_tokens": 0,
        "vision_tokens_per_image": 0,
        "prefix_cache_bytes": 0,
    }


def test_memory_budget_does_not_sum_gpus_or_change_placement():
    worker = memory_worker()
    worker["memory"]["gpu_available_mib"]["gpu-1"] = 8000
    assert recommend([worker])["config"]["model_size_in_billions"] == 4
    result = recommend([worker], gpu_idx=[0])
    assert result["config"]["model_size_in_billions"] == 8
    assert result["config"]["gpu_idx"] == [0]
    for constraints in ({"n_gpu": 2}, {"gpu_idx": [0, 1]}):
        result = recommend([worker], **constraints)
        assert result["config"]["model_size_in_billions"] == 4
        assert not any(r["code"] == "memory_estimate" for r in result["reasons"])


def test_memory_missing_zero_and_all_too_large():
    worker = memory_worker()
    worker["memory"]["gpu_available_mib"]["gpu-0"] = 0
    assert recommend([worker])["config"] is None
    del worker["memory"]["gpu_available_mib"]["gpu-0"]
    assert recommend([worker])["config"]["model_size_in_billions"] == 4
    worker = memory_worker()
    worker["memory_estimates"].clear()
    assert recommend([worker])["config"]["model_size_in_billions"] == 4


def test_estimated_fit_beats_unknown_and_respects_worker_memory():
    unknown = snapshot(address="unknown:1", sizes=(1,))
    measured = memory_worker()
    assert recommend([unknown, measured])["config"]["model_size_in_billions"] == 8
    larger = memory_worker()
    larger["worker_ip"] = "large:1"
    larger["memory"]["gpu_available_mib"] = {"gpu-0": 40000, "gpu-1": 40000}
    assert recommend([measured, larger])["config"]["worker_ip"] == "large:1"


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


@pytest.mark.parametrize("reverse", [False, True])
def test_cached_worker_wins_before_lexical_tie(reverse):
    workers = [snapshot("aa:1"), snapshot("zz:1", cached=True)]
    if reverse:
        workers.reverse()
    result = recommend(workers)
    assert result["config"]["worker_ip"] == "zz:1"
    assert "exact_spec_cached" in {reason["code"] for reason in result["reasons"]}


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
    good.get_model_recommendation_info.assert_awaited_once_with(
        "test", False, model_type="LLM"
    )
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
        return SimpleNamespace(model_specs=[spec], context_length=32768)

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
    from xinference.model.llm.model_metadata import ModelMetadata

    spec.model_metadata = ModelMetadata(
        vocab_size=32000,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
    )
    monkeypatch.setattr(llm_family, "cache_model_config", no_mkdir)
    result = await WorkerActor.get_model_recommendation_info(worker, "test", False)
    assert result["cached_specs"] == data["launch_specs"]
    assert set(result["memory_estimates"]) == data["launch_specs"]
    assert all(value > 0 for value in result["memory_estimates"].values())
    # Unknown quantization must not make otherwise compatible discovery fail.
    worker.query_engines_by_model_name.return_value = {
        "transformers": [
            {
                "model_format": "pytorch",
                "model_size_in_billions": 4,
                "quantizations": ["unsupported"],
            }
        ]
    }
    result = await WorkerActor.get_model_recommendation_info(worker, "test", False)
    assert not result["memory_estimates"]
    assert result["launch_specs"]
    monkeypatch.setattr(llm_family, "match_llm", lambda *args: None)
    result = await WorkerActor.get_model_recommendation_info(worker, "test", False)
    assert not result["launch_specs"] and not result["cached_specs"]


def test_memory_snapshot_failure_and_units(monkeypatch):
    from xinference.core.model_recommendation import recommendation_memory_snapshot

    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: SimpleNamespace(available=1024**3)
    )
    monkeypatch.setattr(
        "xinference.device_utils.get_gpu_info",
        lambda: {"gpu-0": {"free": 2 * 1024**3, "free_memory_mib": 2048}},
    )
    assert recommendation_memory_snapshot() == {
        "host_available_mib": 1024,
        "gpu_available_mib": {"gpu-0": 2048},
    }
    monkeypatch.setattr(
        "xinference.device_utils.get_gpu_info",
        lambda: {"gpu-0": {"free_memory_mib": 2048, "total_memory_mib": 4096}},
    )
    assert recommendation_memory_snapshot()["gpu_total_mib"] == {"gpu-0": 4096}

    def failed():
        raise RuntimeError("monitor unavailable")

    monkeypatch.setattr("xinference.device_utils.get_gpu_info", failed)
    assert recommendation_memory_snapshot() == {"host_available_mib": 1024}


@pytest.mark.parametrize("backend", ["_get_info_by_torch", "_get_rocm_gpu_mem_info"])
def test_unknown_backend_memory_keeps_smallest_fallback(monkeypatch, backend):
    from xinference import device_utils
    from xinference.core.model_recommendation import recommendation_memory_snapshot

    monkeypatch.setattr(
        device_utils,
        "get_gpu_info",
        lambda: {"gpu-0": getattr(device_utils, backend)(0)},
    )
    worker = memory_worker()
    worker["gpu_indices"] = [0]
    worker["memory"] = recommendation_memory_snapshot()
    assert worker["memory"]["gpu_available_mib"] == {}
    result = recommend([worker])
    assert result["config"]["model_size_in_billions"] == 4
    assert not any(r["code"] == "memory_estimate" for r in result["reasons"])


@pytest.mark.parametrize("visible", [False, True])
def test_ascend_memory_is_already_mib(monkeypatch, visible):
    from xinference.core.model_recommendation import recommendation_memory_snapshot
    from xinference.device_utils import get_npu_info

    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    if visible:
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: SimpleNamespace(
            stdout=("| 0 Ascend | OK | info |\n" "| 0 | info | 1000 / 21000 |\n")
        ),
    )
    monkeypatch.setattr("xinference.device_utils.get_gpu_info", get_npu_info)
    worker = memory_worker()
    worker["gpu_indices"] = [0]
    worker["memory"] = recommendation_memory_snapshot()
    assert worker["memory"]["gpu_available_mib"] == {"gpu-0": 20000}
    assert recommend([worker])["config"]["model_size_in_billions"] == 8


@pytest.mark.asyncio
@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
async def test_qwen3_mlx_recommendation_uses_all_sizes_offline(monkeypatch, hub):
    from xinference.core.worker import WorkerActor
    from xinference.model.llm import llm_family

    def no_download(*args, **kwargs):
        raise AssertionError("Recommendation must use bundled metadata")

    monkeypatch.setattr(llm_family, "cache_model_config", no_download)
    monkeypatch.setattr(
        llm_family, "download_from_modelscope", lambda: hub == "modelscope"
    )
    monkeypatch.setattr(llm_family, "download_from_openmind_hub", lambda: False)
    monkeypatch.setattr(llm_family, "download_from_csghub", lambda: False)
    monkeypatch.setattr("xinference.device_utils.get_available_device", lambda: "mps")
    monkeypatch.setattr("xinference.core.worker.gpu_count", lambda: 0)
    monkeypatch.setattr(
        "xinference.core.model_recommendation.recommendation_memory_snapshot",
        lambda: {"host_available_mib": 20 * 1024},
    )
    engines = {
        "MLX": [
            dict(
                model_format="mlx", model_size_in_billions=size, quantizations=["4bit"]
            )
            for size in ("0_6", "1_7", 4, 8, 14, 32)
        ]
    }
    worker = SimpleNamespace(
        _total_gpu_devices=[],
        get_model_registration=AsyncMock(return_value=object()),
        query_engines_by_model_name=AsyncMock(return_value=engines),
    )
    info = await WorkerActor.get_model_recommendation_info(
        worker, "qwen3", False, memory_options={"context_length": 2048}
    )
    assert len(info["memory_estimates"]) == 6
    info["worker_ip"] = "test:1"
    request = ModelRecommendationRequest(
        model_name="qwen3", constraints={"context_length": 2048}
    )
    result = select_recommendation(request, [info], [])
    assert result["config"]["model_size_in_billions"] == 14
    assert result["config"]["quantization"] == "4bit"
    assert any(reason["code"] == "memory_fit_size" for reason in result["reasons"])


@pytest.mark.asyncio
@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
@pytest.mark.parametrize("context_length", [2048, None])
@pytest.mark.parametrize(
    "engine,model_format,quantization,sizes,device,free,expected",
    [
        ("MLX", "mlx", "4bit", ("0_8", 4, 9, 27, 35), "mps", 20, 9),
        ("MLX", "mlx", "4bit", ("0_8", 4, 9, 27, 35), "mps", 48, 35),
        ("Transformers", "pytorch", "none", ("0_8", 4, 9, 27, 35), "cuda", 32, 9),
        ("vLLM", "gptq", "Int4", (27, 35), "cuda", 64, 27),
        ("vLLM", "awq", "Int4", (9, 27), "cuda", 32, 27),
        ("vLLM", "gptq", "Int4", (27, 35), "npu", 64, 27),
        ("SGLang", "fp8", "FP8", (27, 35), "cuda", 64, 35),
        ("llama.cpp", "ggufv2", "Q4_K_M", ("0_8", 4, 9, 27, 35), "cuda", 32, 35),
    ],
)
async def test_qwen35_recommendation_with_bundled_hybrid_metadata(
    monkeypatch,
    hub,
    context_length,
    engine,
    model_format,
    quantization,
    sizes,
    device,
    free,
    expected,
):
    from xinference.core.worker import WorkerActor
    from xinference.model.llm import llm_family

    def no_network(*args, **kwargs):
        raise AssertionError("Recommendation must remain offline")

    monkeypatch.setattr(llm_family, "cache_model_config", no_network)
    monkeypatch.setattr("requests.get", no_network)
    monkeypatch.setattr("requests.head", no_network)
    monkeypatch.setattr(
        llm_family, "download_from_modelscope", lambda: hub == "modelscope"
    )
    monkeypatch.setattr(llm_family, "download_from_openmind_hub", lambda: False)
    monkeypatch.setattr(llm_family, "download_from_csghub", lambda: False)
    monkeypatch.setattr("xinference.device_utils.get_available_device", lambda: device)
    monkeypatch.setattr(
        "xinference.core.worker.gpu_count", lambda: 0 if device == "mps" else 1
    )
    monkeypatch.setattr(
        "xinference.core.worker.platform.system",
        lambda: "Linux" if device == "cuda" else "Darwin",
    )
    monkeypatch.setattr(
        "xinference.core.model_recommendation.recommendation_memory_snapshot",
        lambda: {
            "host_available_mib": free * 1024,
            "gpu_available_mib": {"gpu-0": free * 1024},
            "gpu_total_mib": {"gpu-0": free * 1024},
        },
    )
    engines = {
        engine: [
            dict(
                model_format=model_format,
                model_size_in_billions=size,
                quantizations=[quantization],
            )
            for size in sizes
        ]
    }
    worker = SimpleNamespace(
        _total_gpu_devices=[] if device == "mps" else [0],
        get_model_registration=AsyncMock(return_value=object()),
        query_engines_by_model_name=AsyncMock(return_value=engines),
    )
    # Explicit short-context cases retain coverage of the caller's override.
    info = await WorkerActor.get_model_recommendation_info(
        worker,
        "qwen3.5",
        False,
        memory_options={"context_length": 2048} if context_length else None,
    )
    assert len(info["engine_memory_estimates"]) == len(sizes) - (
        1 if model_format == "gptq" else 0
    )
    if model_format == "mlx" and context_length:
        workload = await WorkerActor.get_model_recommendation_info(
            worker,
            "qwen3.5",
            False,
            memory_options={
                "context_length": 8192,
                "concurrency": 2,
                "vision_tokens": 128,
                "vision_tokens_per_image": 64,
                "prefix_cache_bytes": 1024**3,
            },
        )
        assert all(
            workload["engine_memory_estimates"][key] > value + 1024
            for key, value in info["engine_memory_estimates"].items()
        )
    if device == "npu":
        long = await WorkerActor.get_model_recommendation_info(
            worker, "qwen3.5", False, memory_options={"context_length": 32768}
        )
        assert min(long["engine_memory_estimates"].values()) > 64 * 1024
    info["worker_ip"] = "test:1"
    result = select_recommendation(
        ModelRecommendationRequest(
            model_name="qwen3.5",
            constraints={"context_length": context_length} if context_length else {},
        ),
        [info],
        [],
    )
    if context_length is None:
        if result["status"] == "recommended":
            estimates = [
                reason
                for reason in result["reasons"]
                if reason["code"] == "memory_estimate"
            ]
            assert all(
                "262144 text tokens" in reason["message"] for reason in estimates
            )
            assert result["config"]["context_length"] == 262144
            if engine == "vLLM" and model_format == "awq" and free == 32:
                assert result["config"]["model_size_in_billions"] == 9
            if engine == "llama.cpp":
                assert result["config"]["n_parallel"] == 1
        else:
            assert all(
                value > free * 1024 * 0.8
                for value in info["engine_memory_estimates"].values()
            )
        return
    assert result["config"]["model_engine"] == engine
    assert result["config"]["model_size_in_billions"] == expected
    assert result["config"]["context_length"] == 2048
    if engine == "llama.cpp":
        assert result["config"]["n_parallel"] == 1
    assert any(reason["code"] == "memory_estimate" for reason in result["reasons"])


@pytest.mark.parametrize(
    "engine,expected",
    [
        ("vLLM", {"context_length": 8320}),
        ("SGLang", {"context_length": 8320}),
        ("llama.cpp", {"context_length": 8320, "n_parallel": 2}),
    ],
)
def test_recommendation_launch_context_matches_workload(engine, expected):
    from ...model.llm.context_length import map_context_length

    result = recommend(
        [snapshot(engine=engine)], context_length=8192, vision_tokens=128, concurrency=2
    )
    for key, value in expected.items():
        assert result["config"][key] == value
    mapped = map_context_length(engine, result["config"])
    native_key = {
        "vLLM": "max_model_len",
        "SGLang": "context_length",
        "llama.cpp": "n_ctx",
    }[engine]
    assert mapped[native_key] == (16640 if engine == "llama.cpp" else 8320)


@pytest.mark.parametrize("vision_tokens", [0, 128])
def test_default_context_uses_registered_limit_including_images(vision_tokens):
    worker = memory_worker()
    worker["engines"] = {"vLLM": worker["engines"]["transformers"]}
    worker["installed_engines"] = worker["engines"]
    result = recommend([worker], vision_tokens=vision_tokens)
    assert result["config"]["context_length"] == 32768
    reason = next(r for r in result["reasons"] if r["code"] == "memory_estimate")
    assert f"{32768 - vision_tokens} text tokens" in reason["message"]


def test_old_fixed_context_estimates_are_unverified_by_default():
    worker = memory_worker()
    worker.pop("model_context_lengths")
    result = recommend([worker])
    assert result["config"]["model_size_in_billions"] == 4
    assert not any(r["code"] == "memory_estimate" for r in result["reasons"])


@pytest.mark.asyncio
@pytest.mark.parametrize("context_length", [2048, 8192])
async def test_supervisor_retries_older_workers_without_trusting_estimates(
    context_length,
):
    from xinference.core.supervisor import SupervisorActor

    old = memory_worker()

    async def discover(model_name, enable_virtual_env, *, model_type):
        return old

    worker = SimpleNamespace(
        get_model_recommendation_info=AsyncMock(side_effect=discover)
    )
    supervisor = SimpleNamespace(_worker_address_to_worker={"test:1": worker})
    result = await SupervisorActor.recommend_model(
        supervisor,
        {"model_name": "test", "constraints": {"context_length": context_length}},
    )
    assert worker.get_model_recommendation_info.await_count == 2
    assert worker.get_model_recommendation_info.call_args.kwargs == {
        "model_type": "LLM"
    }
    assert result["config"]["model_size_in_billions"] == 4
    assert not any(r["code"] == "memory_estimate" for r in result["reasons"])
    assert any(
        w["code"] == "worker_memory_options_unsupported" for w in result["warnings"]
    )
    assert old["memory_estimates"]


@pytest.mark.asyncio
async def test_supervisor_does_not_retry_internal_type_errors():
    from xinference.core.supervisor import SupervisorActor

    worker = SimpleNamespace(
        get_model_recommendation_info=AsyncMock(
            side_effect=TypeError("invalid memory_options value")
        )
    )
    result = await SupervisorActor.recommend_model(
        SimpleNamespace(_worker_address_to_worker={"test:1": worker}),
        {"model_name": "test", "constraints": {"context_length": 8192}},
    )
    assert worker.get_model_recommendation_info.await_count == 1
    assert result["status"] == "no_recommendation"
    assert any(w["code"] == "worker_discovery_failed" for w in result["warnings"])


def test_sglang_uses_available_memory_and_zero_budget_is_not_a_sentinel():
    worker = memory_worker()
    worker["engines"] = {"SGLang": worker["engines"]["transformers"]}
    worker["installed_engines"] = worker["engines"]
    worker["memory"]["gpu_total_mib"] = {"gpu-0": 80000, "gpu-1": 80000}
    assert recommend([worker])["config"]["model_engine"] == "SGLang"
    worker["memory"]["gpu_available_mib"] = {"gpu-0": 0, "gpu-1": 0}
    worker["memory_estimates"] = {}
    assert recommend([worker])["config"]["model_size_in_billions"] == 4


@pytest.mark.parametrize(
    "engine,device,budget,total",
    [
        ("vLLM", "cuda", 1000, None),
        ("vLLM", "cuda", 1000, float("nan")),
        ("vLLM", "mps", 1000, 24000),
        ("vLLM", "cuda", None, 24000),
        ("SGLang", "cuda", 1000, 24000),
    ],
)
def test_reservation_unknown_and_non_cuda_fallbacks(engine, device, budget, total):
    from xinference.core.model_recommendation import (
        RecommendationConstraints,
        recommendation_reservation_fits,
    )

    worker = memory_worker()
    worker["memory"]["gpu_total_mib"] = {"gpu-0": total, "gpu-1": total}
    assert recommendation_reservation_fits(
        worker, RecommendationConstraints(), device, engine, budget
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_length", 2**31),
        ("concurrency", 65536),
        ("vision_tokens", 2**31),
        ("vision_tokens_per_image", 2**31),
        ("prefix_cache_bytes", 10**400),
    ],
)
def test_unbounded_memory_workload_is_rejected(field, value):
    with pytest.raises(ValidationError):
        ModelRecommendationRequest(model_name="test", constraints={field: value})


@pytest.mark.asyncio
async def test_supervisor_default_request_omits_workload_kwargs():
    from xinference.core.supervisor import SupervisorActor

    worker = SimpleNamespace(
        get_model_recommendation_info=AsyncMock(return_value=snapshot())
    )
    supervisor = SimpleNamespace(_worker_address_to_worker={"test:1": worker})
    await SupervisorActor.recommend_model(supervisor, {"model_name": "test"})
    assert worker.get_model_recommendation_info.call_args.kwargs == {
        "model_type": "LLM"
    }


@pytest.mark.parametrize("model_type", ["embedding", "rerank", "audio"])
def test_non_llm_recommendation_contract(model_type):
    request = ModelRecommendationRequest(model_name="test", model_type=model_type)
    worker = snapshot()
    worker["candidates"] = [
        {
            "config": {
                "model_engine": "sentence_transformers",
                "model_format": "pytorch",
                "quantization": q,
            },
            "installed": True,
        }
        for q in ["int4", "none"]
    ]
    result = select_recommendation(request, [worker], [])
    assert result["status"] == "recommended"
    assert result["config"]["quantization"] == "none"
    assert "model_size_in_billions" not in result["config"]
    assert result["config"]["worker_ip"] == "host:1"
    assert result["warnings"][0]["code"] == "memory_not_verified"
    with pytest.raises(ValidationError):
        ModelRecommendationRequest(
            model_name="test",
            model_type=model_type,
            constraints={"model_size_in_billions": 7},
        )


@pytest.mark.parametrize("model_type", ["embedding", "rerank", "audio"])
def test_non_llm_constraints_and_dependency_readiness(model_type):
    worker = snapshot()
    worker["candidates"] = [
        {"config": {"model_engine": "transformers"}, "installed": False}
    ]
    request = ModelRecommendationRequest(model_name="test", model_type=model_type)
    assert select_recommendation(request, [worker], [])["config"] is None
    worker["enable_virtual_env"] = True
    assert select_recommendation(request, [worker], [])["config"] is not None
    for constraints in [{"gpu_idx": [9]}, {"n_gpu": 3}]:
        request = ModelRecommendationRequest(
            model_name="test", model_type=model_type, constraints=constraints
        )
        assert select_recommendation(request, [worker], [])["config"] is None
    worker["gpu_indices"] = []
    request = ModelRecommendationRequest(model_name="test", model_type=model_type)
    assert select_recommendation(request, [worker], [])["config"] is None


def test_audio_mlx_only_on_mps_and_cpu_excludes_vllm():
    worker = snapshot(device="mps", platform="Darwin")
    worker.update(gpu_count=0, gpu_indices=[])
    worker["candidates"] = [
        {"config": {"model_engine": engine}, "installed": True}
        for engine in ["transformers", "mlx", "vllm"]
    ]
    request = ModelRecommendationRequest(model_name="test", model_type="audio")
    assert (
        select_recommendation(request, [worker], [])["config"]["model_engine"] == "mlx"
    )
    request = ModelRecommendationRequest(
        model_name="test", model_type="audio", constraints={"n_gpu": None}
    )
    assert (
        select_recommendation(request, [worker], [])["config"]["model_engine"]
        == "transformers"
    )


@pytest.mark.parametrize("n_gpu,expected_engine", [("auto", "MLX"), (None, "PyTorch")])
def test_fish_audio_on_apple_silicon(n_gpu, expected_engine):
    from xinference.core.model_recommendation import non_llm_candidates

    # Use the real FishAudio registry and launch matcher, with both backends
    # discovered as ready. Metal hosts report zero allocatable CUDA GPUs.
    engines = {
        "MLX": [{"model_format": "mlx"}],
        "PyTorch": [{"model_format": "pytorch"}],
    }
    worker = snapshot(device="mps", platform="Darwin")
    worker.update(
        gpu_count=0,
        gpu_indices=[],
        candidates=non_llm_candidates("audio", "FishAudio-S2-Pro", engines, engines),
    )
    request = ModelRecommendationRequest(
        model_name="FishAudio-S2-Pro", model_type="audio", constraints={"n_gpu": n_gpu}
    )
    result = select_recommendation(request, [worker], [])
    assert result["config"]["model_engine"] == expected_engine
    assert result["config"]["n_gpu"] == n_gpu


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", ["embedding", "rerank", "audio"])
async def test_non_llm_worker_discovery_is_read_only(monkeypatch, model_type):
    import importlib

    from xinference.core.worker import WorkerActor

    module_name, matcher = {
        "embedding": ("embedding.embed_family", "match_embedding"),
        "rerank": ("rerank.rerank_family", "match_rerank"),
        "audio": ("audio.core", "match_audio"),
    }[model_type]
    module = importlib.import_module("xinference.model." + module_name)
    monkeypatch.setattr(
        module, matcher, lambda *args, **kwargs: SimpleNamespace(quantization=None)
    )
    params = [{"model_format": "pytorch", "quantization": "none"}]
    worker = SimpleNamespace(
        _total_gpu_devices=[0],
        get_model_registration=AsyncMock(return_value=object()),
        query_engines_by_model_name=AsyncMock(
            return_value={"transformers": params, "unavailable": "not installed"}
        ),
    )

    def no_mkdir(*args, **kwargs):
        raise AssertionError("recommendation must not write")

    with monkeypatch.context() as readonly:
        readonly.setattr("os.makedirs", no_mkdir)
        result = await WorkerActor.get_model_recommendation_info(
            worker, "test", False, model_type=model_type
        )
    worker.get_model_registration.assert_awaited_once_with(model_type, "test")
    worker.query_engines_by_model_name.assert_awaited_once_with(
        "test", model_type, False
    )
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["installed"]


def test_non_llm_excludes_unlaunchable_spec(monkeypatch):
    from xinference.core.model_recommendation import non_llm_candidates
    from xinference.model.embedding import embed_family

    def reject(*args):
        raise ValueError("not registered")

    monkeypatch.setattr(embed_family, "match_embedding", reject)
    assert (
        non_llm_candidates(
            "embedding",
            "test",
            {
                "sentence_transformers": [
                    {"model_format": "pytorch", "quantization": "none"}
                ]
            },
            {},
        )
        == []
    )


@pytest.mark.parametrize(
    "engine", ["vLLM", "SGLang", "MLX", "Transformers", "llama.cpp"]
)
def test_recommendation_preserves_unified_context(engine):
    result = recommend(
        [snapshot(engine=engine, device="mps" if engine == "MLX" else "cuda")],
        context_length=32768,
    )
    assert result["config"]["context_length"] == 32768
    if engine == "llama.cpp":
        assert result["config"]["n_parallel"] == 1
    assert not any(key in result["config"] for key in ["max_model_len", "n_ctx"])


@pytest.mark.asyncio
async def test_supervisor_passes_context_to_worker_and_retries_old_worker():
    from xinference.core.supervisor import SupervisorActor

    info = memory_worker()
    worker = SimpleNamespace(get_model_recommendation_info=AsyncMock(return_value=info))
    supervisor = SimpleNamespace(_worker_address_to_worker={"test:1": worker})
    request = {"model_name": "test", "constraints": {"context_length": 32768}}
    result = await SupervisorActor.recommend_model(supervisor, request)
    assert worker.get_model_recommendation_info.call_args.kwargs["memory_options"] == {
        "context_length": 32768,
        "concurrency": 1,
        "vision_tokens": 0,
        "vision_tokens_per_image": 0,
        "prefix_cache_bytes": 0,
    }
    assert result["config"]["context_length"] == 32768
    worker.get_model_recommendation_info.side_effect = [
        TypeError("got an unexpected keyword argument 'memory_options'"),
        info,
    ]
    result = await SupervisorActor.recommend_model(supervisor, request)
    assert result["config"]
    assert not any(r["code"] == "memory_estimate" for r in result["reasons"])
    assert any(
        w["code"] == "worker_memory_options_unsupported" for w in result["warnings"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, True, "32768", 2**31])
async def test_supervisor_rejects_invalid_launch_context_before_placement(value):
    from xinference.core.supervisor import SupervisorActor

    with pytest.raises(ValueError, match="context_length"):
        await SupervisorActor.launch_builtin_model(
            SimpleNamespace(),
            "test",
            "test",
            4,
            "pytorch",
            "none",
            "vLLM",
            "LLM",
            context_length=value,
        )
