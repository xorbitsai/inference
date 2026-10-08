import importlib.util
from pathlib import Path
from unittest.mock import Mock

import psutil
import pytest

spec = importlib.util.spec_from_file_location(
    "benchmark_pd", Path(__file__).parents[1] / "benchmark_pd.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_goodput_excludes_errors_and_missing_token_timing():
    records = [
        {"ttft_s": 0.1, "tpot_s": 0.02, "latency_s": 0.2, "output_tokens": 6},
        {"ttft_s": 3, "tpot_s": 0.01, "latency_s": 4, "output_tokens": 101},
        {"ttft_s": 0.1, "tpot_s": None, "latency_s": 0.1, "output_tokens": 1},
        {"error": "timeout", "latency_s": 5},
    ]
    result = module.summarize(records, 10, 2, 0.05)
    assert result["successful"] == 3
    assert result["slo_eligible_requests"] == 2
    assert result["goodput_req_s"] == 0.1
    assert result["output_tokens_s"] == 10.8
    assert result["ttft_s"]["p99"] == 3


@pytest.mark.parametrize("stage", ["parent", "children", "memory"])
@pytest.mark.parametrize("error", [psutil.NoSuchProcess, psutil.AccessDenied])
def test_rss_sampling_recovers_from_process_races(monkeypatch, stage, error):
    child = Mock()
    child.memory_info.return_value.rss = 20
    parent = Mock()
    parent.memory_info.return_value.rss = 100
    parent.children.return_value = [child]
    lookup = Mock(return_value=parent)
    monkeypatch.setattr(psutil, "Process", lookup)
    target = {
        "parent": lookup,
        "children": parent.children,
        "memory": child.memory_info,
    }[stage]
    target.side_effect = error(123)
    assert module.process_tree_rss(123) is None
    target.side_effect = None
    assert module.process_tree_rss(123) == 120


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("nvidia-smi missing"),
        module.subprocess.TimeoutExpired("nvidia-smi", 10),
    ],
)
async def test_gpu_sampler_records_failure_and_continues(monkeypatch, error):
    import asyncio
    from types import SimpleNamespace

    attempts = []

    def run(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise error
        return SimpleNamespace(stdout="gpu data", stderr="", returncode=0)

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def __aiter__(self):
            yield SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2), choices=[]
            )

    async def create(**kwargs):
        while len(attempts) < 2:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.01)
        return Stream()

    class Client:
        chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(module, "AsyncOpenAI", lambda **kwargs: Client())
    monkeypatch.setattr(module.subprocess, "run", run)
    records, _, samples = await asyncio.wait_for(
        module.measure("http://unused", "m", [{"messages": []}], 1, 1, True), 5
    )
    assert "error" not in records[0]
    assert str(error) == samples[0]["error"]
    assert samples[1]["gpu_csv"] == "gpu data"
    assert samples[1]["error"] == ""


def test_native_sglang_benchmark_does_not_launch_or_terminate_models(
    monkeypatch, tmp_path
):
    import json
    import sys
    from unittest.mock import AsyncMock

    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"replica_config": [{}, {}]}))
    workload = tmp_path / "workload.jsonl"
    workload.write_text(json.dumps({"messages": [{"role": "user", "content": "test"}]}))
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_pd.py",
            "--endpoint",
            "http://xinference",
            "--launch",
            str(launch),
            "--workload",
            str(workload),
            "--output",
            str(output),
            "--modes",
            "sglang-native",
            "--native-sglang-endpoint",
            "http://native",
            "--native-sglang-model",
            "native-model",
            "--concurrency",
            "1",
        ],
    )
    client = Mock()
    monkeypatch.setattr(module, "Client", lambda endpoint: client)
    record = {"ttft_s": 0.1, "tpot_s": 0.01, "latency_s": 0.2, "output_tokens": 2}
    measure = AsyncMock(return_value=([record], 0.2, []))
    monkeypatch.setattr(module, "measure", measure)
    module.main()
    assert not client.mock_calls
    assert all(
        call.args[:2] == ("http://native", "native-model")
        for call in measure.call_args_list
    )
    report = json.loads(output.read_text())
    assert report["runs"][0]["model_uid"] == "native-model"
    assert report["runs"][0]["summary"]["successful"] == 1


@pytest.mark.asyncio
async def test_completion_workload_uses_completion_stream_and_token_usage(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def __aiter__(self):
            yield SimpleNamespace(usage=None, choices=[SimpleNamespace(text="first")])
            yield SimpleNamespace(usage=None, choices=[SimpleNamespace(text="second")])
            yield SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2), choices=[]
            )

    class Client:
        completions = SimpleNamespace(create=AsyncMock(return_value=Stream()))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    client = Client()
    monkeypatch.setattr(module, "AsyncOpenAI", lambda **kwargs: client)
    records, _, _ = await module.measure(
        "http://unused", "model", [{"prompt": "exact prompt"}], 1, 1
    )
    assert client.completions.create.call_args.kwargs["prompt"] == "exact prompt"
    assert records[0]["text"] == "firstsecond"
    assert records[0]["input_tokens"] == 4 and records[0]["output_tokens"] == 2
    assert records[0]["ttft_s"] >= 0 and records[0]["tpot_s"] >= 0
