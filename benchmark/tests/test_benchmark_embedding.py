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
import importlib
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp import web


@pytest.fixture
def runner_class(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("benchmark_embedding").EmbeddingBenchmarkRunner


@asynccontextmanager
async def embedding_server(handler):
    app = web.Application()
    app.router.add_post("/v1/embeddings", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        yield f"http://127.0.0.1:{port}/v1/embeddings"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 3, 16, 128])
async def test_exact_requests_concurrency_and_connection_reuse(
    runner_class, concurrency
):
    count = 130
    seen = []
    transports = set()
    active = 0
    peak = 0
    first_wave = asyncio.Event()

    async def handle(request):
        nonlocal active, peak
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = await request.json()
        assert payload["model"] == "bge-m3"
        seen.append(payload["input"])
        transports.add(request.transport)
        active += 1
        peak = max(peak, active)
        if len(seen) > 5:
            # Hold the first measured wave until every worker has a connection.
            # This also detects an unintended 100-connection pool cap.
            if active == concurrency:
                first_wave.set()
            await asyncio.wait_for(first_wave.wait(), timeout=5)
        # Keep the last request alive after the other workers finish.
        await asyncio.sleep(0.04 if payload["input"] == str(count - 1) else 0.005)
        active -= 1
        return web.json_response({"data": []})

    async with embedding_server(handle) as url:
        benchmark = runner_class(
            url,
            "bge-m3",
            [{"sentence": str(i)} for i in range(count)],
            False,
            concurrency,
            api_key="test-key",
        )
        await benchmark.run()
        assert active == 0
        assert len(benchmark.outputs) == count
        assert all(output.success for output in benchmark.outputs)
        assert benchmark.left == 0
        assert seen[:5] == [str(i) for i in range(5)]
        assert Counter(seen[5:]) == Counter(str(i) for i in range(count))
        assert peak == concurrency
        assert len(transports) <= concurrency
        assert benchmark._session is None


@pytest.mark.asyncio
async def test_failures_are_counted_and_excluded_from_throughput(runner_class, capsys):
    async def handle(request):
        value = (await request.json())["input"]
        if value == "http-error":
            return web.Response(status=503, text="unavailable")
        if value == "invalid-json":
            return web.Response(text="{", content_type="application/json")
        if value == "disconnect":
            request.transport.close()
            return web.Response()
        return web.json_response({"data": []})

    async with embedding_server(handle) as url:
        benchmark = runner_class(
            url,
            "bge-m3",
            [
                {"sentence": s}
                for s in ["ok", "http-error", "invalid-json", "disconnect"]
            ],
            False,
            8,
        )
        await benchmark.run()
    assert len(benchmark.outputs) == 4
    assert sum(output.success for output in benchmark.outputs) == 1
    assert all(output.error for output in benchmark.outputs if not output.success)
    benchmark.benchmark_time = 2.0
    benchmark.print_stats()
    output = capsys.readouterr().out
    assert "Successful requests: 1" in output
    assert "Failed requests: 3" in output
    assert "Throughput: 0.50 requests/s" in output
    for result in benchmark.outputs:
        result.success = False
    benchmark.print_stats()
    assert "Throughput: 0.00 requests/s" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_cancellation_closes_session_and_workers(runner_class):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handle(request):
        entered.set()
        await release.wait()
        return web.json_response({"data": []})

    async def no_warmup():
        pass

    async with embedding_server(handle) as url:
        benchmark = runner_class(url, "bge-m3", [{"sentence": "a"}] * 5, False, 2)
        benchmark.warm_up = no_warmup
        task = asyncio.create_task(benchmark.run())
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            session = benchmark._session
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert session.closed
            assert benchmark._session is None
            assert not benchmark.outputs
            assert not any(
                "EmbeddingBenchmarkRunner.worker" in t.get_coro().__qualname__
                for t in asyncio.all_tasks()
                if not t.done()
            )
        finally:
            release.set()


@pytest.mark.parametrize("inputs,concurrency", [([], 1), ([{"sentence": "a"}], 0)])
def test_invalid_workload(runner_class, inputs, concurrency):
    with pytest.raises(ValueError):
        runner_class(
            "http://localhost/v1/embeddings", "bge-m3", inputs, False, concurrency
        )
