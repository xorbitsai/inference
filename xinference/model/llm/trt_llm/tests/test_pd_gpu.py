# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Explicitly opt-in two-GPU P/D acceptance; never selects busy GPUs itself."""

import asyncio
import json
import os
import signal
import sys

import pytest
import xoscar as xo

from ...sglang.xavier.directory import XavierPDDirectory

pytestmark = pytest.mark.skipif(
    sys.platform != "linux"
    or not os.environ.get("XINFERENCE_TRTLLM_TEST_MODEL")
    or not os.environ.get("XINFERENCE_TRTLLM_PD_TEST_GPUS"),
    reason="Requires TRT-LLM 1.2.1, a model and two explicitly reserved GPU IDs",
)

SCRIPT = r"""
import asyncio
import json
import sys
from pathlib import Path
from xinference.model.llm.llm_family import match_llm
from xinference.model.llm.trt_llm.core import TRTLLMModel

async def wait(path):
    for _ in range(1800):
        if path.exists():
            return
        await asyncio.sleep(.1)
    raise TimeoutError(str(path))

async def main():
    root, role, backend = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    args = json.loads((root/'config.json').read_text())
    config = dict(max_seq_len=1024, dtype='bfloat16', enable_thinking=False,
                  max_batch_size=4, max_num_tokens=1024,
                  cuda_graph_config=None, kv_cache_config={'max_tokens':2048})
    if backend == 'xavier':
        config['_xavier_cache_config'] = dict(address=args['address'], uid='trt-pd-live', role=role, rank=0 if role=='prefill' else 1, host='127.0.0.1')
    else:
        config['_nixl_config'] = dict(role=role)
    model = TRTLLMModel('live', match_llm('qwen3', 'pytorch', '0_6', 'none'), args['model'], config)
    tokens = dict(max_tokens=16, temperature=0, ignore_eos=True, stop=[], stop_token_ids=[])
    prompt = 'The capital of France is Paris. '*24
    try:
        model.load()
        if role == 'prefill':
            if args.get('fault'):
                fault = await model.async_generate(prompt, {**tokens, '_pd_kv_transfer_params': {'do_remote_decode':True}}, 'fault')
                (root/'fault.json').write_text(json.dumps(fault['_pd_kv_transfer_params']))
                await wait(root/'fault-done')
                await model.release_pd_request('fault')
            result = await model.async_generate(prompt, {**tokens, '_pd_kv_transfer_params': {'do_remote_decode':True}}, 'live-pd')
            (root/'handoff.json').write_text(json.dumps(result['_pd_kv_transfer_params']))
            await wait(root/'release')
            await model.release_pd_request('live-pd')
            assert not model._pd.active
        else:
            pd = model._pd
            model._pd = None
            try:
                baseline = await model.async_generate(prompt, tokens)
            finally:
                model._pd = pd
            (root/'ready').touch()
            if args.get('fault'):
                from xinference.model.llm.trt_llm.gpu import service
                transport = service()
                original_submit = transport.submit
                import threading
                started, release = threading.Event(), threading.Event()
                def submit(method, *args):
                    pending = original_submit(method, *args)
                    async def delayed():
                        # Exercise real NIXL writes, then hold completion. The
                        # destination must remain unschedulable until released.
                        await asyncio.wrap_future(pending)
                        started.set()
                        await asyncio.to_thread(release.wait)
                        if args_config['fault'] == 'fail':
                            raise RuntimeError('injected transfer failure')
                    return transport.client.submit(delayed())
                args_config = args
                transport.submit = submit
                await wait(root/'fault.json')
                transfer = json.loads((root/'fault.json').read_text())
                task = asyncio.create_task(model.async_generate(prompt, {**tokens, '_pd_kv_transfer_params': transfer}, 'fault'))
                assert await asyncio.to_thread(started.wait, 10)
                assert not task.done()
                if args['fault'] == 'cancel':
                    task.cancel()
                release.set()
                try:
                    await task
                except (RuntimeError, asyncio.CancelledError) as error:
                    if args['fault'] == 'fail':
                        assert 'injected transfer failure' in str(error), error
                    else:
                        assert isinstance(error, asyncio.CancelledError), error
                    (root/'fault-observed.json').write_text(json.dumps(dict(exception=type(error).__name__)))
                else:
                    raise AssertionError('faulted load must not generate output')
                transport.submit = original_submit
                assert not model._requests and not model._pd.active
                (root/'fault-done').touch()
            await wait(root/'handoff.json')
            transfer = json.loads((root/'handoff.json').read_text())
            config = {**tokens, 'stream':args['stream'], 'stream_options': {'include_usage':True}, '_pd_kv_transfer_params':transfer}
            result = await model.async_generate(prompt, config, 'live-pd')
            if args['stream']:
                chunks = [chunk async for chunk in result]
                text = ''.join(c['text'] for chunk in chunks for c in chunk['choices'])
                usage = chunks[-1]['usage']
            else:
                text, usage = result['choices'][0]['text'], result['usage']
            assert text == baseline['choices'][0]['text'], (text, baseline)
            assert usage == baseline['usage'], (usage, baseline)
            assert not model._requests
            transfer_stats = None
            if backend == 'xavier':
                from xinference.model.llm.trt_llm.gpu import service
                import os
                transfer_stats = service().rpc('get_stats')
                assert transfer_stats['transfer_pid'] != os.getpid()
                assert transfer_stats['gpu_batches'] > 0 and transfer_stats['cpu_batches'] == 0
            (root/'result.json').write_text(json.dumps(dict(text=text, usage=usage, transfer_stats=transfer_stats)))
    finally:
        model.stop()

asyncio.run(main())
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend,stream,fault",
    [
        ("xavier", False, None),
        ("xavier", True, None),
        ("nixl", False, None),
        ("nixl", True, None),
        ("xavier", False, "fail"),
        ("xavier", False, "cancel"),
    ],
)
async def test_live_two_gpu_pd_matches_baseline(tmp_path, backend, stream, fault):
    gpu_ids = os.environ["XINFERENCE_TRTLLM_PD_TEST_GPUS"].split(",")
    assert len(gpu_ids) == 2 and len(set(gpu_ids)) == 2
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        cache = None
        if backend == "xavier":
            cache = await xo.create_actor(
                XavierPDDirectory,
                address=pool.external_address,
                uid="trt-pd-live",
            )
        (tmp_path / "config.json").write_text(
            json.dumps(
                dict(
                    address=pool.external_address,
                    model=os.environ["XINFERENCE_TRTLLM_TEST_MODEL"],
                    stream=stream,
                    fault=fault,
                )
            )
        )
        processes, logs = [], []

        async def start(role, gpu):
            log = (tmp_path / f"{role}.log").open("w")
            logs.append(log)
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                SCRIPT,
                str(tmp_path),
                role,
                backend,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)

        async def wait(name):
            for _ in range(3000):
                for process in processes:
                    if process.returncode not in (None, 0):
                        pytest.fail(
                            "\n".join(
                                p.read_text()[-10000:] for p in tmp_path.glob("*.log")
                            )
                        )
                if (tmp_path / name).exists():
                    return
                await asyncio.sleep(0.2)
            raise TimeoutError(f"TRT P/D did not produce {name}; logs in {tmp_path}")

        try:
            await start("decode", gpu_ids[1])
            await wait("ready")
            await start("prefill", gpu_ids[0])
            await wait("result.json")
            (tmp_path / "release").touch()
            await asyncio.wait_for(
                asyncio.gather(*(p.wait() for p in processes)), timeout=60
            )
            assert all(p.returncode == 0 for p in processes)
            if cache is not None:
                stats = await cache.get_stats()
                assert stats["gpu_bytes"] > 0 and stats["host_bytes"] == 0
                assert stats["imported_tokens"] > 0 and stats["active_handoffs"] == 0
        finally:
            (tmp_path / "release").touch()
            for process in processes:
                if process.returncode is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(process.wait(), timeout=30)
                    except asyncio.TimeoutError:
                        os.killpg(process.pid, signal.SIGKILL)
                        await process.wait()
            for log in logs:
                log.close()
