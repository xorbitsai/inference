# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Opt-in acceptance across separate TRT and vLLM/SGLang environments."""

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
    or not all(
        os.environ.get(name)
        for name in (
            "XINFERENCE_TRTLLM_TEST_MODEL",
            "XINFERENCE_TRTLLM_PD_TEST_GPUS",
            "XINFERENCE_TRTLLM_CUDA_PYTHON",
        )
    ),
    reason="Requires reserved GPUs and prepared TRT and vLLM/SGLang environments",
)

SCRIPT = r"""
import asyncio, json, sys
from pathlib import Path
from xinference.model.llm.llm_family import match_llm

async def main():
    root, role, engine = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    args = json.loads((root/'config.json').read_text())
    async def wait(name):
        for _ in range(3000):
            if (root/name).exists(): return
            await asyncio.sleep(.1)
        raise TimeoutError(name)
    if engine == 'trt-llm':
        from xinference.model.llm.trt_llm.core import TRTLLMModel as Model
        config = dict(max_seq_len=1024, dtype='bfloat16', max_batch_size=4,
                      max_num_tokens=1024, cuda_graph_config=None,
                      kv_cache_config={'max_tokens':2048})
    elif engine == 'vllm':
        from xinference.model.llm.vllm.core import VLLMModel as Model
        config = dict(max_model_len=1024, dtype='bfloat16', enforce_eager=True,
                      gpu_memory_utilization=.25, num_gpu_blocks_override=32,
                      max_num_seqs=4, block_size=64)
    else:
        from xinference.model.llm.sglang.core import SGLANGModel as Model
        config = dict(context_length=1024, dtype='bfloat16', disable_cuda_graph=True,
                      mem_fraction_static=.3, max_total_tokens=2048, page_size=64,
                      chunked_prefill_size=-1, triton_attention_reduce_in_fp32=False)
    config['enable_thinking'] = False
    family = match_llm('qwen3', 'pytorch', '0_6', 'none')
    prompts = ['The capital of France is Paris. '*n for n in (1, 9, 24)]
    options = dict(max_tokens=16, temperature=0, ignore_eos=True, stop=[], stop_token_ids=[])
    def create(pd):
        cfg = dict(config)
        if pd:
            cfg['_xavier_cache_config'] = dict(address=args['address'], uid='trt-cross-live',
                role=role, rank=1 if role=='prefill' else 2, host='127.0.0.1',
                heterogeneous=True, logical_dtype='bfloat16', gpu_cache_bytes=0)
            if role == 'prefill' and args.get('history'):
                cfg['_xavier_cache_config']['prefix_cache'] = args['history']
        return Model('live', family, args['model'], cfg)
    baseline = []
    if role == 'decode':
        model = create(False)
        try:
            model.load()
            if hasattr(model, "wait_for_load"): model.wait_for_load()
            for prompt in prompts:
                baseline.append(await model.async_generate(prompt, options))
        finally:
            model.stop()
    model = create(True)
    report = []
    try:
        model.load()
        if hasattr(model, "wait_for_load"): model.wait_for_load()
        if role == 'decode': (root/'ready').touch()
        else: await wait('ready')
        for i, prompt in enumerate(prompts):
            for stream in (False, True):
                name = f'{i}-{int(stream)}'
                handoff = dict(room=100+i*2+int(stream), mode='gpu', heterogeneous=True)
                transfer = dict(sglang_xavier=handoff)
                transfer['do_remote_decode' if role=='prefill' else 'do_remote_prefill'] = True
                request_id = 'request-'+name
                output = await model.async_generate(prompt, dict(options, stream=stream if role=='decode' else False,
                    _pd_kv_transfer_params=transfer), request_id=request_id)
                if role == 'decode':
                    if stream:
                        chunks = [chunk async for chunk in output]
                        text = ''.join(c['text'] for chunk in chunks for c in chunk['choices'])
                    else: text = output['choices'][0]['text']
                    expected = baseline[i]['choices'][0]['text']
                    report.append(dict(prompt=prompt, stream=stream, text=text, baseline=expected))
                    (root/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
                    assert text == expected, (engine, text, expected)
                    (root/(name+'.done')).touch()
                else:
                    await wait(name+'.done')
                    if engine == 'trt-llm': await model.release_pd_request(request_id)
        (root/(role+'.done')).touch()
        if role == 'prefill': await wait('release')
    finally:
        model.stop()

if __name__ == '__main__': asyncio.run(main())
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prefill,decode",
    [
        ("trt-llm", "vllm"),
        ("vllm", "trt-llm"),
        ("trt-llm", "sglang"),
        ("sglang", "trt-llm"),
    ],
)
async def test_live_cross_engine_pd(tmp_path, prefill, decode):
    gpu_ids = os.environ["XINFERENCE_TRTLLM_PD_TEST_GPUS"].split(",")
    assert len(gpu_ids) == 2 and len(set(gpu_ids)) == 2
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    processes, logs = [], []
    async with pool:
        directory = await xo.create_actor(
            XavierPDDirectory, address=pool.external_address, uid="trt-cross-live"
        )
        (tmp_path / "config.json").write_text(
            json.dumps(
                dict(
                    address=pool.external_address,
                    model=os.environ["XINFERENCE_TRTLLM_TEST_MODEL"],
                )
            )
        )

        async def start(role, engine, gpu):
            python = (
                sys.executable
                if engine == "trt-llm"
                else os.environ["XINFERENCE_TRTLLM_CUDA_PYTHON"]
            )
            log = (tmp_path / f"{role}.log").open("w")
            logs.append(log)
            processes.append(
                await asyncio.create_subprocess_exec(
                    python,
                    "-c",
                    SCRIPT,
                    str(tmp_path),
                    role,
                    engine,
                    env=dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES=gpu,
                        TOKENIZERS_PARALLELISM="false",
                        SGLANG_PLUGINS="__xinference_test__",
                    ),
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
            )

        async def wait(name):
            for _ in range(3000):
                assert all(
                    p.returncode in (None, 0) for p in processes
                ), f"Engine failed; inspect {tmp_path}"
                if (tmp_path / name).exists():
                    return
                await asyncio.sleep(0.2)
            raise TimeoutError(f"Missing {name}; inspect {tmp_path}")

        try:
            await start("decode", decode, gpu_ids[1])
            await wait("ready")
            await start("prefill", prefill, gpu_ids[0])
            await wait("decode.done")
            await wait("prefill.done")
            (tmp_path / "release").touch()
            await asyncio.wait_for(asyncio.gather(*(p.wait() for p in processes)), 60)
            assert all(p.returncode == 0 for p in processes)
            stats = await directory.get_stats()
            assert stats["completed_requests"] == 6
            assert (
                stats["gpu_bytes"] > 0
                and stats["host_bytes"] == stats["active_handoffs"] == 0
            )
        finally:
            (tmp_path / "release").touch()
            for p in processes:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                if p.returncode is None:
                    try:
                        await asyncio.wait_for(p.wait(), 30)
                    except asyncio.TimeoutError:
                        os.killpg(p.pid, signal.SIGKILL)
                        await p.wait()
            for log in logs:
                log.close()
