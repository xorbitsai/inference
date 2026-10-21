# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Opt-in source-side cancellation regressions without a decode receiver."""

import asyncio
import os
import signal
import sys

import pytest
import xoscar as xo

from ...sglang.xavier.directory import XavierPDDirectory
from .test_pd_gpu import pytestmark as pytestmark

SETUP = r"""
import asyncio, json, sys, time, threading
from pathlib import Path
from xinference.model.llm.llm_family import match_llm
from xinference.model.llm.trt_llm.core import TRTLLMModel
root = Path(sys.argv[1])
model_path = __import__('os').environ['XINFERENCE_TRTLLM_TEST_MODEL']
config = dict(max_seq_len=1024, dtype='bfloat16', enable_thinking=False,
              max_batch_size=4, max_num_tokens=1024, cuda_graph_config=None,
              kv_cache_config={'max_tokens':2048})
tokens = dict(max_tokens=1, temperature=0, ignore_eos=True, stop=[], stop_token_ids=[])
family = match_llm('qwen3','pytorch','0_6','none')
"""

NATIVE = (
    SETUP
    + r"""
async def main():
    reuse = sys.argv[2] == 'True'
    config['kv_cache_config']['enable_block_reuse'] = reuse
    config['cache_transceiver_config'] = {'kv_transfer_timeout_ms':None}
    config['_nixl_config'] = {'role':'prefill'}
    model = TRTLLMModel('native-release',family,model_path,config)
    times=[]
    try:
        model.load()
        executor=model._engine._executor.engine
        total_tokens=0
        for i in range(20):
            prompt=f'Unique native request {i}. '+ 'The capital of France is Paris. '*24
            total_tokens += len(model._tokenizer.encode(prompt))
            rid=f'no-receiver-{i}'
            result=await model.async_generate(prompt,{**tokens,'_pd_kv_transfer_params':{'do_remote_decode':True}},rid)
            ticket=result['_pd_kv_transfer_params']['trt_llm']['context']['ctx_request_id']
            assert ticket in executor.ctx_in_transmission_requests
            request=model._pd.active[rid]['request']
            # This is the retired mapping that makes result.abort() ineffective.
            assert request.request_id not in model._engine._executor._client_id_to_request_id
            start=time.monotonic()
            await asyncio.wait_for(model.release_pd_request(rid), 5)
            times.append(time.monotonic()-start)
            assert ticket not in executor.ctx_in_transmission_requests
            assert not request.aborted()
            assert not model._pd.active
        assert total_tokens>2048
        (root/'result.json').write_text(json.dumps(dict(reuse=reuse,timeout=None,requests=20,max_release_s=max(times),total_tokens=total_tokens)))
    finally:
        model.stop()
asyncio.run(main())
"""
)

NATIVE_EXHAUSTION = (
    SETUP
    + r"""
async def main():
    reuse=sys.argv[2]=='True'
    config['kv_cache_config']['enable_block_reuse']=reuse
    config['cache_transceiver_config']={
        'kv_transfer_timeout_ms':None,
        'kv_transfer_sender_future_timeout_ms':None,
    }
    config['_nixl_config']={'role':'prefill'}
    model=TRTLLMModel('native-exhaustion',family,model_path,config)
    pending=None
    try:
        model.load()
        executor=model._engine._executor.engine
        assert executor.llm_args.cache_transceiver_config.kv_transfer_sender_future_timeout_ms==1000
        exhausted=threading.Event()
        original=executor._check_disagg_ctx_cache_transfer_status
        def check(count=0):
            if count and not model._pd._native._pending and not model._pd._native._cancelled:
                exhausted.set()
            return original(count)
        executor._check_disagg_ctx_cache_transfer_status=check
        async def generate(i):
            prompt=f'Unique exhaustion request {i}. '+'The capital of France is Paris. '*110
            return await model.async_generate(prompt,{**tokens,'_pd_kv_transfer_params':{'do_remote_decode':True}},str(i))
        first=await generate(0)
        await generate(1)
        pending=asyncio.create_task(generate(2))
        assert await asyncio.to_thread(exhausted.wait,10)
        assert not pending.done()
        # The no-fitting-request check must already be inside native C++,
        # before release can populate the adapter's pending cancellation queue.
        await asyncio.sleep(.1)
        ticket=first['_pd_kv_transfer_params']['trt_llm']['context']['ctx_request_id']
        start=time.monotonic()
        await asyncio.wait_for(model.release_pd_request('0'),5)
        elapsed=time.monotonic()-start
        assert ticket not in executor.ctx_in_transmission_requests
        await asyncio.wait_for(pending,10)
        await model.release_pd_request('1')
        await model.release_pd_request('2')
        assert not model._pd.active and not executor.ctx_in_transmission_requests
        (root/'result.json').write_text(json.dumps(dict(reuse=reuse,sender_timeout_input=None,sender_timeout_effective=1000,release_s=elapsed)))
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending,return_exceptions=True)
        model.stop()
asyncio.run(main())
"""
)

XAVIER = (
    SETUP
    + r"""
async def main():
    address,barrier=sys.argv[2:4]
    config['_xavier_cache_config']=dict(address=address,uid='trt-cancel-live',role='prefill',rank=0,host='127.0.0.1')
    model=TRTLLMModel('xavier-cancel',family,model_path,config)
    primer_started,primer_release=threading.Event(),threading.Event()
    cancelled_started,callback_release=threading.Event(),threading.Event()
    tasks=[]
    try:
        model.load()
        from xinference.model.llm.trt_llm.gpu import service
        from xinference.model.llm.trt_llm.pd import request_state
        executor=model._engine._executor.engine
        prompt='The capital of France is Paris. '*24
        pd=model._pd
        model._pd=None
        try:
            baseline=await model.async_generate(prompt,tokens)
        finally:
            model._pd=pd
        original_forward=executor.model_engine.forward
        original_rpc=service().rpc
        def forward(batch,*args,**kwargs):
            states=[request_state(r) for r in batch.context_requests]
            output=original_forward(batch,*args,**kwargs)
            state=model._pd.active.get('primer',{}).get('context')
            if state is not None and any(s is state for s in states):
                primer_started.set()
                assert primer_release.wait(10)
            state=model._pd.active.get('cancelled',{}).get('context')
            if state is not None and any(s is state for s in states):
                assert len(executor.active_requests)>=2
                if barrier=='forward':
                    cancelled_started.set()
                    assert callback_release.wait(10)
            return output
        def rpc(method,*args):
            state=model._pd.active.get('cancelled',{}).get('context')
            if barrier=='publish' and method=='publish' and state is not None and args[0]==state.room:
                cancelled_started.set()
                assert callback_release.wait(10)
            return original_rpc(method,*args)
        executor.model_engine.forward=forward
        service().rpc=rpc
        async def generate(rid):
            return await model.async_generate(prompt,{**tokens,'_pd_kv_transfer_params':{'do_remote_decode':True}},rid)
        primer=asyncio.create_task(generate('primer'));tasks.append(primer)
        assert await asyncio.to_thread(primer_started.wait,10)
        cancelled=asyncio.create_task(generate('cancelled'));tasks.append(cancelled)
        other=asyncio.create_task(generate('other'));tasks.append(other)
        for _ in range(100):
            if len(model._requests)==3:break
            await asyncio.sleep(.01)
        assert len(model._requests)==3
        primer_release.set()
        assert await asyncio.to_thread(cancelled_started.wait,10)
        room=model._pd.active['cancelled']['context'].room
        cancelled.cancel()
        try:
            await asyncio.wait_for(cancelled,5)
        except asyncio.CancelledError:
            pass
        else:raise AssertionError('cancellation did not propagate')
        assert 'cancelled' not in model._pd.active
        directory=await xo.actor_ref(address=address,uid='trt-cancel-live')
        try:await directory.request_info(room)
        except RuntimeError:pass
        else:raise AssertionError('cancelled room must be dropped before late callback')
        callback_release.set()
        results=await asyncio.wait_for(asyncio.gather(primer,other),10)
        assert all(r['choices'][0]['text']==baseline['choices'][0]['text'] for r in results)
        await model.release_pd_request('primer')
        await model.release_pd_request('other')
        follow=await generate('follow-up')
        assert follow['choices'][0]['text']==baseline['choices'][0]['text']
        await model.release_pd_request('follow-up')
        assert not model._pd.active and not model._requests
        stats=await directory.get_stats()
        assert stats['active_handoffs']==0
        (root/'result.json').write_text(json.dumps(dict(barrier=barrier,stats=stats)))
    finally:
        primer_release.set();callback_release.set()
        for task in tasks:
            if not task.done():task.cancel()
        if tasks:await asyncio.gather(*tasks,return_exceptions=True)
        model.stop()
import xoscar as xo
asyncio.run(main())
"""
)


async def run_script(tmp_path, script, *args):
    with (tmp_path / "worker.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            *args,
            env=dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=os.environ["XINFERENCE_TRTLLM_PD_TEST_GPUS"].split(
                    ","
                )[0],
            ),
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            await asyncio.wait_for(process.wait(), 120)
            assert process.returncode == 0, (tmp_path / "worker.log").read_text()[
                -18000:
            ]
            assert (tmp_path / "result.json").exists()
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), 10)
                except asyncio.TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("reuse", [True, False])
async def test_live_native_release_without_receiver(tmp_path, reuse):
    await run_script(tmp_path, NATIVE, str(reuse))


@pytest.mark.asyncio
@pytest.mark.parametrize("reuse", [True, False])
async def test_live_native_release_after_no_receiver_exhaustion(tmp_path, reuse):
    await run_script(tmp_path, NATIVE_EXHAUSTION, str(reuse))


@pytest.mark.asyncio
@pytest.mark.parametrize("barrier", ["forward", "publish"])
async def test_live_cancelled_prefill_keeps_other_request_alive(tmp_path, barrier):
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        directory = await xo.create_actor(
            XavierPDDirectory, address=pool.external_address, uid="trt-cancel-live"
        )
        await run_script(tmp_path, XAVIER, directory.address, barrier)
