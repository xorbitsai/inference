# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
import importlib
import json
import platform
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import xoscar as xo

from ...xavier.backends.bytes.storage import XavierBytesCacheActor
from ...xavier.contract import KVCacheContract
from ..xavier import MLXXavierCache, configure_xavier

metal = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Requires Apple Metal",
)


def contract():
    return KVCacheContract(
        *["0" * 64] * 4,
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=64,
        logical_dtype="float16",
    )


def model(model_type="qwen2"):
    import mlx.core as mx

    module = importlib.import_module(f"mlx_lm.models.{model_type}")
    mx.random.seed(42)
    args = dict(
        model_type=model_type,
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
    )
    if model_type == "qwen3":
        args.update(
            head_dim=8,
            max_position_embeddings=2048,
            rope_theta=10000.0,
            tie_word_embeddings=False,
        )
    value = module.Model(module.ModelArgs(**args))
    value.set_dtype(mx.float16)
    mx.eval(value.parameters())
    return value


@pytest.mark.parametrize("role", [None, "prefill", "decode"])
def test_vision_model_rejects_xavier_before_loading(role):
    from ..core import MLXVisionModel

    family = SimpleNamespace(
        model_specs=[SimpleNamespace(quantization="bf16", model_revision=None)],
        model_ability=["chat", "vision"],
    )
    with pytest.raises(ValueError, match="does not support vision"):
        MLXVisionModel(
            "vision-rep0",
            family,
            "unused",
            {"_xavier_cache_config": dict(address="address", uid="cache", role=role)},
        )
    assert MLXVisionModel("vision-rep0", family, "unused", {})._xavier_config is None


@pytest.mark.asyncio
async def test_pd_metadata_is_checked_before_reading():
    client = MLXXavierCache(
        contract(), dict(address="address", uid="cache", role="decode")
    )
    client._call = AsyncMock()
    with pytest.raises(ValueError, match="handoff"):
        await client.fetch(
            [1, 2, 3], {"mlx_xavier": dict(engine="mlx", address="wrong")}
        )
    client._call.assert_not_awaited()


@pytest.mark.asyncio
async def test_decode_claims_before_returning_lazy_stream():
    from ..core import MLXModel

    wrapper = object.__new__(MLXModel)
    wrapper._model = object()
    wrapper._tokenizer = SimpleNamespace(encode=lambda prompt: [1, 2, 3])
    wrapper._xavier = SimpleNamespace(
        role="decode", fetch=AsyncMock(return_value=([], 2))
    )
    wrapper._model_generation_config = {}
    wrapper.model_uid = "mlx"
    wrapper._batch_model = SimpleNamespace(generate_stream=AsyncMock())
    transfer = {"do_remote_prefill": True}
    stream = await wrapper.async_generate(
        "prompt", {"stream": True, "max_tokens": 2, "_pd_kv_transfer_params": transfer}
    )
    wrapper._xavier.fetch.assert_awaited_once_with([1, 2, 3], transfer)
    wrapper._batch_model.generate_stream.assert_not_called()
    await stream.aclose()


@pytest.mark.asyncio
async def test_decode_passes_prepared_cache_and_tokens_to_batch_stream():
    from ..core import MLXModel

    wrapper = object.__new__(MLXModel)
    wrapper._model = object()
    wrapper._tokenizer = SimpleNamespace(encode=lambda prompt: [1, 2, 3])
    prepared = ([object()], 2)
    wrapper._xavier = SimpleNamespace(
        role="decode", fetch=AsyncMock(return_value=prepared)
    )
    wrapper._model_generation_config = {}
    wrapper.model_uid = "mlx"

    async def chunks(**kwargs):
        yield {"choices": [{"text": "token"}]}

    wrapper._batch_model = SimpleNamespace(
        generate_stream=MagicMock(side_effect=chunks)
    )
    transfer = {"do_remote_prefill": True}
    stream = await wrapper.async_generate(
        "prompt", {"stream": True, "max_tokens": 2, "_pd_kv_transfer_params": transfer}
    )
    assert [c["model"] async for c in stream] == ["mlx"]
    kwargs = wrapper._batch_model.generate_stream.call_args.kwargs
    assert kwargs["prepared_cache"] is prepared and kwargs["prompt_token_ids"] == [
        1,
        2,
        3,
    ]
    wrapper._xavier.fetch.assert_awaited_once_with([1, 2, 3], transfer)


@pytest.mark.asyncio
async def test_chat_prefill_preserves_handoff_metadata():
    from ..core import MLXChatModel

    wrapper = object.__new__(MLXChatModel)
    wrapper._model = object()
    wrapper._tokenizer = SimpleNamespace(encode=lambda prompt: [1, 2, 3])
    wrapper._model_generation_config = {}
    wrapper.model_uid = "mlx"
    wrapper.model_family = SimpleNamespace(
        model_family="qwen2",
        model_name="qwen2",
        chat_template="template",
        stop=None,
        stop_token_ids=None,
    )
    wrapper.reasoning_parser = None
    wrapper.get_full_context = MagicMock(return_value="chat prompt")
    wrapper._get_reusable_prompt_prefix_len = MagicMock(return_value=2)
    transfer = {"do_remote_prefill": True, "mlx_xavier": {"ticket": "ticket"}}
    wrapper._xavier = SimpleNamespace(
        role="prefill", prefill=AsyncMock(return_value=transfer)
    )
    result = await wrapper.async_chat(
        [{"role": "user", "content": "hello"}],
        {"_pd_kv_transfer_params": {"do_remote_decode": True}},
    )
    assert (
        result["object"] == "chat.completion"
        and result["_pd_kv_transfer_params"] is transfer
    )
    wrapper._xavier.prefill.assert_awaited_once_with(wrapper._model, [1, 2, 3], 2)


@pytest.mark.asyncio
async def test_publication_failure_does_not_drop_batch_results(monkeypatch, caplog):
    from ..core import MLXBatchModel

    monkeypatch.setattr(MLXBatchModel, "_lock", None)
    monkeypatch.setattr(MLXBatchModel, "_is_new_mlx_lm", lambda: True)
    wrapper = object.__new__(MLXBatchModel)
    wrapper._prompt_cache = None
    wrapper._xavier = SimpleNamespace(
        publish=MagicMock(side_effect=[ValueError("bad cache"), None])
    )
    prompts = [SimpleNamespace(uid=i, end_of_prompt=True) for i in (1, 2)]
    results = [SimpleNamespace(uid=i, finish_reason="stop", token=7) for i in (1, 2)]
    generator = SimpleNamespace(
        sampler="test",
        next=MagicMock(side_effect=[(prompts, results)]),
        extract_cache=lambda uids: {uids[0]: ([], [])},
    )
    queues = {i: asyncio.Queue() for i in (1, 2)}
    state = dict(
        generator=generator,
        queues=queues,
        pending={},
        active={1, 2},
        cache_boundaries={},
        xavier_prompts={i: ([1, 2], 0) for i in (1, 2)},
        xavier_writes={},
    )
    task = asyncio.create_task(wrapper._background_worker(state))
    try:
        delivered = await asyncio.wait_for(
            asyncio.gather(*(q.get() for q in queues.values())), 1
        )
        assert delivered == results and not state["active"]
        assert wrapper._xavier.publish.call_count == 2
        assert "prefix publication failed" in caplog.text
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_flush_keeps_other_requests_writes_independent():
    client = MLXXavierCache(contract(), {})
    client._ref = SimpleNamespace(
        configure=AsyncMock(return_value={"capacity_pages": 4096, "max_keys": 4096})
    )
    client.encode = MagicMock(return_value=[b"page"])
    gates = {client.keys([i])[0]: asyncio.Event() for i in (1, 2)}

    async def put(method, namespace, keys, pages, start):
        await gates[keys[0]].wait()

    client._call = put
    first, second = (client.publish([], [i]) for i in (1, 2))
    flush = asyncio.create_task(client.flush([first]))
    await asyncio.sleep(0)
    flush.cancel()
    with pytest.raises(asyncio.CancelledError):
        await flush
    assert not first.cancelled() and not second.cancelled()
    gates[client.keys([2])[0]].set()
    await asyncio.wait_for(client.flush([second]), 1)
    assert not first.done()
    gates[client.keys([1])[0]].set()
    await asyncio.wait_for(client.flush([first]), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("capacity,length", [(2, 129), (4097, 4096 * 64 + 1)])
async def test_oversized_shared_prompt_skips_encoding_and_publication(capacity, length):
    c = contract()
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        ref = await xo.create_actor(
            XavierBytesCacheActor,
            capacity * c.layer_nbytes * c.num_layers,
            address=pool.external_address,
        )
        client = MLXXavierCache(c, dict(address=ref.address, uid=ref.uid))
        client.encode = MagicMock(side_effect=AssertionError("must not encode"))
        client.keys = MagicMock(side_effect=AssertionError("must not build keys"))
        for _ in range(2):
            await client.publish([], [1] * length)
        client.encode.assert_not_called()
        client.keys.assert_not_called()
        assert client._capacity_pages == min(capacity, 4096)
        stats = await ref.get_stats()
        assert stats["stored_pages"] == 0 and stats["pages"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("decode_error", [False, True])
async def test_failed_release_keeps_decode_result_or_original_error(
    decode_error, caplog
):
    cfg = dict(address="address", uid="cache", role="decode")
    client = MLXXavierCache(contract(), cfg)
    tokens = [1, 2, 3]
    transfer = {
        "mlx_xavier": dict(
            engine="mlx",
            address="address",
            uid="cache",
            namespace=client.contract.fingerprint,
            keys=client.keys(tokens[:-1]),
            tokens=2,
            ticket="ticket",
        )
    }
    client._call = AsyncMock(side_effect=[[b"page"], TimeoutError("release failed")])
    client.decode = MagicMock(
        side_effect=ValueError("decode failed") if decode_error else None,
        return_value="cache",
    )
    if decode_error:
        with pytest.raises(ValueError, match="decode failed"):
            await client.fetch(tokens, transfer)
    else:
        assert await client.fetch(tokens, transfer) == ("cache", 2)
    assert "handoff release failed" in caplog.text


@metal
@pytest.mark.asyncio
async def test_load_registers_contract_and_rejects_incompatible_replica(tmp_path):
    from ..core import MLXModel

    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        ref = await xo.create_actor(
            XavierBytesCacheActor, address=pool.external_address, uid="mlx-cache"
        )
        for i in (0, 1):
            path = tmp_path / str(i)
            path.mkdir()
            (path / "config.json").write_text(
                json.dumps({"model_type": "qwen2", "max_position_embeddings": 2048})
            )
            (path / "tokenizer.json").write_text("{}")
            (path / "model.safetensors").write_bytes(bytes([i]))
            wrapper = object.__new__(MLXModel)
            wrapper._model = model()
            wrapper._loading_thread = None
            wrapper._model_config = {}
            wrapper._xavier_config = dict(
                address=ref.address, uid=ref.uid, role="prefill" if i == 0 else "decode"
            )
            wrapper._n_worker = 1
            wrapper._batch_model = object()
            wrapper.allow_batch = True
            wrapper._loop = asyncio.get_running_loop()
            wrapper.model_path = str(path)
            wrapper._update_model_generation_config = MagicMock()
            if i == 0:
                await asyncio.to_thread(wrapper.wait_for_load)
                assert (await ref.get_stats())[
                    "namespace"
                ] == wrapper._xavier.contract.fingerprint
            else:
                with pytest.raises(ValueError, match="weights_fingerprint"):
                    await asyncio.to_thread(wrapper.wait_for_load)


@metal
def test_weight_fingerprint_cache_avoids_rehash_and_invalidates_changes(
    tmp_path, monkeypatch
):
    from pathlib import Path

    import xinference.model.llm.mlx.xavier as module

    monkeypatch.setattr(module, "XINFERENCE_CACHE_DIR", str(tmp_path / "cache"))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (tmp_path / "tokenizer.json").write_text("{}")
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"first")
    reads = []
    saved = Path.open

    def tracked(path, mode="r", *args, **kwargs):
        if path == weight and mode == "rb":
            reads.append(path)
        return saved(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked)
    first = configure_xavier(model(), tmp_path, {}, {}, 1).contract
    second = configure_xavier(model(), tmp_path, {}, {}, 1).contract
    assert first.weights_fingerprint == second.weights_fingerprint and len(reads) == 1
    weight.write_bytes(b"changed")
    third = configure_xavier(model(), tmp_path, {}, {}, 1).contract
    assert third.weights_fingerprint != first.weights_fingerprint and len(reads) == 2


@metal
@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [0, 64, 65, 128])
async def test_publish_encodes_only_pages_beyond_remote_prefix(cached):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    m = model()
    tokens = [i % 128 for i in range(130)]
    cache = make_prompt_cache(m)
    mx.eval(m(mx.array(tokens)[None], cache=cache))
    client = MLXXavierCache(contract(), {})
    complete = client.encode(cache, tokens)
    client._ref = SimpleNamespace(
        configure=AsyncMock(return_value={"capacity_pages": 4096, "max_keys": 4096})
    )
    client._call = AsyncMock()
    await client.publish(cache, tokens, cached_tokens=cached)
    client._call.assert_awaited_once_with(
        "put",
        client.contract.fingerprint,
        client.keys(tokens),
        complete[cached // 64 :],
        cached // 64,
    )


@metal
@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen3_5"},
        {"model_type": "qwen2", "quantization": {"bits": 4}},
        {"model_type": "qwen2", "use_sliding_window": True},
        {"model_type": "qwen2", "layer_types": ["linear_attention"]},
    ],
)
def test_unsupported_model_contract_rejected(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="full-attention"):
        configure_xavier(model(), tmp_path, {}, {}, 1)


@metal
def test_effective_model_overrides_change_namespace(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (tmp_path / "model.safetensors").write_bytes(b"fingerprint-fixture")
    (tmp_path / "tokenizer.json").write_text("{}")
    m = model()
    first = configure_xavier(m, tmp_path, {}, {}, 1).contract
    m.args.rope_theta *= 2
    second = configure_xavier(m, tmp_path, {}, {}, 1).contract
    with pytest.raises(ValueError, match="fingerprint"):
        first.require_match(second)


@metal
@pytest.mark.parametrize("model_type", ["qwen2", "qwen3", "llama"])
def test_supported_architectures_use_actual_kv_geometry(tmp_path, model_type):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (tmp_path / "model.safetensors").write_bytes(b"fingerprint-fixture")
    (tmp_path / "tokenizer.json").write_text("{}")
    m = model(model_type)
    client = configure_xavier(m, tmp_path, {}, {}, 1)
    cache = make_prompt_cache(m)
    tokens = [1, 2, 3]
    mx.eval(m(mx.array(tokens)[None], cache=cache))
    restored = client.decode(client.encode(cache, tokens), len(tokens))
    assert client.contract.head_dim == 8
    assert all(
        bool(mx.array_equal(a, b))
        for old, new in zip(cache, restored)
        for a, b in zip(old.state, new.state)
    )


@metal
@pytest.mark.parametrize("length", [0, 1, 63, 64, 65, 129])
def test_real_metal_canonical_roundtrip(length):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    m = model()
    cache = make_prompt_cache(m)
    tokens = [i % 128 for i in range(length)]
    if length:
        mx.eval(m(mx.array(tokens)[None], cache=cache))
    client = MLXXavierCache(contract(), {})
    pages = client.encode(cache, tokens)
    restored = client.decode(pages, length)
    assert [c.offset for c in restored] == [length] * 2
    if length:
        assert all(
            bool(mx.array_equal(a, b))
            for old, new in zip(cache, restored)
            for a, b in zip(old.state, new.state)
        )
        cache[0].keys = cache[0].keys.astype(mx.bfloat16)
        with pytest.raises(ValueError, match="dtype"):
            client.encode(cache, tokens)


@metal
@pytest.mark.asyncio
async def test_real_pd_import_skips_prefill_and_matches_logits():
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.models.cache import make_prompt_cache

    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        ref = await xo.create_actor(
            XavierBytesCacheActor, address=pool.external_address, uid="mlx-cache"
        )
        cfg = dict(address=ref.address, uid="mlx-cache")
        p = MLXXavierCache(contract(), dict(cfg, role="prefill"))
        d = MLXXavierCache(contract(), dict(cfg, role="decode"))
        tokens = [i % 128 for i in range(130)]
        m = model()
        transfer = await p.prefill(m, tokens, prefix_length=24)
        cache, reused = await d.fetch(tokens, transfer)
        assert reused == len(tokens) - 1 and d.imported_tokens == reused
        reference = BatchGenerator(m)
        uid = reference.insert_segments([[tokens[:24], tokens[24:]]], max_tokens=[1])[0]
        while True:
            prompts, _ = reference.next()
            if any(r.end_of_prompt for r in prompts):
                break
        native_cache, _ = reference.extract_cache([uid])[uid]
        assert all(
            bool(mx.array_equal(a[:, :, :reused], b))
            for native, restored in zip(native_cache, cache)
            for a, b in zip(native.state, restored.state)
        )
        # D evaluates exactly one prompt token against imported cache state.
        actual = m(mx.array(tokens[reused:])[None], cache=cache)[:, -1]
        expected = m(mx.array(tokens)[None], cache=make_prompt_cache(m))[:, -1]
        mx.eval(actual, expected)
        assert bool(mx.array_equal(mx.argmax(actual, -1), mx.argmax(expected, -1)))
        assert (
            float(
                mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32)))
            )
            < 0.02
        )
        stats = await ref.get_stats()
        assert (
            stats["handoff_reads"] == 1
            and stats["read_pages"] == 3
            and stats["active_handoffs"] == 0
        )
        with pytest.raises(ValueError, match="handoff"):
            await d.fetch(tokens, transfer)
        p.imported_tokens = 0
        next_transfer = await p.prefill(m, tokens, prefix_length=24)
        assert p.imported_tokens == 0  # Complete warm pages stay on CPU until D reads.
        await d.fetch(tokens, next_transfer)
        assert (await ref.get_stats())["active_handoffs"] == 0


@metal
@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [64, 128])
async def test_partial_hit_prefill_computes_and_publishes_only_missing_suffix(cached):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        ref = await xo.create_actor(
            XavierBytesCacheActor, address=pool.external_address, uid="mlx-cache"
        )
        cfg = dict(address=ref.address, uid=ref.uid)
        p = MLXXavierCache(contract(), dict(cfg, role="prefill"))
        d = MLXXavierCache(contract(), dict(cfg, role="decode"))
        tokens = [i % 128 for i in range(130)]
        m = model()
        seed = await p.prefill(m, tokens[: cached + 1], prefix_length=24)
        await d.fetch(tokens[: cached + 1], seed)
        p.imported_tokens = 0
        p.encode = MagicMock(wraps=p.encode)
        transfer = await p.prefill(m, tokens, prefix_length=24)
        assert p.imported_tokens == cached
        p.encode.assert_called_once()
        assert p.encode.call_args.args[1:] == (tokens[:-1], cached)
        cache, reused = await d.fetch(tokens, transfer)
        assert reused == len(tokens) - 1
        actual = m(mx.array(tokens[reused:])[None], cache=cache)[:, -1]
        expected = m(mx.array(tokens)[None], cache=make_prompt_cache(m))[:, -1]
        mx.eval(actual, expected)
        assert bool(mx.array_equal(mx.argmax(actual, -1), mx.argmax(expected, -1)))
        assert float(mx.max(mx.abs(actual.astype(mx.float32) - expected))) < 0.02
        assert (await ref.get_stats())["active_handoffs"] == 0


@metal
@pytest.mark.asyncio
@pytest.mark.parametrize("release_fails", [False, True])
async def test_pd_failed_prefill_releases_reserved_pages(release_fails):
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        ref = await xo.create_actor(
            XavierBytesCacheActor, address=pool.external_address, uid="mlx-cache"
        )
        p = MLXXavierCache(
            contract(), dict(address=ref.address, uid="mlx-cache", role="prefill")
        )
        if release_fails:
            original_call = p._call

            async def call(method, *args):
                if method == "release_handoff":
                    raise TimeoutError("cleanup failed")
                return await original_call(method, *args)

            p._call = call
        m = model()

        def fail(*args, **kwargs):
            raise RuntimeError("prefill failed")

        m.__class__.__call__, saved = fail, m.__class__.__call__
        try:
            with pytest.raises(RuntimeError, match="prefill failed"):
                await p.prefill(m, [1, 2, 3])
        finally:
            m.__class__.__call__ = saved
        assert (await ref.get_stats())["active_handoffs"] == int(release_fails)


@metal
@pytest.mark.asyncio
async def test_real_batch_generation_reuses_remote_cache(monkeypatch):
    from ..core import MLXBatchModel

    class Tokenizer:
        eos_token_ids = []

        def encode(self, text):
            return list(text.encode())

        def decode(self, tokens, **kwargs):
            return "".join(f"<{t}>" for t in tokens)

    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        ref = await xo.create_actor(
            XavierBytesCacheActor, address=pool.external_address, uid="mlx-cache"
        )
        cfg = dict(address=ref.address, uid="mlx-cache")
        clients = []

        async def generate(role=None, transfer=None):
            monkeypatch.setattr(MLXBatchModel, "_batch_generators", {})
            monkeypatch.setattr(MLXBatchModel, "_lock", None)
            client = MLXXavierCache(contract(), dict(cfg, role=role))
            client.publish = MagicMock(wraps=client.publish)
            clients.append(client)
            wrapper = MLXBatchModel(
                model(), Tokenizer(), prompt_cache_size=0, xavier=client
            )
            try:
                tokens = Tokenizer().encode("a" * 70)
                prepared = await client.fetch(tokens, transfer)
                return await wrapper.generate(
                    "a" * 70,
                    8,
                    temperature=0,
                    prepared_cache=prepared,
                    prompt_token_ids=tokens,
                )
            finally:
                tasks = [
                    g["task"] for g in wrapper._batch_generators.values() if g["task"]
                ]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        cold, cold_usage = await generate()
        warm, warm_usage = await generate()
        assert (
            warm == cold and cold_usage["prompt_tokens_details"]["cached_tokens"] == 0
        )
        assert warm_usage["prompt_tokens_details"]["cached_tokens"] == 69
        p = MLXXavierCache(contract(), dict(cfg, role="prefill"))
        transfer = await p.prefill(model(), Tokenizer().encode("a" * 70))
        pd, pd_usage = await generate("decode", transfer)
        assert pd == cold and pd_usage["prompt_tokens_details"]["cached_tokens"] == 69
        assert [c.publish.call_count for c in clients] == [1, 0, 0]
        assert (await ref.get_stats())["handoff_reads"] == 1
