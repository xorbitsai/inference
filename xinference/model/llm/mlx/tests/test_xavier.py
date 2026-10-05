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
async def test_pd_failed_prefill_releases_reserved_pages():
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        ref = await xo.create_actor(
            XavierBytesCacheActor, address=pool.external_address, uid="mlx-cache"
        )
        p = MLXXavierCache(
            contract(), dict(address=ref.address, uid="mlx-cache", role="prefill")
        )
        m = model()

        def fail(*args, **kwargs):
            raise RuntimeError("prefill failed")

        m.__class__.__call__, saved = fail, m.__class__.__call__
        try:
            with pytest.raises(RuntimeError, match="prefill failed"):
                await p.prefill(m, [1, 2, 3])
        finally:
            m.__class__.__call__ = saved
        assert (await ref.get_stats())["active_handoffs"] == 0


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
                return await wrapper.generate(
                    "a" * 70, 8, temperature=0, kv_transfer_params=transfer
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
