# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_load_uses_registered_packed_hybrid_cache():
    pytest.importorskip("vllm")
    from ..v1_connector import XavierConnector, XavierConnectorMetadata

    packed = object()
    conv, ssm = object(), object()
    request = SimpleNamespace(lease="")
    load = Mock()
    connector = SimpleNamespace(
        _is_consumer=True,
        _registered_kv_caches={"linear_attn": packed},
        _get_connector_metadata=lambda: XavierConnectorMetadata(
            load_requests=[request]
        ),
        _load_layer_blocks=load,
    )
    context = SimpleNamespace(
        no_compile_layers={"linear_attn": SimpleNamespace(kv_cache=(conv, ssm))}
    )
    XavierConnector.start_load_kv(connector, context)
    load.assert_called_once_with("linear_attn", packed, request)


@pytest.mark.asyncio
async def test_read_large_cache_blocks_incrementally():
    pytest.importorskip("vllm")
    from unittest.mock import AsyncMock

    import torch

    from ..v1_connector import XavierConnector

    transfer = SimpleNamespace(
        read_layer_blocks_v1=AsyncMock(
            side_effect=[torch.tensor([[10.0]]), torch.tensor([[20.0]])]
        )
    )
    connector = SimpleNamespace(_get_transfer_ref=AsyncMock(return_value=transfer))
    result = await XavierConnector._read_layer_blocks(
        connector, "layer", 1, {7: 2, 9: 3}, (2, 1), torch.float32
    )
    assert result.tolist() == [[10.0], [20.0]]
    assert transfer.read_layer_blocks_v1.await_count == 2
    transfer.read_layer_blocks_v1.assert_any_await(
        1, "layer", {7: 2}, (1, 1), torch.float32
    )
    transfer.read_layer_blocks_v1.assert_any_await(
        1, "layer", {9: 3}, (1, 1), torch.float32
    )


def test_hybrid_cache_is_rejected_before_serving(monkeypatch):
    pytest.importorskip("vllm")
    from ..v1_connector import KVConnectorBase_V1, XavierConnector

    def init_base(self, **kwargs):
        self._kv_transfer_config = SimpleNamespace(
            get_from_extra_config=lambda *args: {"role": "prefill"}
        )

    monkeypatch.setattr(KVConnectorBase_V1, "__init__", init_base)
    config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
    caches = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(mamba_cache_mode="all"))
        ]
    )
    with pytest.raises(ValueError, match="does not yet support hybrid/recurrent"):
        XavierConnector(config, None, caches)
