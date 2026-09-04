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
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache, LlamaConfig

from .. import utils as transformers_utils
from ..core import PytorchModel
from ..utils import (
    convert_to_cache_cls,
    get_batch_size_and_seq_len_from_kv_cache,
    get_kv_cache_layer,
)

_HAS_LAYER_BASED_CACHE = hasattr(DynamicCache(), "layers")


class _LegacyDynamicCache:
    """The DynamicCache layout used by Transformers 4.46 through 4.52."""

    def __init__(self, key_cache, value_cache):
        self.key_cache = key_cache
        self.value_cache = value_cache

    def __len__(self):
        return len(self.key_cache)

    def get_seq_length(self, layer_idx=0):
        if layer_idx >= len(self.key_cache):
            return 0
        key = self.key_cache[layer_idx]
        if key is None or (isinstance(key, (list, tuple)) and not key):
            return 0
        return key.shape[2]


def _model_for_cache_tests(config=None):
    model = object.__new__(PytorchModel)
    model._model = SimpleNamespace(config=config)
    return model


def _cache_with_tensor(value, batch_size=1, seq_len=3, config=None):
    cache = DynamicCache(config=config) if config is not None else DynamicCache()
    key = torch.full((batch_size, 2, seq_len, 4), value, dtype=torch.float32)
    cache.update(key, key.clone(), 0)
    return cache


def test_get_batch_size_supports_legacy_and_layer_cache_layouts():
    key = torch.zeros((2, 3, 5, 7))
    legacy_cache = _LegacyDynamicCache([key], [key.clone()])
    model = _model_for_cache_tests()

    assert get_batch_size_and_seq_len_from_kv_cache(legacy_cache, model) == (2, 5)

    layer_cache = _cache_with_tensor(1, batch_size=3, seq_len=4)
    assert get_batch_size_and_seq_len_from_kv_cache(layer_cache, model) == (3, 4)

    skipped_key = torch.zeros((4, 3, 6, 7))
    cache_with_skipped_layer = DynamicCache()
    cache_with_skipped_layer.update(skipped_key, skipped_key.clone(), 1)
    assert get_batch_size_and_seq_len_from_kv_cache(
        cache_with_skipped_layer, model
    ) == (4, 6)

    legacy_cache_with_skipped_layer = _LegacyDynamicCache(
        [[], skipped_key], [[], skipped_key.clone()]
    )
    assert get_batch_size_and_seq_len_from_kv_cache(
        legacy_cache_with_skipped_layer, model
    ) == (4, 6)


def test_convert_to_cache_cls_supports_tuple_and_list_legacy_cache():
    key = torch.ones((2, 2, 3, 4))
    value = torch.ones((2, 2, 3, 4))

    for legacy_cache in (((key, value),), [(key, value)]):
        converted = convert_to_cache_cls(legacy_cache)

        assert isinstance(converted, DynamicCache)
        converted_key, converted_value = get_kv_cache_layer(converted, 0)
        torch.testing.assert_close(converted_key, key)
        torch.testing.assert_close(converted_value, value)


def test_convert_to_cache_cls_falls_back_to_update(monkeypatch):
    key = torch.ones((2, 2, 3, 4))
    value = torch.ones((2, 2, 3, 4))

    class _DynamicCache:
        def __init__(self):
            self.layers = []

        def update(self, key_states, value_states, layer_idx):
            assert layer_idx == len(self.layers)
            self.layers.append((key_states, value_states))

    monkeypatch.setattr(transformers_utils, "DynamicCache", _DynamicCache)

    converted = transformers_utils.convert_to_cache_cls(((key, value),))

    assert isinstance(converted, _DynamicCache)
    converted_key, converted_value = converted.layers[0]
    torch.testing.assert_close(converted_key, key)
    torch.testing.assert_close(converted_value, value)


def test_empty_cache_layers_are_reported_without_indexing_failures():
    model = _model_for_cache_tests()
    assert get_batch_size_and_seq_len_from_kv_cache(DynamicCache(), model) == (0, 0)
    assert get_batch_size_and_seq_len_from_kv_cache(
        _LegacyDynamicCache([[]], [[]]), model
    ) == (0, 0)


def test_merge_and_reduce_legacy_dynamic_cache_layout():
    model = _model_for_cache_tests()
    past_key = torch.ones((1, 2, 3, 4))
    new_key = torch.full((1, 2, 2, 4), 2, dtype=torch.float32)
    past_cache = _LegacyDynamicCache([past_key], [past_key.clone()])
    new_cache = _LegacyDynamicCache([new_key], [new_key.clone()])

    merged = model.merge_kv_cache(past_cache, new_cache)
    merged_key, _ = get_kv_cache_layer(merged, 0)
    assert merged_key.shape == (2, 2, 3, 4)

    model.build_reduced_kv_cache(merged, {0})
    reduced_key, _ = get_kv_cache_layer(merged, 0)
    torch.testing.assert_close(reduced_key, torch.ones((1, 2, 3, 4)))


def test_merge_and_reduce_layer_based_dynamic_cache():
    model = _model_for_cache_tests()
    past_cache = _cache_with_tensor(1, seq_len=3)
    new_cache = _cache_with_tensor(2, seq_len=2)

    merged = model.merge_kv_cache(past_cache, new_cache)
    key, value = get_kv_cache_layer(merged, 0)

    assert key.shape == value.shape == (2, 2, 3, 4)
    torch.testing.assert_close(key[0, :, 0], torch.zeros((2, 4)))
    torch.testing.assert_close(
        key[0, :, 1:], torch.full((2, 2, 4), 2, dtype=torch.float32)
    )
    torch.testing.assert_close(key[1], torch.ones((2, 3, 4)))

    reduced = model.build_reduced_kv_cache(merged, {0})
    reduced_key, _ = get_kv_cache_layer(reduced, 0)
    torch.testing.assert_close(reduced_key, torch.ones((1, 2, 3, 4)))


def test_merge_mixed_legacy_and_layer_cache_layouts():
    model = _model_for_cache_tests()
    past_key = torch.ones((1, 2, 3, 4))
    past_cache = _LegacyDynamicCache([past_key], [past_key.clone()])
    new_cache = _cache_with_tensor(2, seq_len=2)

    merged = model.merge_kv_cache(past_cache, new_cache)
    merged_key, _ = get_kv_cache_layer(merged, 0)

    assert merged_key.shape == (2, 2, 3, 4)
    torch.testing.assert_close(merged_key[0, :, 0], torch.zeros((2, 4)))
    torch.testing.assert_close(
        merged_key[0, :, 1:], torch.full((2, 2, 4), 2, dtype=torch.float32)
    )
    torch.testing.assert_close(merged_key[1], past_key[0])


@pytest.mark.skipif(
    not _HAS_LAYER_BASED_CACHE,
    reason="Layer-based DynamicCache was introduced after Transformers 4.52",
)
def test_merge_preserves_sliding_cache_state_and_shared_layers():
    config = LlamaConfig(num_hidden_layers=1, sliding_window=4)
    model = _model_for_cache_tests(config)
    past_cache = _cache_with_tensor(1, seq_len=3, config=config)
    new_cache = _cache_with_tensor(2, seq_len=2, config=config)
    past_cache.shared_layers = {1: (torch.ones((1, 2, 3, 4)), torch.ones((1, 2, 3, 4)))}
    new_cache.shared_layers = {
        1: (
            torch.full((1, 2, 2, 4), 2, dtype=torch.float32),
            torch.full((1, 2, 2, 4), 2, dtype=torch.float32),
        )
    }

    merged = model.merge_kv_cache(past_cache, new_cache)

    assert merged.layers[0].is_sliding
    assert merged.layers[0].cumulative_length == 3
    shared_key, _ = merged.shared_layers[1]
    assert shared_key.shape == (2, 2, 3, 4)

    model.build_reduced_kv_cache(merged, {1})
    shared_key, _ = merged.shared_layers[1]
    assert shared_key.shape == (1, 2, 3, 4)
    torch.testing.assert_close(
        shared_key[:, :, 1:],
        torch.full((1, 2, 2, 4), 2, dtype=torch.float32),
    )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not _HAS_LAYER_BASED_CACHE,
    reason="requires at least two CUDA devices and layer-based DynamicCache",
)
def test_dynamic_cache_reduction_supports_layers_on_multiple_cuda_devices():
    model = _model_for_cache_tests()
    cache = DynamicCache()
    cache.update(
        torch.ones((3, 2, 4, 4), device="cuda:0"),
        torch.ones((3, 2, 4, 4), device="cuda:0"),
        0,
    )
    cache.update(
        torch.ones((3, 2, 4, 4), device="cuda:1"),
        torch.ones((3, 2, 4, 4), device="cuda:1"),
        1,
    )

    # Exercise the historical failure mode: without an explicit CPU device,
    # torch.tensor() follows this non-CPU default and produces indices on
    # cuda:0, which cannot be used for the cache layer on cuda:1.
    with torch.device("cuda:0"):
        reduced = model.build_reduced_kv_cache(cache, {1})

    assert [layer.keys.device for layer in reduced.layers] == [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    ]
    assert all(layer.keys.shape[0] == 2 for layer in reduced.layers)
