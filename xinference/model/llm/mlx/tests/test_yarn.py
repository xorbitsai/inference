# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.rope_utils import YarnRoPE
from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding

from ..yarn import apply_auto_yarn


def parameters(factor):
    return {
        "rope_type": "yarn",
        "rope_theta": 10000000.0,
        "original_max_position_embeddings": 262144,
        "factor": factor,
    }


class TinyModel(nn.Module):
    def __init__(self, rotary, vision=False):
        super().__init__()
        self.args = SimpleNamespace(rope_scaling=None)
        if vision:
            self.rotary_emb = rotary
        else:
            self.rope = rotary


@pytest.mark.parametrize("dims", [8, 64])
@pytest.mark.parametrize("factor", [2.0, 3.0, 4.0])
def test_text_rotary_matches_upstream_yarn_beyond_native_window(dims, factor):
    model = TinyModel(nn.RoPE(dims, traditional=False, base=10000000.0))
    x = mx.arange(2 * (dims + 8), dtype=mx.float32).reshape(1, 1, 2, dims + 8) / 100
    original = model.rope(x, offset=262145)
    apply_auto_yarn(model, parameters(factor))
    actual = model.rope(x, offset=262145)
    reference = YarnRoPE(
        dims,
        base=10000000.0,
        scaling_factor=factor,
        original_max_position_embeddings=262144,
        max_position_embeddings=int(262144 * factor),
    )
    expected = reference(x, offset=262145)
    mx.eval(actual, expected, original)
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-5)
    assert not np.allclose(np.array(actual)[..., :dims], np.array(original)[..., :dims])
    np.testing.assert_array_equal(np.array(actual)[..., dims:], np.array(x)[..., dims:])
    assert model.args.rope_scaling["factor"] == factor


@pytest.mark.parametrize("factor", [2.0, 4.0])
def test_multimodal_rotary_preserves_position_axes_and_attention_scale(factor):
    native = Qwen3_5RotaryEmbedding(64, base=10000000.0, mrope_section=[11, 11, 10])
    model = TinyModel(native, vision=True)
    x = mx.ones((1, 2, 3, 80), dtype=mx.float32)
    position_ids = mx.array(
        [[[262145, 262146, 262147]], [[17, 18, 19]], [[31, 32, 33]]]
    )
    original_cos, _ = native(x, position_ids)
    apply_auto_yarn(model, parameters(factor))
    actual_cos, actual_sin = model.rotary_emb(x, position_ids)
    reference = YarnRoPE(
        64,
        base=10000000.0,
        scaling_factor=factor,
        original_max_position_embeddings=262144,
    )
    expected_rotary = Qwen3_5RotaryEmbedding(
        64, base=10000000.0, mrope_section=[11, 11, 10]
    )
    if isinstance(getattr(type(expected_rotary), "inv_freq", None), property):
        expected_rotary._inv_freq = 1 / reference._freqs
    else:
        expected_rotary.inv_freq = 1 / reference._freqs
    cos, sin = expected_rotary(x, position_ids)
    mx.eval(actual_cos, actual_sin, original_cos, cos, sin)
    np.testing.assert_allclose(
        np.array(actual_cos), np.array(cos * reference.mscale), atol=1e-5
    )
    np.testing.assert_allclose(
        np.array(actual_sin), np.array(sin * reference.mscale), atol=1e-5
    )
    assert not np.allclose(np.array(actual_cos), np.array(original_cos))
    q, k = model.rotary_emb.apply_rotary(x, x, position_ids)
    mx.eval(q, k)
    np.testing.assert_array_equal(np.array(q)[..., 64:], np.array(x)[..., 64:])
    np.testing.assert_array_equal(np.array(q), np.array(k))
    # Reapplying at a load boundary must not compound frequency or magnitude scaling.
    apply_auto_yarn(model, parameters(factor))
    again, _ = model.rotary_emb(x, position_ids)
    np.testing.assert_array_equal(np.array(again), np.array(actual_cos))


def test_vision_encoder_rotary_is_not_changed():
    language = TinyModel(nn.RoPE(8, base=10000000.0))
    vision = TinyModel(nn.RoPE(8, base=10000.0))
    model = nn.Module()
    model.language_model = language
    model.vision_tower = vision
    before = vision.rope
    apply_auto_yarn(model, parameters(4))
    assert vision.rope is before
    assert isinstance(language.rope, YarnRoPE)


def test_unknown_rotary_is_rejected_instead_of_silently_ignoring_yarn():
    with pytest.raises(ValueError, match="no compatible MLX RoPE"):
        apply_auto_yarn(TinyModel(object()), parameters(4))


@pytest.mark.parametrize("automatic", [False, True])
def test_vision_loader_applies_automatic_and_checkpoint_yarn(
    monkeypatch, tmp_path, automatic
):
    import json

    import mlx_vlm

    from ..core import MLXVisionModel

    rotary = Qwen3_5RotaryEmbedding(64, base=10000000.0, mrope_section=[11, 11, 10])
    loaded = nn.Module()
    loaded.language_model = TinyModel(rotary, vision=True)
    data = {
        "text_config": {
            "max_position_embeddings": 262144,
            "rope_parameters": parameters(4),
        }
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(data))
    before = config.read_bytes()
    adapter = object.__new__(MLXVisionModel)
    adapter.model_path = str(tmp_path)
    adapter.model_family = SimpleNamespace(
        context_length=262144, has_architecture=lambda *a: True
    )
    adapter._model_config = {}
    adapter._auto_yarn_parameters = parameters(4) if automatic else None
    monkeypatch.setattr(mlx_vlm, "load", lambda path: (loaded, "processor"))
    model, processor = adapter._load_model()
    assert processor == "processor"
    x = mx.ones((1, 1, 1, 64))
    cos, _ = model.language_model.rotary_emb(x, mx.array([[0]]))
    # At position zero the phase is unchanged; only YaRN's magnitude can
    # produce this value, proving the loader did more than change a limit.
    np.testing.assert_allclose(np.array(cos), 1 + 0.1 * np.log(4), atol=1e-6)
    assert config.read_bytes() == before
