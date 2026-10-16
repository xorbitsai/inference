import base64
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from .. import load_model_family_from_json
from ..core import create_image_model_instance
from ..engine import DiffusersImageModel, TransformersIrisImageModel
from ..iris import IrisModel


@pytest.fixture(autouse=True)
def no_external_iris_package(monkeypatch):
    # All adapter and real sampling tests must work without the Git package.
    monkeypatch.setitem(sys.modules, "iris3b", None)


def _spec(turbo=False, hub="huggingface"):
    name = "iris-3b-turbo" if turbo else "iris-3b"
    families = {}
    path = Path(__file__).parents[1] / "models" / f"{name}.json"
    load_model_family_from_json(str(path), families)
    return next(spec for spec in families[name] if spec.model_hub == hub)


@pytest.mark.parametrize("turbo", [False, True])
def test_iris_catalog_and_default_engine(monkeypatch, tmp_path, turbo):
    from ..core import BUILTIN_IMAGE_MODELS
    from ..engine_family import IMAGE_ENGINES, generate_engine_config_by_model_name

    specs = [_spec(turbo, hub) for hub in ("huggingface", "modelscope")]
    for spec in specs:
        assert spec.model_id == "speridlabs/iris-3b"
        assert spec.model_revision == (
            "main" if spec.model_hub == "huggingface" else "master"
        )
        assert spec.model_ability == ["text2image"]
        assert not any("git+" in package for package in spec.virtualenv.packages)
        assert any("omegaconf>=2.3" in package for package in spec.virtualenv.packages)
        assert TransformersIrisImageModel.match(spec)
        assert not DiffusersImageModel.match(spec)
        expected = (
            ["turbo/config.yaml", "turbo/distill.yaml", "turbo/model.safetensors"]
            if turbo
            else ["config.yaml", "model.safetensors"]
        )
        assert spec.cache_config["allow_patterns"] == expected
    name = specs[0].model_name
    monkeypatch.setitem(BUILTIN_IMAGE_MODELS, name, specs)
    engines = {}
    generate_engine_config_by_model_name(specs[0], engines)
    monkeypatch.setitem(IMAGE_ENGINES, name, engines[name])
    assert set(engines[name]) == {"transformers"}
    model = create_image_model_instance(
        "custom-iris-uid",
        name,
        model_path=str(tmp_path),
        download_hub="modelscope",
        enable_virtual_env=False,
    )
    assert isinstance(model, TransformersIrisImageModel)
    assert model.model_spec.model_hub == "modelscope"
    assert model.model_spec.model_engine == "transformers"


@pytest.fixture
def loaded_model():
    pytest.importorskip("omegaconf")
    model = IrisModel("iris", "/unused", _spec(), device="cpu")
    model._model = object()
    model._text_encoder = object()
    model._config = SimpleNamespace(
        model=SimpleNamespace(patch_size=16),
        flow=SimpleNamespace(num_train_timesteps=1000, prediction="v", shift=4.0),
        sample=SimpleNamespace(
            steps=100, order=2, cfg_scale=3.0, cfg_interval=[0.0, 1.0]
        ),
    )
    return model


def test_iris_sampling_parameters_seeds_and_pixels(monkeypatch, loaded_model):
    from ....thirdparty.iris3b import sampling

    calls = []

    def generate(model, encoder, prompts, **kwargs):
        calls.append((prompts, kwargs))
        return torch.stack([torch.full((3, 32, 48), -1.0)])

    monkeypatch.setattr(sampling, "generate", generate)
    result = loaded_model.text_to_image(
        "a cat",
        n=2,
        size="48x32",
        seed=[13, 17],
        negative_prompt="blurry",
        num_inference_steps=20,
        guidance_scale=4.5,
        response_format="b64_json",
    )
    assert len(result["data"]) == 2
    for (_, call), seed in zip(calls, (13, 17)):
        assert call["generator"].initial_seed() == seed
        assert call["steps"] == 20
        assert call["cfg_scale"] == 4.5
        assert call["negative_prompt"] == "blurry"
        assert (call["width"], call["height"]) == (48, 32)
        assert call["order"] == 2
        assert call["shift"] == 4.0
    image = Image.open(BytesIO(base64.b64decode(result["data"][0]["b64_json"])))
    assert image.size == (48, 32)
    assert image.getpixel((0, 0)) == (0, 0, 0)
    assert loaded_model.model_spec.default_generate_config["num_inference_steps"] == 100


def test_iris_turbo_preserves_checkpoint_schedule(monkeypatch, loaded_model):
    from ....thirdparty.iris3b import sampling

    loaded_model._model_spec = _spec(True)
    loaded_model._sigmas = [1.0, 0.938, 0.833, 0.625]
    calls = []

    def generate_few_step(model, encoder, prompts, **kwargs):
        calls.append(kwargs)
        return torch.ones(1, 3, 16, 32)

    monkeypatch.setattr(
        sampling, "generate", lambda *a, **k: pytest.fail("base sampler used")
    )
    monkeypatch.setattr(sampling, "generate_few_step", generate_few_step)
    result = loaded_model.text_to_image(
        "a cat",
        n=2,
        size="32*16",
        seed=42,
        response_format="b64_json",
    )
    assert len(result["data"]) == 2
    assert [call["generator"].initial_seed() for call in calls] == [42, 43]
    for call in calls:
        assert call["sigmas"] == loaded_model._sigmas
        assert not set(call) & {
            "steps",
            "order",
            "cfg_scale",
            "shift",
            "negative_prompt",
        }


@pytest.mark.parametrize(
    "options",
    [
        {"num_inference_steps": 8},
        {"guidance_scale": 3.0},
        {"negative_prompt": "blurry"},
        {"order": 2},
        {"shift": 4.0},
        {"sigmas": [1.0, 0.5]},
        {"cfg_interval": [0.0, 1.0]},
    ],
)
def test_iris_turbo_rejects_conflicting_parameters(loaded_model, options):
    loaded_model._model_spec = _spec(True)
    loaded_model._sigmas = [1.0, 0.938, 0.833, 0.625]
    with pytest.raises(ValueError, match="Iris-3B Turbo"):
        loaded_model.text_to_image("a cat", **options)


@pytest.mark.parametrize(
    "options",
    [
        {"n": 0},
        {"n": True},
        {"seed": -2},
        {"seed": True},
        {"size": "1000*1024"},
        {"size": "0x16"},
        {"size": "garbage"},
        {"width": 15},
        {"height": -16},
    ],
)
def test_iris_rejects_invalid_requests(loaded_model, options):
    with pytest.raises(ValueError):
        loaded_model.text_to_image("a cat", **options)


@pytest.mark.parametrize("turbo", [False, True])
@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
def test_iris_load_and_real_sampling_with_tiny_checkpoint(
    monkeypatch, tmp_path, turbo, hub
):
    """Exercise exported weights, bundled IrisDiT and both real samplers on CPU."""
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf
    from safetensors.torch import save_file

    from ....thirdparty.iris3b.config import ModelConfig, PixelStageConfig
    from ....thirdparty.iris3b.models import IrisDiT
    from ....thirdparty.iris3b.registry import TEXT_ENCODERS
    from ....thirdparty.iris3b.text.base import TextEncoding
    from ...cache_manager import CacheManager

    cfg = ModelConfig(
        hidden_size=32,
        depth=2,
        dual_depth=1,
        num_heads=4,
        num_kv_heads=2,
        patch_size=4,
        text_dim=16,
        text_len=8,
        text_adapter="linear",
        repa_layer=0,
        pixel=PixelStageConfig(
            depth=1,
            hidden_size=8,
            attn_hidden_size=32,
            num_heads=4,
        ),
    )
    path = tmp_path / "turbo" if turbo else tmp_path
    path.mkdir(exist_ok=True)
    OmegaConf.save(OmegaConf.structured({"model": cfg}), path / "config.yaml")
    save_file(IrisDiT(cfg).state_dict(), path / "model.safetensors")
    if turbo:
        OmegaConf.save(
            OmegaConf.create({"dmd": {"sigmas": [1.0, 0.938, 0.833, 0.625]}}),
            path / "distill.yaml",
        )
    encoder_dir = tmp_path / "encoder"
    encoder_dir.mkdir()
    downloads = []

    def cache_encoder(manager):
        downloads.append(manager._model_family)
        return str(encoder_dir)

    monkeypatch.setattr(CacheManager, "cache", cache_encoder)

    class TinyEncoder:
        def encode(self, prompts):
            return TextEncoding(
                torch.zeros(len(prompts), 8, 16),
                torch.ones(len(prompts), 8, dtype=torch.long),
            )

        def null(self, negative_prompt):
            return self.encode([negative_prompt])

    encoder_config = []

    def build_encoder(name, cfg, device):
        encoder_config.append(cfg)
        return TinyEncoder()

    monkeypatch.setattr(TEXT_ENCODERS, "build", build_encoder)
    model = IrisModel("iris", str(tmp_path), _spec(turbo, hub), device="cpu")
    model.load()
    assert model._model is not None
    assert next(model._model.parameters()).dtype == torch.float32
    assert encoder_config[0].pretrained == str(encoder_dir)
    assert encoder_config[0].dtype == "float32"
    assert downloads[0].model_id == "Qwen/Qwen3-VL-4B-Instruct"
    assert downloads[0].model_hub == hub
    assert downloads[0].model_revision == ("main" if hub == "huggingface" else "master")
    options = {} if turbo else {"num_inference_steps": 2}
    result = model.text_to_image(
        "a cat", size="16*16", seed=123, response_format="b64_json", **options
    )
    repeated = model.text_to_image(
        "a cat", size="16*16", seed=123, response_format="b64_json", **options
    )
    assert result["data"] == repeated["data"]
    image = Image.open(BytesIO(base64.b64decode(result["data"][0]["b64_json"])))
    assert image.size == (16, 16)
