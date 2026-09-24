from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from .. import load_model_family_from_json
from ..core import ImageModelFamilyV2
from ..engine import DiffusersImageModel, MingImageEngineModel
from ..ming_image import MingImageModel


@pytest.mark.parametrize(
    "name,abilities,cfg",
    [
        ("Ming-Image-0.1-Design", ["text2image", "image2image"], 1.0),
        ("Ming-Image-0.1-Design-Layer", ["image2image"], 2.0),
    ],
)
def test_ming_image_catalog(name, abilities, cfg):
    families = {}
    path = Path(__file__).parents[1] / "models" / f"{name}.json"
    load_model_family_from_json(str(path), families)
    specs = {spec.model_hub: spec for spec in families[name]}
    assert set(specs) == {"huggingface", "modelscope"}
    assert specs["huggingface"].model_revision == "main"
    assert specs["modelscope"].model_revision == "master"
    for spec in specs.values():
        assert spec.model_id == f"inclusionAI/{name}"
        assert spec.model_ability == abilities
        assert spec.default_generate_config["num_inference_steps"] == 12
        assert spec.default_generate_config["guidance_scale"] == cfg
        assert (
            'flash-attn==2.8.3 ; #engine# == "diffusers" and sys_platform == '
            '"linux" and platform_machine == "x86_64"' in spec.virtualenv.packages
        )
        assert MingImageEngineModel.match(spec)
        assert not DiffusersImageModel.match(spec)


def _spec(name):
    return ImageModelFamilyV2(
        model_family="ming_image",
        model_name=name,
        model_id=f"inclusionAI/{name}",
        model_revision="main",
        model_ability=(
            ["image2image"]
            if name.endswith("-Layer")
            else ["text2image", "image2image"]
        ),
        default_generate_config={
            "num_inference_steps": 12,
            "guidance_scale": 2.0 if name.endswith("-Layer") else 1.0,
        },
    )


def test_ming_image_load_uses_upstream_contract(monkeypatch, tmp_path):
    from .. import ming_image

    calls = {}

    class Profile:
        def validate_task(self, task, **kwargs):
            calls["task"] = task

    upstream = SimpleNamespace(
        load_checkpoint_capabilities=lambda path: Profile(),
        load_model_and_processor=lambda path, args: (
            calls.update(args=args) or object(),
            object(),
        ),
        _dtype=lambda name: getattr(torch, name),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def import_upstream(name):
        assert name == "xinference.thirdparty.ming_image.infer"
        return upstream

    monkeypatch.setattr(ming_image.importlib, "import_module", import_upstream)
    model = MingImageModel(
        "ming", str(tmp_path), _spec("Ming-Image-0.1-Design"), torch_dtype="bfloat16"
    )
    model.load()

    assert calls["task"] == "text-to-image"
    assert calls["args"].device == "cuda:0"
    assert calls["args"].attn_implementation == "eager"
    assert model._dtype == torch.bfloat16


@pytest.mark.parametrize(
    "available,expected",
    [(True, "flash_attention_2"), (False, "sdpa")],
)
def test_ming_image_attention_fallback_configures_both_submodels(
    monkeypatch, available, expected
):
    _require_ming_image_attention_dependencies()
    from xinference.thirdparty.ming_image import modeling_bailingmm2

    monkeypatch.setattr(
        modeling_bailingmm2, "is_flash_attn_2_available", lambda: available
    )
    vision_config = SimpleNamespace()
    llm_config = SimpleNamespace()
    assert (
        modeling_bailingmm2._configure_attn_implementation(vision_config, llm_config)
        == expected
    )
    assert vision_config._attn_implementation == expected
    assert llm_config._attn_implementation == expected


def test_ming_image_sdpa_attention_uses_torch_sdpa(monkeypatch):
    _require_ming_image_attention_dependencies()
    from xinference.thirdparty.ming_image import modeling_bailing_moe_v2 as modeling
    from xinference.thirdparty.ming_image.configuration_bailing_moe_v2 import (
        BailingMoeV2Config,
    )

    config = BailingMoeV2Config(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        head_dim=4,
        attention_dropout=0.0,
        rope_scaling={"type": "mrope"},
        _attn_implementation="sdpa",
    )
    attention_cls = modeling.ATTENTION_CLASSES["sdpa"]
    attention = attention_cls(config, layer_idx=0)
    monkeypatch.setattr(
        modeling,
        "apply_3d_rotary_pos_emb",
        lambda query, key, *args, **kwargs: (query, key),
    )
    sdpa = torch.nn.functional.scaled_dot_product_attention
    calls = []

    def track_sdpa(*args, **kwargs):
        calls.append(kwargs)
        return sdpa(*args, **kwargs)

    monkeypatch.setattr(modeling.F, "scaled_dot_product_attention", track_sdpa)
    output, weights, _ = attention(
        torch.randn(1, 3, 8), position_embeddings=(None, None)
    )

    assert output.shape == (1, 3, 8)
    assert weights is None
    assert calls == [
        {
            "attn_mask": None,
            "dropout_p": 0.0,
            "is_causal": True,
        }
    ]


def _require_ming_image_attention_dependencies():
    requirements = {"transformers": "4.57.1", "diffusers": "0.36.0"}
    for package, required_version in requirements.items():
        pytest.importorskip(package)
        installed_version = version(package)
        if installed_version != required_version:
            pytest.skip(
                f"Ming-Image attention tests require {package}=={required_version}; "
                f"found {installed_version}"
            )


@pytest.mark.parametrize(
    "name,expected_task,expected_count",
    [
        ("Ming-Image-0.1-Design", "image-edit", 1),
        ("Ming-Image-0.1-Design-Layer", "layer-decompose", 2),
    ],
)
def test_ming_image_edit_outputs(monkeypatch, name, expected_task, expected_count):
    from .. import ming_image

    calls = []
    profile = SimpleNamespace(
        resolve_sampling_parameters=lambda **kwargs: SimpleNamespace(**kwargs),
        validate_task=lambda *args, **kwargs: None,
    )

    def run_generation(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["input_image"].is_file()
        if expected_task == "layer-decompose":
            return [
                Image.new("RGB", (8, 8)),
                Image.new("RGBA", (8, 8)),
                Image.new("RGBA", (8, 8)),
            ]
        return [Image.new("RGB", (8, 8))]

    upstream = SimpleNamespace(
        resolve_task_resolution=lambda task, value: value or 1024,
        parse_num_layers=lambda prompt: 2,
        run_generation=run_generation,
    )
    monkeypatch.setattr(ming_image, "handle_image_result", lambda fmt, images: images)
    model = MingImageModel("ming", "/unused", _spec(name))
    model._model = object()
    model._processor = object()
    model._profile = profile
    model._infer = upstream
    model._dtype = torch.bfloat16

    images = model.image_to_image(
        Image.new("RGB", (8, 8)), "Decompose this image into 2 layers.", seed=7
    )
    assert len(images) == expected_count
    assert calls[0]["task"] == expected_task
    assert calls[0]["prompt"] == "Decompose this image into 2 layers."
    assert calls[0]["seed"] == 7
    assert not calls[0]["input_image"].exists()

    images = model.image_to_image(
        Image.new("RGB", (8, 8)), ["Decompose this image into 2 layers."], seed=7
    )
    assert len(images) == expected_count
    assert calls[1]["prompt"] == "Decompose this image into 2 layers."

    with pytest.raises(ValueError, match="exactly one prompt"):
        model.image_to_image(Image.new("RGB", (8, 8)), ["first", "second"])

    with pytest.raises(ValueError, match="exactly one reference image"):
        model.image_to_image(
            Image.new("RGB", (8, 8)),
            "edit",
            reference_images=[Image.new("RGB", (8, 8))],
        )
    with pytest.raises(ValueError, match="negative_prompt"):
        model.image_to_image(Image.new("RGB", (8, 8)), "edit", negative_prompt="bad")
    assert model.image_to_image(Image.new("RGB", (8, 8)), "edit", negative_prompt=" ")


def test_ming_image_text_to_image_uses_one_call_per_seed(monkeypatch):
    from .. import ming_image

    calls = []
    upstream = SimpleNamespace(
        resolve_task_resolution=lambda task, value: value,
        run_generation=lambda *args, **kwargs: (
            calls.append(kwargs) or [Image.new("RGBA", (8, 8))]
        ),
    )
    profile = SimpleNamespace(
        resolve_sampling_parameters=lambda **kwargs: SimpleNamespace(**kwargs),
        validate_task=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(ming_image, "handle_image_result", lambda fmt, images: images)
    model = MingImageModel("ming", "/unused", _spec("Ming-Image-0.1-Design"))
    model._model = object()
    model._processor = object()
    model._profile = profile
    model._infer = upstream
    model._dtype = torch.bfloat16

    images = model.text_to_image("a poster", n=2, size="2048x2048", seed=[3, 4])
    assert len(images) == 2
    assert [call["seed"] for call in calls] == [3, 4]
    assert all(call["task"] == "text-to-image" for call in calls)
    assert all(call["resolution"] == 2048 for call in calls)


def test_ming_image_vendored_inference_imports():
    from xinference.thirdparty.ming_image import infer

    assert callable(infer.load_checkpoint_capabilities)
    assert callable(infer.load_model_and_processor)
    assert callable(infer.run_generation)
