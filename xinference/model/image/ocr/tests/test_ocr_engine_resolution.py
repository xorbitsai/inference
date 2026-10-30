# CI: linux

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

from pathlib import Path

import pytest

from xinference.core.utils import filter_virtualenv_packages_by_markers

from ... import load_model_family_from_json
from ...core import (
    BUILTIN_IMAGE_MODELS,
    create_image_model_instance,
    resolve_image_model_engine,
)
from ...engine_family import IMAGE_ENGINES
from .. import register_builtin_ocr_engines
from ..ocr_family import OCR_ENGINES, generate_engine_config_by_model_name


@pytest.mark.parametrize(
    "engines, kwargs, expected",
    [
        ({"transformers": [], "vllm": []}, {}, "transformers"),
        ({"vllm": [], "transformers": []}, {}, "transformers"),
        ({"deepdoc": []}, {}, "deepdoc"),
        ({"transformers": [], "vllm": []}, {"model_engine": ""}, "transformers"),
        ({"transformers": [], "vllm": []}, {"model_engine": "vllm"}, "vllm"),
        (
            {"transformers": [], "llama.cpp": []},
            {"model_format": "ggufv2"},
            "llama.cpp",
        ),
        (
            {"transformers": [], "llama.cpp": []},
            {"gguf_quantization": "Q4_K_M"},
            "llama.cpp",
        ),
        (
            {"transformers": [], "llama.cpp": []},
            {"gguf_model_path": "/models/ocr.gguf"},
            "llama.cpp",
        ),
    ],
)
def test_resolve_ocr_model_engine(monkeypatch, engines, kwargs, expected):
    monkeypatch.setitem(OCR_ENGINES, "test-ocr", engines)

    assert resolve_image_model_engine("test-ocr", **kwargs) == expected


def test_resolve_image_model_engine_keeps_other_models_unchanged(monkeypatch):
    monkeypatch.setitem(IMAGE_ENGINES, "test-image", {"diffusers": [], "vLLM": []})

    assert resolve_image_model_engine("test-image") == "diffusers"
    assert resolve_image_model_engine("test-image", model_format="ggufv2") == (
        "diffusers"
    )
    assert resolve_image_model_engine("test-image", "vllm") == "vLLM"
    assert resolve_image_model_engine("test-image", "sglang") == "sglang"
    assert resolve_image_model_engine("unregistered-image") is None
    assert resolve_image_model_engine("unregistered-image", "vllm") == "vllm"


def _builtin_ocr_families():
    families = {}
    load_model_family_from_json(str(Path(__file__).parents[2] / "models"), families)
    return {
        name: specs
        for name, specs in families.items()
        if any("ocr" in (spec.model_ability or []) for spec in specs)
    }


def test_builtin_ocr_default_engine_matches_created_model(monkeypatch, tmp_path):
    families = _builtin_ocr_families()
    register_builtin_ocr_engines()
    engines = {}
    for specs in families.values():
        for spec in specs:
            generate_engine_config_by_model_name(spec, engines)
    for name, specs in families.items():
        monkeypatch.setitem(BUILTIN_IMAGE_MODELS, name, specs)
        monkeypatch.setitem(OCR_ENGINES, name, engines[name])

    launches_with_guarded_packages = 0
    for name, specs in sorted(families.items()):
        # MLX specs need an explicit engine, so they have no default to check.
        model_formats = {getattr(spec, "model_format", None) for spec in specs}
        for model_format in sorted(model_formats - {"mlx"}, key=str):
            engine = resolve_image_model_engine(name, model_format=model_format)
            model = create_image_model_instance(
                f"{name}-test",
                name,
                download_hub="huggingface",
                model_path=str(tmp_path),
                model_format=model_format,
                enable_virtual_env=False,
            )
            assert model.model_family.model_engine == engine, (name, model_format)

            # A launch without an engine dropped every `#engine#`-guarded
            # package; the resolved engine keeps the ones meant for it.
            virtualenv = model.model_family.virtualenv
            packages = virtualenv.packages if virtualenv else []
            without_engine = filter_virtualenv_packages_by_markers(packages, None, None)
            with_engine = filter_virtualenv_packages_by_markers(packages, engine, None)
            assert set(without_engine) <= set(with_engine), (name, model_format)
            if len(with_engine) > len(without_engine):
                launches_with_guarded_packages += 1

    assert launches_with_guarded_packages > 0
