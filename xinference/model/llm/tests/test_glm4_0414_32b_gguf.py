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

# CI: linux

import os

import pytest

# Single-file GGUFs in bartowski/THUDM_GLM-4-32B-0414-GGUF (Hugging Face and
# the ModelScope mirror).
GLM_4_32B_FILES = {
    f"THUDM_GLM-4-32B-0414-{q}.gguf"
    for q in (
        "IQ2_M", "IQ2_S", "IQ2_XS", "IQ3_M", "IQ3_XS", "IQ3_XXS", "IQ4_NL",
        "IQ4_XS", "Q2_K", "Q2_K_L", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q3_K_XL",
        "Q4_0", "Q4_1", "Q4_K_L", "Q4_K_M", "Q4_K_S", "Q5_K_L", "Q5_K_M",
        "Q5_K_S", "Q6_K", "Q6_K_L", "Q8_0",
    )
}  # fmt: skip


def test_glm4_0414_32b_gguf_downloads_the_32b_model():
    from ...utils import generate_model_file_names_with_quantization_parts
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name("glm4-0414")
    specs = [
        s
        for s in family.model_specs
        if s.model_format == "ggufv2" and s.model_size_in_billions == 32
    ]
    pytorch = next(
        s
        for s in family.model_specs
        if s.model_format == "pytorch" and s.model_size_in_billions == 32
    )
    assert specs
    for spec in specs:
        assert spec.model_id == "bartowski/THUDM_GLM-4-32B-0414-GGUF"
        names, _, _ = generate_model_file_names_with_quantization_parts(spec)
        assert set(names) <= GLM_4_32B_FILES
        for field in ("num_hidden_layers", "hidden_size", "num_attention_heads"):
            assert getattr(spec.model_metadata, field) == getattr(
                pytorch.model_metadata, field
            )


def _glm4_32b_family(hub):
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name("glm4-0414").copy()
    family.model_specs = [
        s
        for s in family.model_specs
        if s.model_format == "ggufv2"
        and s.model_size_in_billions == 32
        and s.model_hub == hub
        and s.quantization == "Q4_K_M"
    ]
    assert len(family.model_specs) == 1
    return family


def _old_32b_cache(family, tmp_path, monkeypatch):
    """What the pre-fix spec left behind: its directory, holding the 9B file."""
    from .... import constants
    from ..cache_manager import LLMCacheManager

    monkeypatch.setattr(constants, "XINFERENCE_CACHE_DIR", str(tmp_path / "cache"))
    cache_dir = LLMCacheManager.get_cache_dir_for_spec(
        family.model_name, family.model_specs[0]
    )
    os.makedirs(cache_dir)
    with open(os.path.join(cache_dir, "THUDM_GLM-4-9B-0414-Q4_K_M.gguf"), "wb"):
        pass
    return cache_dir


@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
def test_old_9b_cache_downloads_the_32b_file(hub, tmp_path, monkeypatch):
    import huggingface_hub
    from modelscope.hub import file_download as modelscope_file_download

    from ... import utils
    from ..cache_manager import LLMCacheManager
    from ..gguf_model_metadata import resolve_gguf_model_path

    family = _glm4_32b_family(hub)
    spec = family.model_specs[0]
    old_cache_dir = _old_32b_cache(family, tmp_path, monkeypatch)
    blob = tmp_path / "blob.gguf"
    blob.write_bytes(b"GGUF")
    fetched = []

    def download(model_id, filename, revision=None):
        fetched.append((model_id, filename))
        return str(blob)

    monkeypatch.setattr(utils, "IS_NEW_HUGGINGFACE_HUB", True)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(modelscope_file_download, "model_file_download", download)

    expected_file = "THUDM_GLM-4-32B-0414-Q4_K_M.gguf"
    manager = LLMCacheManager(family)
    assert not manager.get_cache_status()

    cache_dir = manager.cache()

    assert cache_dir == old_cache_dir
    assert fetched == [("bartowski/THUDM_GLM-4-32B-0414-GGUF", expected_file)]
    loader_path = resolve_gguf_model_path(cache_dir, spec)
    assert os.path.basename(loader_path) == expected_file
    assert os.path.realpath(loader_path) == os.path.realpath(blob)
    assert manager.get_cache_status()

    manager.cache()
    assert len(fetched) == 1


def test_gguf_cache_status_keeps_a_legacy_model_bin(tmp_path, monkeypatch):
    from ..cache_manager import LLMCacheManager

    family = _glm4_32b_family("huggingface")
    cache_dir = _old_32b_cache(family, tmp_path, monkeypatch)
    assert not LLMCacheManager(family).get_cache_status()

    with open(os.path.join(cache_dir, "model.bin"), "wb"):
        pass
    assert LLMCacheManager(family).get_cache_status()


def test_gguf_cache_status_does_not_look_inside_a_model_uri_cache(
    tmp_path, monkeypatch
):
    from ..cache_manager import LLMCacheManager

    family = _glm4_32b_family("huggingface")
    monkeypatch.setattr(family.model_specs[0], "model_uri", "file:///models/glm4")
    _old_32b_cache(family, tmp_path, monkeypatch)
    assert LLMCacheManager(family).get_cache_status()
