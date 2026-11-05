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
import glob
import json
import os

import pytest

from ....constants import XINFERENCE_ENV_MODEL_SRC

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Repos that no longer exist on Hugging Face. The last three are still on
# ModelScope, so their specs keep their ModelScope source.
DELETED = {
    "mlx-community/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen-1_8B-Chat",
    "Skywork/Skywork-OR1-32B-Preview",
    "Skywork/Skywork-OR1-7B-Preview",
}

# PyTorch variants that lost their Hugging Face source and stay on other hubs.
ONLY_ON_OTHER_HUBS = [
    ("qwen-chat", "1_8"),
    ("skywork-or1-preview", 7),
    ("skywork-or1-preview", 32),
]


@pytest.fixture
def default_hub_is_huggingface(monkeypatch):
    monkeypatch.setattr(
        "xinference.model.utils.is_locale_chinese_simplified", lambda: False
    )
    monkeypatch.setattr(
        "xinference.model.utils.auto_detect_download_hub", lambda: "huggingface"
    )


def _set_model_src(monkeypatch, model_src):
    if model_src is None:
        monkeypatch.delenv(XINFERENCE_ENV_MODEL_SRC, raising=False)
    else:
        monkeypatch.setenv(XINFERENCE_ENV_MODEL_SRC, model_src)


def test_no_spec_points_at_a_deleted_repo():
    found = []
    for path in glob.glob(os.path.join(MODEL_DIR, "*.json")):
        with open(path, encoding="utf-8") as f:
            for family in json.load(f):
                for spec in family["model_specs"]:
                    src = spec["model_src"].get("huggingface") or {}
                    if src.get("model_id") in DELETED:
                        found.append((family["model_name"], src["model_id"]))
    assert found == []


def test_llama_3_1_instruct_mlx_8b_unquantized_uses_the_bf16_repo():
    # Same naming as the 70B spec: mlx-community/Meta-Llama-3.1-70B-Instruct-bf16.
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name("llama-3.1-instruct")
    ids = {
        s.model_id
        for s in family.model_specs
        if s.model_format == "mlx"
        and s.model_size_in_billions == 8
        and s.quantization == "none"
    }
    assert ids == {"mlx-community/Meta-Llama-3.1-8B-Instruct-bf16"}


def test_specs_only_on_modelscope_keep_their_modelscope_source():
    ms_ids = set()
    for name in ("qwen-chat", "skywork-or1-preview"):
        with open(os.path.join(MODEL_DIR, f"{name}.json"), encoding="utf-8") as f:
            for family in json.load(f):
                for spec in family["model_specs"]:
                    src = spec["model_src"].get("modelscope") or {}
                    ms_ids.add(src.get("model_id"))
    assert {
        "qwen/Qwen-1_8B-Chat",
        "Skywork/Skywork-OR1-32B-Preview",
        "Skywork/Skywork-OR1-7B-Preview",
    } <= ms_ids


def test_specs_only_on_modelscope_still_get_a_cache_version_entry():
    from ..core import generate_llm_version_info
    from ..llm_family import get_llm_family_by_name

    qwen = generate_llm_version_info(get_llm_family_by_name("qwen-chat"))["qwen-chat"]
    skywork = generate_llm_version_info(get_llm_family_by_name("skywork-or1-preview"))[
        "skywork-or1-preview"
    ]
    versions = [v["model_version"] for v in qwen + skywork]
    assert {
        "qwen-chat--1_8B--pytorch--none",
        "skywork-or1-preview--7B--pytorch--none",
        "skywork-or1-preview--32B--pytorch--none",
    } <= set(versions)
    # A variant hosted on several hubs is still listed once.
    assert len(versions) == len(set(versions))


def test_every_builtin_variant_gets_a_cache_version_entry():
    from .. import BUILTIN_LLM_FAMILIES
    from ..core import generate_llm_version_info
    from ..utils import get_model_version

    assert BUILTIN_LLM_FAMILIES
    missing = []
    for family in BUILTIN_LLM_FAMILIES:
        recorded = {
            v["model_version"]
            for v in generate_llm_version_info(family)[family.model_name]
        }
        missing += [
            version
            for version in (
                get_model_version(
                    family.model_name,
                    s.model_format,
                    s.model_size_in_billions,
                    s.quantization,
                )
                for s in family.model_specs
            )
            if version not in recorded
        ]
    assert missing == []


def _cache_variant(name, size, model_format="pytorch"):
    from ..cache_manager import LLMCacheManager
    from ..llm_family import get_llm_family_by_name

    spec = next(
        s
        for s in get_llm_family_by_name(name).model_specs
        if s.model_format == model_format and str(s.model_size_in_billions) == str(size)
    )
    cache_dir = LLMCacheManager.get_cache_dir_for_spec(name, spec)
    os.makedirs(cache_dir)
    return cache_dir


@pytest.fixture
def empty_cache(monkeypatch, tmp_path):
    from .... import constants

    monkeypatch.setattr(constants, "XINFERENCE_CACHE_DIR", str(tmp_path))


@pytest.fixture
def stub_engine_without_downloads(monkeypatch):
    from .. import llm_family

    class Engine:
        def __init__(self, uid, family, path, config):
            self.model_family = family
            self.model_path = path

    def no_download(*args, **kwargs):
        pytest.fail("This launch must not download anything")

    monkeypatch.setattr(
        llm_family, "check_engine_by_spec_parameters", lambda *args: Engine
    )
    monkeypatch.setattr("xinference.model.utils.retry_download", no_download)
    monkeypatch.setattr("xinference.model.utils.retry_snapshot_download", no_download)


@pytest.mark.parametrize("model_src", [None, "auto", "huggingface"])
@pytest.mark.parametrize("name,size", ONLY_ON_OTHER_HUBS)
def test_default_launch_reuses_the_cache_of_a_variant_on_another_hub(
    monkeypatch,
    empty_cache,
    default_hub_is_huggingface,
    stub_engine_without_downloads,
    model_src,
    name,
    size,
):
    from ...core import create_model_instance

    _set_model_src(monkeypatch, model_src)
    cache_dir = _cache_variant(name, size)

    model = create_model_instance(
        "uid",
        "LLM",
        name,
        "Transformers",
        "pytorch",
        size,
        "none",
        enable_virtual_env=False,
    )

    assert model.model_path == cache_dir
    assert model.model_family.model_specs[0].model_hub == "modelscope"


@pytest.mark.parametrize("model_src", [None, "auto", "huggingface", "modelscope"])
def test_cached_deepseek_v3_1_gptq_is_still_launched_listed_and_deleted(
    monkeypatch,
    empty_cache,
    default_hub_is_huggingface,
    stub_engine_without_downloads,
    model_src,
):
    # Its repo is gone from both hubs, so a cold launch fails at download as it
    # did before. The spec stays so that caches people already have keep working.
    from ...core import create_model_instance
    from ..utils import get_model_version

    _set_model_src(monkeypatch, model_src)
    cache_dir = _cache_variant("Deepseek-V3.1", 671, "gptq")

    model = create_model_instance(
        "uid",
        "LLM",
        "Deepseek-V3.1",
        "Transformers",
        "gptq",
        671,
        "Int4",
        enable_virtual_env=False,
    )

    assert model.model_path == cache_dir
    _assert_listed_and_deletable(
        "Deepseek-V3.1", get_model_version("Deepseek-V3.1", "gptq", 671, "Int4")
    )


@pytest.mark.parametrize("download_hub", [None, "auto", "huggingface"])
@pytest.mark.parametrize("name,size", ONLY_ON_OTHER_HUBS)
def test_local_launch_of_a_variant_on_another_hub_uses_the_supplied_path(
    monkeypatch,
    empty_cache,
    default_hub_is_huggingface,
    stub_engine_without_downloads,
    tmp_path,
    download_hub,
    name,
    size,
):
    from ...core import create_model_instance

    _set_model_src(monkeypatch, None)
    local_path = tmp_path / "weights"
    local_path.mkdir()

    model = create_model_instance(
        "uid",
        "LLM",
        name,
        "Transformers",
        "pytorch",
        size,
        "none",
        download_hub=download_hub,
        model_path=str(local_path),
        enable_virtual_env=False,
    )

    assert model.model_path == str(local_path)
    assert model.model_family.model_specs[0].model_hub == "modelscope"
    assert os.listdir(tmp_path) == ["weights"]


@pytest.mark.parametrize(
    "name,size,quantization,download_hub,detected_hub,tokenizer_hub",
    [
        ("qwen2.5-instruct", 7, "q4_k_m", "auto", "modelscope", "modelscope"),
        ("qwen2.5-instruct", 7, "q4_k_m", "auto", "huggingface", "huggingface"),
        # Skywork's GGUF files are on Hugging Face, its PyTorch repos only on ModelScope.
        ("skywork-or1-preview", 7, "Q4_K_M", "auto", "huggingface", "modelscope"),
        ("skywork-or1-preview", 32, "Q4_K_M", "auto", "huggingface", "modelscope"),
        ("skywork-or1-preview", 7, "Q4_K_M", "huggingface", None, "modelscope"),
        ("skywork-or1-preview", 32, "Q4_K_M", "huggingface", None, "modelscope"),
        ("skywork-or1-preview", 7, "Q4_K_M", "modelscope", None, "modelscope"),
        ("skywork-or1-preview", 32, "Q4_K_M", "modelscope", None, "modelscope"),
    ],
)
def test_a_local_gguf_takes_its_tokenizer_from_the_selected_hub_first(
    monkeypatch,
    empty_cache,
    stub_engine_without_downloads,
    tmp_path,
    name,
    size,
    quantization,
    download_hub,
    detected_hub,
    tokenizer_hub,
):
    from ...core import create_model_instance
    from .. import llm_family
    from ..vllm import core as vllm_core

    tokenizer_specs = []

    def fetch_tokenizer(family):
        tokenizer_specs.append(family.model_specs[0])
        return "tokenizer-dir"

    _set_model_src(monkeypatch, None)
    monkeypatch.setattr(
        "xinference.model.utils.is_locale_chinese_simplified", lambda: False
    )
    monkeypatch.setattr(
        "xinference.model.utils.auto_detect_download_hub", lambda: detected_hub
    )
    monkeypatch.setattr(vllm_core, "cache_model_tokenizer_and_config", fetch_tokenizer)
    monkeypatch.setattr(
        llm_family, "check_engine_by_spec_parameters", lambda *args: vllm_core.VLLMModel
    )
    gguf_file = tmp_path / "model.gguf"
    gguf_file.touch()

    model = create_model_instance(
        "uid",
        "LLM",
        name,
        "vLLM",
        "ggufv2",
        size,
        quantization,
        download_hub=download_hub,
        model_path=str(gguf_file),
        enable_virtual_env=False,
    )
    model._preprocess_load_gguf()

    assert model.model_path == str(gguf_file)
    assert [
        (spec.model_hub, spec.model_size_in_billions, spec.quantization)
        for spec in tokenizer_specs
    ] == [(tokenizer_hub, size, "none")]
    assert model._model_config["tokenizer"] == "tokenizer-dir"
    assert model._model_config["hf_config_path"] == "tokenizer-dir"


@pytest.mark.parametrize("name,size", ONLY_ON_OTHER_HUBS)
def test_a_cold_cache_does_not_download_from_a_hub_nobody_selected(
    monkeypatch, empty_cache, default_hub_is_huggingface, name, size
):
    from ...core import create_model_instance
    from ..llm_family import match_llm

    _set_model_src(monkeypatch, None)
    assert match_llm(name, "pytorch", size, "none") is None
    assert match_llm(name, "pytorch", size, "none", "huggingface") is None
    with pytest.raises(ValueError, match="Model not found"):
        create_model_instance(
            "uid", "LLM", name, "Transformers", "pytorch", size, "none"
        )

    selected = match_llm(name, "pytorch", size, "none", "modelscope")
    assert selected.model_specs[0].model_hub == "modelscope"


def test_the_selected_hub_wins_over_a_cached_copy(
    monkeypatch, empty_cache, default_hub_is_huggingface
):
    from ..llm_family import match_llm

    _set_model_src(monkeypatch, None)
    _cache_variant("qwen-chat", 7)

    selected = match_llm("qwen-chat", "pytorch", 7, "none")
    assert selected.model_specs[0].model_hub == "huggingface"
    assert match_llm("qwen-chat").model_specs[0].model_hub == "huggingface"


def _assert_listed_and_deletable(name, version):
    from ....core.cache_tracker import CacheTrackerActor
    from ..core import generate_llm_version_info
    from ..llm_family import get_llm_family_by_name

    tracker = CacheTrackerActor.__new__(CacheTrackerActor)
    tracker._model_name_to_version_info = {}
    tracker.record_model_version(
        generate_llm_version_info(get_llm_family_by_name(name)), "worker"
    )

    tracker.update_cache_status("worker", name, version, "/cache/path")

    assert version in {
        v["model_version"] for v in tracker.list_cached_models("worker", name)
    }
    assert tracker.list_deletable_models(version, "worker") == "/cache/path"


@pytest.mark.parametrize("name,size", ONLY_ON_OTHER_HUBS)
def test_download_from_another_hub_is_listed_and_deletable(name, size):
    from ..utils import get_model_version

    _assert_listed_and_deletable(name, get_model_version(name, "pytorch", size, "none"))
