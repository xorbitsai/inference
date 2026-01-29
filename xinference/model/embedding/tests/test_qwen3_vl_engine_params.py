# Copyright 2022-2026 XProbe Inc.
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

import os
import site
import sys

import pytest
import torch

from xinference.constants import XINFERENCE_VIRTUAL_ENV_DIR
from xinference.core import worker as worker_mod
from xinference.core.worker import WorkerActor
from xinference.model.embedding import _install
from xinference.model.embedding.cache_manager import EmbeddingCacheManager
from xinference.model.embedding.core import create_embedding_model_instance
from xinference.model.embedding.embed_family import match_embedding
from xinference.model.utils import (
    check_dependency_available,
    get_engine_params_by_name_with_virtual_env,
)


def _assert_engine_params(params, engine_name):
    assert engine_name in params
    assert isinstance(params[engine_name], list)
    assert params[engine_name], f"{engine_name} params should not be empty"
    item = params[engine_name][0]
    assert item.get("model_format") == "pytorch"
    assert item.get("quantization") == "none"


def test_qwen3_vl_embedding_engine_params_with_virtualenv():
    _install()
    params = get_engine_params_by_name_with_virtual_env(
        "embedding", "Qwen3-VL-Embedding-2B", enable_virtual_env=True
    )
    assert isinstance(params, dict)
    _assert_engine_params(params, "sentence_transformers")
    _assert_engine_params(params, "vllm")


def _get_cached_model_path():
    family = match_embedding("Qwen3-VL-Embedding-2B", "pytorch", "none")
    cache_manager = EmbeddingCacheManager(family)
    return cache_manager.cache()


def _get_virtualenv_site_packages(env_path: str) -> str:
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if os.name == "nt":
        return os.path.join(env_path, "Lib", "site-packages")
    return os.path.join(env_path, "lib", f"python{py_version}", "site-packages")


def _purge_modules(prefixes):
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            sys.modules.pop(name, None)


def _prepare_engine_virtualenv(engine_name: str):
    family = match_embedding("Qwen3-VL-Embedding-2B", "pytorch", "none")
    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    env_path = os.path.join(
        XINFERENCE_VIRTUAL_ENV_DIR,
        "v4",
        family.model_name,
        engine_name.lower(),
        py_version,
    )
    manager = WorkerActor._create_virtual_env_manager(True, None, env_path)
    assert manager is not None
    previous_skip_installed = worker_mod.XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED
    worker_mod.XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED = False
    WorkerActor._prepare_virtual_env(
        manager, family.virtualenv, virtual_env_packages=None, model_engine=engine_name
    )
    worker_mod.XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED = previous_skip_installed
    site_packages = _get_virtualenv_site_packages(env_path)
    if os.path.isdir(site_packages):
        site.addsitedir(site_packages)
        if site_packages in sys.path:
            sys.path.remove(site_packages)
            sys.path.insert(0, site_packages)
    return env_path


def test_qwen3_vl_embedding_sentence_transformers_startup_virtualenv():
    if not torch.cuda.is_available():
        pytest.skip("Qwen3-VL embedding startup tests require GPU")
    _install()
    _prepare_engine_virtualenv("sentence_transformers")
    _purge_modules(["transformers", "qwen_vl_utils", "PIL"])
    for module_name in ("transformers", "qwen_vl_utils", "PIL"):
        dep_check = check_dependency_available(module_name, module_name)
        if dep_check != True:
            pytest.skip(dep_check[1])
    model_path = _get_cached_model_path()
    model = create_embedding_model_instance(
        "qwen3-vl-embedding-st",
        "Qwen3-VL-Embedding-2B",
        "sentence_transformers",
        model_format="pytorch",
        quantization="none",
        model_path=model_path,
        enable_virtual_env=True,
    )
    model.load()


def test_qwen3_vl_embedding_vllm_startup_virtualenv():
    if not torch.cuda.is_available():
        pytest.skip("Qwen3-VL embedding startup tests require GPU")
    _install()
    _prepare_engine_virtualenv("vllm")
    _purge_modules(["vllm"])
    try:
        import vllm
        from packaging.version import Version

        if Version(vllm.__version__) < Version("0.14.0"):
            pytest.skip("vLLM version is lower than 0.14.0")
    except Exception as exc:
        pytest.skip(f"Failed to import vLLM: {exc}")
    model_path = _get_cached_model_path()
    model = create_embedding_model_instance(
        "qwen3-vl-embedding-vllm",
        "Qwen3-VL-Embedding-2B",
        "vllm",
        model_format="pytorch",
        quantization="none",
        model_path=model_path,
        enable_virtual_env=True,
    )
    model.load()
