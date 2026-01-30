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
from xinference.model.rerank import _install
from xinference.model.rerank.cache_manager import RerankCacheManager
from xinference.model.rerank.core import create_rerank_model_instance
from xinference.model.rerank.rerank_family import match_rerank
from xinference.model.utils import check_dependency_available


def _get_cached_model_path():
    family = match_rerank("Qwen3-VL-Reranker-2B", "pytorch", "none")
    cache_manager = RerankCacheManager(family)
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


def _prepare_engine_virtualenv(engine_name: str, virtual_env_packages=None):
    family = match_rerank("Qwen3-VL-Reranker-2B", "pytorch", "none")
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
        manager,
        family.virtualenv,
        virtual_env_packages=virtual_env_packages,
        model_engine=engine_name,
    )
    worker_mod.XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED = previous_skip_installed
    site_packages = _get_virtualenv_site_packages(env_path)
    if os.path.isdir(site_packages):
        site.addsitedir(site_packages)
        if site_packages in sys.path:
            sys.path.remove(site_packages)
            sys.path.insert(0, site_packages)
    return env_path


def test_qwen3_vl_reranker_sentence_transformers_startup_virtualenv():
    if not torch.cuda.is_available():
        pytest.skip("Qwen3-VL reranker startup tests require GPU")
    _install()
    _prepare_engine_virtualenv(
        virtual_env_packages=[
            "transformers>=4.57.0",
            "qwen-vl-utils>=0.0.14",
            "Pillow",
            "scipy",
        ],
    )
    _purge_modules(["transformers", "qwen_vl_utils", "PIL", "scipy"])
    for module_name, friendly in (
        ("transformers", "transformers"),
        ("qwen_vl_utils", "qwen-vl-utils"),
        ("PIL", "Pillow"),
        ("scipy", "scipy"),
    ):
        dep_check = check_dependency_available(module_name, friendly)
        if dep_check != True:
            pytest.skip(dep_check[1])
    model_path = _get_cached_model_path()
    model = create_rerank_model_instance(
        "qwen3-vl-reranker-st",
        "Qwen3-VL-Reranker-2B",
        "sentence_transformers",
        model_format="pytorch",
        quantization="none",
        model_path=model_path,
        enable_virtual_env=True,
    )
    model.load()
