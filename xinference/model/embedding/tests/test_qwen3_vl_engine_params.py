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
    get_engine_params_by_name,
    get_engine_params_by_name_with_virtual_env,
)


def _assert_engine_params(params, engine_name):
    assert engine_name in params
    # Skip assertion if engine is unavailable (returns error message string)
    if isinstance(params[engine_name], str):
        return
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
    # vLLM is a declared virtualenv marker for this model, so it must be
    # offered even when vLLM is not installed locally (installed on demand).
    # This guards the regression where the vision match_json rejected the
    # engine with "vLLM is not installed" before the virtualenv exemption.
    assert isinstance(
        params["vllm"], list
    ), f"vLLM should be available under virtualenv, got {params['vllm']!r}"


def test_qwen3_vl_embedding_strict_with_old_vllm_when_virtualenv_disabled(
    monkeypatch,
):
    """Request-level enable_virtual_env=False must stay strict.

    Exercises the exact function the Worker calls on the disabled path
    (get_engine_params_by_name), not just the virtualenv-aware wrapper. Even
    when the process-global XINFERENCE_ENABLE_VIRTUAL_ENV default is true, a
    disabled discovery must not exempt the Qwen3-VL vLLM>=0.14.0 version check:
    with an old vLLM installed and no virtualenv, launching would fail, so vLLM
    must not be listed. When virtualenv is enabled it is offered again
    (installed on demand). Regression for the registry baking the exemption in
    at import time and the strict path reusing it.
    """
    import sys
    import types

    import xinference.constants as constants

    _install()

    # Global default true (as shipped), plus a simulated old local vLLM.
    monkeypatch.setattr(constants, "XINFERENCE_ENABLE_VIRTUAL_ENV", True)
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "0.11.2"
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    # Worker's disabled path.
    disabled = get_engine_params_by_name(
        "embedding", "Qwen3-VL-Embedding-2B", enable_virtual_env=False
    )
    assert not isinstance(
        disabled.get("vllm"), list
    ), f"vLLM must be strict (not available) with old vLLM and virtualenv disabled, got {disabled.get('vllm')!r}"

    # Worker's virtualenv-enabled path.
    enabled = get_engine_params_by_name_with_virtual_env(
        "embedding", "Qwen3-VL-Embedding-2B", enable_virtual_env=True
    )
    assert isinstance(
        enabled.get("vllm"), list
    ), f"vLLM should be available under virtualenv, got {enabled.get('vllm')!r}"


def test_qwen3_vl_embedding_prepared_install_list_has_single_vllm_req():
    """Only one vLLM requirement must reach the installer.

    The virtualenv spec must not carry both #vllm_dependencies# (vllm>=0.11.2)
    and vllm>=0.14.0: with skip_installed and a parent vLLM 0.11.2-0.13.x, the
    resolver pins the first (satisfied) requirement and then conflicts with the
    second. This asserts the *prepared install list* (expansion + marker
    filtering, i.e. the Worker path), not just discovery, contains exactly one
    vLLM requirement pinned to >=0.14.0.
    """
    from xinference.core.utils import filter_virtualenv_packages_by_markers
    from xinference.core.virtual_env_manager import (
        expand_engine_dependency_placeholders,
    )

    _install()
    family = match_embedding("Qwen3-VL-Embedding-2B", "pytorch", "none")
    expanded = expand_engine_dependency_placeholders(family.virtualenv.packages, "vllm")
    prepared = filter_virtualenv_packages_by_markers(expanded, "vllm", None)
    vllm_reqs = [p for p in prepared if p.lower().startswith("vllm")]
    assert vllm_reqs == [
        "vllm>=0.14.0"
    ], f"expected a single vllm>=0.14.0 requirement, got {vllm_reqs!r}"


def test_qwen3_vl_embedding_format_check_not_bypassed_under_virtualenv(monkeypatch):
    """virtualenv must not bypass format/prefix compatibility checks.

    The virtualenv exemption only covers the missing-library / old-version
    rejection; an incompatible model_format (e.g. ggufv2, which the vLLM
    embedding engine does not support) must still be rejected even when vLLM is
    absent and virtualenv is enabled, since virtualenv cannot make an
    incompatible spec work. Regression for the early ``return True``.
    """
    import sys

    from xinference.model.embedding.vllm.core import VLLMEmbeddingModel
    from xinference.model.utils import virtualenv_discovery_var

    _install()

    # vLLM absent + virtualenv enabled for this discovery scope.
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    token = virtualenv_discovery_var.set(True)
    try:
        family = match_embedding("Qwen3-VL-Embedding-2B", "pytorch", "none")
        good_spec = family.model_specs[0]
        assert good_spec.model_format == "pytorch"

        # Derive an incompatible spec by flipping the format to ggufv2.
        if hasattr(good_spec, "model_copy"):
            bad_spec = good_spec.model_copy(update={"model_format": "ggufv2"})
        else:
            bad_spec = good_spec.copy(update={"model_format": "ggufv2"})

        result = VLLMEmbeddingModel.match_json(family, bad_spec, "none")
        assert result is not True, "ggufv2 must not be accepted by vLLM embedding"
        assert isinstance(result, tuple) and result[0] is False

        assert VLLMEmbeddingModel.match_json(family, good_spec, "none") is True
    finally:
        virtualenv_discovery_var.reset(token)


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


def _prepare_engine_virtualenv(engine_name: str, virtual_env_packages=None):
    family = match_embedding("Qwen3-VL-Embedding-2B", "pytorch", "none")
    settings = family.virtualenv
    if virtual_env_packages:
        # If the caller pins numpy, drop any system numpy placeholder to avoid conflicts.
        if any(pkg.lower().startswith("numpy") for pkg in virtual_env_packages):
            if settings is not None:
                if hasattr(settings, "model_copy"):
                    settings = settings.model_copy(deep=True)
                else:
                    settings = settings.copy(deep=True)
                assert settings is not None  # for mypy type narrowing
                settings.packages = [
                    pkg
                    for pkg in settings.packages
                    if not pkg.strip().startswith("#system_numpy#")
                ]
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
        settings,
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
