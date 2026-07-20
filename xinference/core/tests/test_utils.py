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

from ..utils import (
    build_replica_model_uid,
    build_subpool_envs_for_virtual_env,
    iter_replica_model_uid,
    parse_replica_model_uid,
)
from ..virtual_env_manager import (
    ENGINE_VIRTUALENV_PACKAGES,
    XLLAMACPP_CUDA_INDEX_URLS,
    get_xllamacpp_cuda_index_url,
)


def test_replica_model_uid():
    all_gen_ids = []
    for replica_model_uid in iter_replica_model_uid("abc", 5):
        rebuild_replica_model_uid = build_replica_model_uid(
            *parse_replica_model_uid(replica_model_uid)
        )
        assert rebuild_replica_model_uid == replica_model_uid
        all_gen_ids.append(replica_model_uid)
    assert len(all_gen_ids) == 5
    assert len(set(all_gen_ids)) == 5


class DummyVirtualEnvManager:
    def __init__(self, python_path: str):
        self._python_path = python_path

    def get_python_path(self) -> str:
        return self._python_path


def test_sentence_transformers_virtualenv_packages_include_accelerate():
    pkgs = ENGINE_VIRTUALENV_PACKAGES.get("sentence_transformers", [])
    pkg_names = [p.split(">=")[0].split("==")[0].strip() for p in pkgs]
    assert (
        "accelerate" in pkg_names
    ), "accelerate must be in sentence_transformers venv packages"
    # torchvision is no longer in ENGINE_VIRTUALENV_PACKAGES;
    # it is supplied per-model via #system_torchvision# in model_spec.


def test_get_xllamacpp_cuda_index_url():
    cu132 = XLLAMACPP_CUDA_INDEX_URLS["cu132"]
    cu128 = XLLAMACPP_CUDA_INDEX_URLS["cu128"]

    # CUDA 13.x -> cu132
    assert get_xllamacpp_cuda_index_url("13.2") == cu132
    assert get_xllamacpp_cuda_index_url("13.0") == cu132
    # Major-only version strings are handled too.
    assert get_xllamacpp_cuda_index_url("13") == cu132
    # A future CUDA major line still maps to the latest available wheel index.
    assert get_xllamacpp_cuda_index_url("14.0") == cu132
    assert get_xllamacpp_cuda_index_url("14") == cu132

    # CUDA 12.8+ -> cu128
    assert get_xllamacpp_cuda_index_url("12.8") == cu128
    assert get_xllamacpp_cuda_index_url("12.9") == cu128

    # Older CUDA 12 lines have no prebuilt wheel -> fall back to CPU (None).
    # A major-only "12" is treated as 12.0, which is below the cu128 cutoff.
    assert get_xllamacpp_cuda_index_url("12") is None
    assert get_xllamacpp_cuda_index_url("12.6") is None
    assert get_xllamacpp_cuda_index_url("12.1") is None
    assert get_xllamacpp_cuda_index_url("11.8") is None

    # No CUDA detected / unparsable -> None.
    assert get_xllamacpp_cuda_index_url(None) is None
    assert get_xllamacpp_cuda_index_url("") is None
    assert get_xllamacpp_cuda_index_url("unknown") is None


def _run_prepare_virtual_env(
    cuda_version, inherited_index_url, skip_installed=True, cuda_available=True
):
    """
    Drive WorkerActor._prepare_virtual_env for the llama.cpp engine and report
    what happened: the settings handed to install_packages(), and which
    packages were uninstalled first. CUDA detection, real device availability
    and pip-config inheritance are mocked so the test does not depend on the
    host.

    ``skip_installed`` mirrors XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED, which
    defaults to on. ``cuda_available`` mirrors whether a CUDA device is actually
    usable (distinct from the PyTorch-build CUDA version).
    """
    import contextlib
    from unittest import mock

    from ...model.core import VirtualEnvSettings
    from ..worker import WorkerActor

    result = {"conf": None, "packages": None, "uninstalled": []}

    class _FakeVEM:
        env_path = "/tmp/fake_venv"

        def install_packages(self, packages, **conf):
            result["packages"] = packages
            result["conf"] = conf

    @contextlib.contextmanager
    def _nullctx(*args, **kwargs):
        yield

    from .. import worker as worker_mod

    pip_config = {"index_url": inherited_index_url} if inherited_index_url else {}
    with mock.patch.object(
        worker_mod, "get_pip_config_args", return_value=pip_config
    ), mock.patch(
        "xoscar.virtualenv.platform.get_cuda_version", return_value=cuda_version
    ), mock.patch.object(
        worker_mod, "_exclusive_venv_path_lock", new=_nullctx
    ), mock.patch.object(
        worker_mod, "XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED", skip_installed
    ), mock.patch.object(
        WorkerActor, "_is_cuda_device_available", return_value=cuda_available
    ), mock.patch.object(
        WorkerActor,
        "_uninstall_venv_package",
        side_effect=lambda _vem, pkg: result["uninstalled"].append(pkg),
    ):
        WorkerActor._prepare_virtual_env(
            virtual_env_manager=_FakeVEM(),
            settings=VirtualEnvSettings(packages=["xllamacpp>=0.2.6"]),
            virtual_env_packages=None,
            model_engine="llama.cpp",
            model_name="qwen3",
            architectures=None,
        )
    return result


def test_prepare_virtual_env_llama_cpp_gpu_uses_exclusive_index():
    # On a GPU host with an inherited PyPI mirror, xllamacpp must install from
    # the CUDA-matched GPU index exclusively -- the mirror is dropped so the
    # resolver cannot satisfy the same-version CPU wheel from PyPI.
    result = _run_prepare_virtual_env(
        cuda_version="12.8",
        inherited_index_url="https://mirrors.cloud.tencent.com/pypi/simple/",
    )
    conf = result["conf"]
    assert conf["index_url"] == XLLAMACPP_CUDA_INDEX_URLS["cu128"]
    assert not conf.get("extra_index_url")


def test_prepare_virtual_env_llama_cpp_gpu_force_reinstalls_over_skip_installed():
    # Even with skip-installed on (the default), a GPU launch must force the GPU
    # wheel in: the pre-existing (possibly CPU) xllamacpp is uninstalled first
    # and skip_installed is turned off for this install, so uv is actually
    # invoked against the GPU index instead of the requirement being filtered
    # out as already-satisfied.
    result = _run_prepare_virtual_env(
        cuda_version="12.8",
        inherited_index_url="https://mirrors.cloud.tencent.com/pypi/simple/",
        skip_installed=True,
    )
    assert result["uninstalled"] == ["xllamacpp"]
    assert result["conf"]["skip_installed"] is False
    assert result["conf"]["index_url"] == XLLAMACPP_CUDA_INDEX_URLS["cu128"]


def test_prepare_virtual_env_llama_cpp_cpu_keeps_inherited_mirror():
    # On a CPU host (no CUDA) the inherited mirror is kept and the CPU wheel is
    # installed normally, without any forced uninstall.
    result = _run_prepare_virtual_env(
        cuda_version=None,
        inherited_index_url="https://mirrors.cloud.tencent.com/pypi/simple/",
    )
    assert (
        result["conf"]["index_url"] == "https://mirrors.cloud.tencent.com/pypi/simple/"
    )
    assert result["uninstalled"] == []


def test_prepare_virtual_env_llama_cpp_no_gpu_device_keeps_cpu_index():
    # get_cuda_version() reports the PyTorch build (e.g. "13.0") even on a
    # CPU-only host with a CUDA-built torch. When no CUDA device is actually
    # available, the GPU wheel would fail to import (missing libcuda.so.1), so
    # keep the inherited index and do not force a reinstall.
    result = _run_prepare_virtual_env(
        cuda_version="13.0",
        inherited_index_url="https://mirrors.cloud.tencent.com/pypi/simple/",
        cuda_available=False,
    )
    assert (
        result["conf"]["index_url"] == "https://mirrors.cloud.tencent.com/pypi/simple/"
    )
    assert result["uninstalled"] == []


def test_build_subpool_envs_for_virtual_env_disabled():
    base_envs = {"PATH": "/usr/bin", "FLASHINFER_NINJA_PATH": "/custom/ninja"}
    result = build_subpool_envs_for_virtual_env(base_envs, False, None)

    assert result == base_envs
    assert result is not base_envs


def test_build_subpool_envs_for_virtual_env_enabled():
    manager = DummyVirtualEnvManager("/venv/bin/python")
    base_envs = {"PATH": "/usr/bin", "FLASHINFER_NINJA_PATH": "/custom/ninja"}

    result = build_subpool_envs_for_virtual_env(base_envs, True, manager)

    import os

    assert result["PATH"] == "/venv/bin" + os.pathsep + "/usr/bin"
    assert result["VIRTUAL_ENV"] == "/venv"
    assert result["FLASHINFER_NINJA_PATH"] == "/custom/ninja"
    assert result is not base_envs


def test_model_specs_pin_system_torch_with_torchvision():
    """Every model that pins torchvision to the system version must also pin
    torch to the system version under the same environment markers/conditions.

    Otherwise torch is pulled fresh from PyPI into the sub venv while
    torchvision stays on the (older) system version, producing an ABI
    mismatch such as ``operator torchvision::nms does not exist`` (see #5208).
    """
    import json
    import os

    here = os.path.dirname(__file__)
    spec_files = [
        os.path.join(here, "..", "..", "model", "embedding", "model_spec.json"),
        os.path.join(here, "..", "..", "model", "rerank", "model_spec.json"),
    ]

    offenders = []
    for spec_file in spec_files:
        with open(spec_file) as f:
            data = json.load(f)
        for m in data:
            pkgs = (m.get("virtualenv") or {}).get("packages") or []
            parsed_pkgs = []
            for p in pkgs:
                parts = p.split(";", 1)
                name = parts[0].strip()
                marker = parts[1].strip() if len(parts) > 1 else ""
                parsed_pkgs.append((name, marker))

            torchvision_markers = {
                marker for name, marker in parsed_pkgs if name == "#system_torchvision#"
            }
            torch_markers = {
                marker for name, marker in parsed_pkgs if name == "#system_torch#"
            }

            if torchvision_markers - torch_markers:
                family = os.path.basename(os.path.dirname(spec_file))
                offenders.append("{}/{}".format(family, m.get("model_name")))

    assert not offenders, (
        "Models declare #system_torchvision# but not #system_torch# under the "
        "same environment markers (torch/torchvision would be mixed-source): "
        + ", ".join(offenders)
    )
