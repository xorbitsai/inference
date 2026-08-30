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

from types import SimpleNamespace

import pytest
from packaging import version


def _model(
    *,
    architecture="Qwen4ExpForConditionalGeneration",
    gpu_count=8,
    n_worker=1,
    backend="auto",
    model_config=None,
    xavier=False,
):
    from ..core import VLLMModel

    model = object.__new__(VLLMModel)
    model.model_uid = "test-model-0"
    model.model_family = SimpleNamespace(
        accelerators=[str(index) for index in range(gpu_count)],
        has_architecture=lambda candidate: candidate == architecture,
        _resolve_architectures=lambda: [architecture],
    )
    model._model_config = dict(model_config or {})
    model._xinference_vllm_executor_backend = backend
    model._n_worker = n_worker
    model._device_count = gpu_count
    model._xavier_config = {} if xavier else None
    return model


@pytest.fixture(autouse=True)
def native_mp_capable_vllm(monkeypatch):
    from .. import core

    monkeypatch.setattr(core, "VLLM_VERSION", version.parse("0.28.0"))
    monkeypatch.setattr(core.VLLMModel, "_is_vllm_v1", lambda self: True)


def test_qwen4_exp_single_worker_multi_gpu_auto_uses_native_mp():
    model = _model()

    assert model._native_mp_route() == (
        True,
        "Qwen4Exp single-worker multi-GPU auto route",
    )
    assert model.need_create_pools is False
    assert model._get_native_mp_parallelism() == (8, 1)


def test_qwen4_exp_single_gpu_keeps_existing_route():
    model = _model(gpu_count=1)

    use_native_mp, reason = model._native_mp_route()

    assert use_native_mp is False
    assert reason == "allocated GPU count is not greater than 1"
    assert model.need_create_pools is True


def test_qwen4_exp_multi_worker_keeps_xoscar_route():
    model = _model(n_worker=2)

    assert model._native_mp_route() == (False, "n_worker is not 1")
    assert model.need_create_pools is True


def test_non_qwen_auto_keeps_xoscar_route():
    model = _model(architecture="LlamaForCausalLM")

    assert model._native_mp_route() == (
        False,
        "auto route is limited to Qwen4Exp architecture",
    )
    assert model.need_create_pools is True


def test_explicit_native_mp_supports_other_eligible_architecture():
    model = _model(architecture="LlamaForCausalLM", backend="native_mp")

    assert model._native_mp_route() == (True, "explicit native_mp backend")
    assert model.need_create_pools is False


def test_explicit_xoscar_always_keeps_existing_route():
    model = _model(backend="xoscar")

    assert model._native_mp_route() == (False, "explicit xoscar backend")
    assert model.need_create_pools is True


@pytest.mark.parametrize("backend", ["invalid", "ray", "mp"])
def test_invalid_xinference_executor_selector_is_rejected(backend):
    model = _model(backend=backend)

    with pytest.raises(ValueError, match="must be one of"):
        model._native_mp_route()


def test_launch_parameter_takes_priority_over_environment(monkeypatch):
    monkeypatch.setenv("XINFERENCE_VLLM_EXECUTOR_BACKEND", "native_mp")
    model = _model(backend="xoscar")

    assert model._get_xinference_executor_backend() == "xoscar"


def test_environment_selector_is_used_when_launch_parameter_is_absent(monkeypatch):
    monkeypatch.setenv("XINFERENCE_VLLM_EXECUTOR_BACKEND", "native_mp")
    model = _model(backend=None)

    assert model._get_xinference_executor_backend() == "native_mp"


def test_explicit_native_mp_rejects_multi_worker():
    model = _model(backend="native_mp", n_worker=2)

    with pytest.raises(ValueError, match="only supports n_worker=1"):
        model._native_mp_route()


def test_explicit_native_mp_rejects_single_gpu():
    model = _model(backend="native_mp", gpu_count=1)

    with pytest.raises(ValueError, match="more than one allocated GPU"):
        model._native_mp_route()


def test_explicit_native_mp_rejects_xavier():
    model = _model(backend="native_mp", xavier=True)

    with pytest.raises(ValueError, match="not supported with Xavier"):
        model._native_mp_route()


def test_auto_route_rejects_xavier_without_failing_launch():
    model = _model(xavier=True)

    assert model._native_mp_route() == (False, "Xavier is enabled")


def test_auto_route_keeps_existing_path_when_vllm_version_unavailable(monkeypatch):
    from .. import core

    monkeypatch.setattr(core, "VLLM_VERSION", None)
    model = _model()

    assert model._native_mp_route() == (False, "vLLM version is unavailable")


def test_explicit_native_mp_rejects_unavailable_vllm_version(monkeypatch):
    from .. import core

    monkeypatch.setattr(core, "VLLM_VERSION", None)
    model = _model(backend="native_mp")

    with pytest.raises(ValueError, match="version is unavailable"):
        model._native_mp_route()


def test_auto_route_keeps_existing_path_for_old_vllm(monkeypatch):
    from .. import core

    monkeypatch.setattr(core, "VLLM_VERSION", version.parse("0.27.0"))
    model = _model()

    use_native_mp, reason = model._native_mp_route()

    assert use_native_mp is False
    assert "lower than 0.28.0" in reason


def test_explicit_native_mp_requires_vllm_v1(monkeypatch):
    from .. import core

    monkeypatch.setattr(core.VLLMModel, "_is_vllm_v1", lambda self: False)
    model = _model(backend="native_mp")

    with pytest.raises(ValueError, match="requires vLLM V1"):
        model._native_mp_route()


@pytest.mark.parametrize(
    ("tensor_parallel_size", "pipeline_parallel_size"),
    [(8, 1), (4, 2), (2, 4)],
)
def test_native_mp_accepts_parallelism_covering_all_allocated_gpus(
    tensor_parallel_size, pipeline_parallel_size
):
    model = _model(
        model_config={
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
        }
    )

    assert model._get_native_mp_parallelism() == (
        tensor_parallel_size,
        pipeline_parallel_size,
    )


@pytest.mark.parametrize(
    ("tensor_parallel_size", "pipeline_parallel_size"),
    [(4, 1), (8, 2)],
)
def test_native_mp_rejects_parallelism_not_matching_allocated_gpus(
    tensor_parallel_size, pipeline_parallel_size
):
    model = _model(
        model_config={
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
        }
    )

    with pytest.raises(ValueError, match="to equal the allocated GPU count"):
        model._get_native_mp_parallelism()
