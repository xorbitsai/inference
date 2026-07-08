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

"""Tests for flashinfer AOT post-install hook.

See optimize/20260702/2026070209.md for root cause analysis.
"""

import os
from unittest import mock

import pytest

from ..virtual_env_manager import (
    FLASHINFER_AOT_ARCHES,
    FLASHINFER_AOT_PACKAGES,
    FLASHINFER_AOT_WHEEL_URL,
    apply_flashinfer_aot_post_install,
    needs_flashinfer_aot,
)


class TestNeedsFlashinferAot:
    """Tests for needs_flashinfer_aot() gate logic."""

    def test_vllm_qwen3_5_moe_triggers(self):
        assert (
            needs_flashinfer_aot("vllm", ["Qwen3_5MoeForConditionalGeneration"], "13.0")
            is True
        )

    def test_vllm_qwen3_5_moe_case_insensitive_engine(self):
        assert (
            needs_flashinfer_aot("VLLM", ["Qwen3_5MoeForConditionalGeneration"], "13.0")
            is True
        )

    def test_vllm_multiple_archs_including_target(self):
        assert (
            needs_flashinfer_aot(
                "vllm",
                ["LlamaForCausalLM", "Qwen3_5MoeForConditionalGeneration"],
                "13.0",
            )
            is True
        )

    def test_non_cu130_cuda_skipped(self):
        """AOT packages are +cu130 only; CUDA 12.x must skip to avoid install failures."""
        assert (
            needs_flashinfer_aot("vllm", ["Qwen3_5MoeForConditionalGeneration"], "12.0")
            is False
        )

    def test_none_cuda_skipped(self):
        """Unknown CUDA version must skip — can't safely install +cu130 wheels."""
        assert (
            needs_flashinfer_aot("vllm", ["Qwen3_5MoeForConditionalGeneration"], None)
            is False
        )

    def test_non_vllm_engine_skipped(self):
        assert (
            needs_flashinfer_aot(
                "sglang", ["Qwen3_5MoeForConditionalGeneration"], "13.0"
            )
            is False
        )

    def test_non_target_arch_skipped(self):
        assert needs_flashinfer_aot("vllm", ["LlamaForCausalLM"], "13.0") is False

    def test_empty_architectures_skipped(self):
        assert needs_flashinfer_aot("vllm", [], "13.0") is False

    def test_none_architectures_skipped(self):
        assert needs_flashinfer_aot("vllm", None, "13.0") is False

    def test_none_engine_skipped(self):
        assert (
            needs_flashinfer_aot(None, ["Qwen3_5MoeForConditionalGeneration"], "13.0")
            is False
        )

    def test_empty_engine_skipped(self):
        assert (
            needs_flashinfer_aot("", ["Qwen3_5MoeForConditionalGeneration"], "13.0")
            is False
        )

    def test_constants_sanity(self):
        assert "Qwen3_5MoeForConditionalGeneration" in FLASHINFER_AOT_ARCHES
        assert len(FLASHINFER_AOT_PACKAGES) == 3
        assert any("flashinfer-jit-cache" in p for p in FLASHINFER_AOT_PACKAGES)
        assert "flashinfer.ai" in FLASHINFER_AOT_WHEEL_URL


class TestApplyFlashinferAotPostInstall:
    """Tests for apply_flashinfer_aot_post_install() behavior."""

    @pytest.fixture
    def fake_venv_manager(self):
        """Build a minimal fake virtual_env_manager with _get_uv_path and env_path."""
        m = mock.MagicMock()
        m._get_uv_path.return_value = "/fake/uv"
        m.env_path = "/fake/venv"
        return m

    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        """Auto-clean FLASHINFER_DISABLE_VERSION_CHECK before each test."""
        monkeypatch.delenv("FLASHINFER_DISABLE_VERSION_CHECK", raising=False)

    def test_skipped_for_non_target_arch(self, fake_venv_manager):
        """Non-target architecture should not invoke subprocess."""
        with mock.patch(
            "xinference.core.virtual_env_manager.subprocess.run"
        ) as run_mock:
            apply_flashinfer_aot_post_install(
                "vllm", ["LlamaForCausalLM"], fake_venv_manager, {}
            )
            run_mock.assert_not_called()

    def test_skipped_for_non_vllm_engine(self, fake_venv_manager):
        with mock.patch(
            "xinference.core.virtual_env_manager.subprocess.run"
        ) as run_mock:
            apply_flashinfer_aot_post_install(
                "sglang", ["Qwen3_5MoeForConditionalGeneration"], fake_venv_manager, {}
            )
            run_mock.assert_not_called()

    def test_success_no_env_var_set(self, fake_venv_manager):
        """Successful upgrade should NOT set FLASHINFER_DISABLE_VERSION_CHECK."""
        result = mock.MagicMock()
        result.returncode = 0
        with mock.patch(
            "xinference.core.virtual_env_manager.subprocess.run", return_value=result
        ) as run_mock:
            apply_flashinfer_aot_post_install(
                "vllm",
                ["Qwen3_5MoeForConditionalGeneration"],
                fake_venv_manager,
                {},
                "13.0",
            )
            run_mock.assert_called_once()
            cmd = run_mock.call_args[0][0]
            assert "--no-deps" in cmd
            assert "--upgrade" in cmd
            assert "flashinfer.ai" in " ".join(cmd)
            for pkg in FLASHINFER_AOT_PACKAGES:
                assert pkg in cmd
        assert "FLASHINFER_DISABLE_VERSION_CHECK" not in os.environ

    def test_failure_sets_fallback_env_var(self, fake_venv_manager):
        """Failed upgrade should set FLASHINFER_DISABLE_VERSION_CHECK=1."""
        result = mock.MagicMock()
        result.returncode = 1
        result.stderr = "network unreachable"
        with mock.patch(
            "xinference.core.virtual_env_manager.subprocess.run", return_value=result
        ):
            apply_flashinfer_aot_post_install(
                "vllm",
                ["Qwen3_5MoeForConditionalGeneration"],
                fake_venv_manager,
                {},
                "13.0",
            )
        assert os.environ.get("FLASHINFER_DISABLE_VERSION_CHECK") == "1"

    def test_exception_sets_fallback_env_var(self, fake_venv_manager):
        """Subprocess exception should set FLASHINFER_DISABLE_VERSION_CHECK=1."""
        with mock.patch(
            "xinference.core.virtual_env_manager.subprocess.run",
            side_effect=FileNotFoundError("uv not found"),
        ):
            apply_flashinfer_aot_post_install(
                "vllm",
                ["Qwen3_5MoeForConditionalGeneration"],
                fake_venv_manager,
                {},
                "13.0",
            )
        assert os.environ.get("FLASHINFER_DISABLE_VERSION_CHECK") == "1"

    def test_extra_index_url_merged_from_conf(self, fake_venv_manager):
        """conf['extra_index_url'] should be merged with flashinfer.ai URL."""
        result = mock.MagicMock()
        result.returncode = 0
        with mock.patch(
            "xinference.core.virtual_env_manager.subprocess.run", return_value=result
        ) as run_mock:
            apply_flashinfer_aot_post_install(
                "vllm",
                ["Qwen3_5MoeForConditionalGeneration"],
                fake_venv_manager,
                {"extra_index_url": ["https://wheels.vllm.ai/0.19.0/cu130"]},
                "13.0",
            )
            cmd = run_mock.call_args[0][0]
            cmd_str = " ".join(cmd)
            assert "wheels.vllm.ai" in cmd_str
            assert "flashinfer.ai" in cmd_str

    def test_extra_index_url_string_form(self, fake_venv_manager):
        """conf['extra_index_url'] as string should also be handled."""
        result = mock.MagicMock()
        result.returncode = 0
        with mock.patch(
            "xinference.core.virtual_env_manager.subprocess.run", return_value=result
        ) as run_mock:
            apply_flashinfer_aot_post_install(
                "vllm",
                ["Qwen3_5MoeForConditionalGeneration"],
                fake_venv_manager,
                {"extra_index_url": "https://wheels.vllm.ai/0.19.0/cu130"},
                "13.0",
            )
            cmd = run_mock.call_args[0][0]
            cmd_str = " ".join(cmd)
            assert "wheels.vllm.ai" in cmd_str
            assert "flashinfer.ai" in cmd_str

    def test_skipped_for_non_cu130_cuda(self, fake_venv_manager):
        """CUDA 12.x must skip — AOT packages are +cu130 only."""
        with mock.patch(
            "xinference.core.virtual_env_manager.subprocess.run"
        ) as run_mock:
            apply_flashinfer_aot_post_install(
                "vllm",
                ["Qwen3_5MoeForConditionalGeneration"],
                fake_venv_manager,
                {},
                "12.0",
            )
            run_mock.assert_not_called()
