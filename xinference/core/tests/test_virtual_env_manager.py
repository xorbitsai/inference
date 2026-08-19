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

import http.server
import importlib.metadata
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from ..virtual_env_manager import (
    FLASHINFER_AOT_ARCHES,
    FLASHINFER_AOT_PACKAGES,
    FLASHINFER_AOT_WHEEL_URL,
    FLASHINFER_CUBIN_WHEEL_URL,
    _run_uv_install_with_source_fallback,
    apply_flashinfer_aot_post_install,
    build_uv_source_options,
    ensure_flashinfer_cubin_matches_post_install,
    ensure_sglang_inherited_packages_compatible_post_install,
    get_engine_critical_dependency_specs,
    merge_virtual_env_find_links,
    needs_flashinfer_aot,
    validate_virtual_env_find_links,
)


def _create_test_wheel(
    tmp_path: Path, package_name: str, version: str = "1.0.0"
) -> Path:
    module_name = package_name.replace("-", "_")
    wheel_name = f"{module_name}-{version}-py3-none-any.whl"
    wheel_path = tmp_path / wheel_name
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(f"{module_name}/__init__.py", f'__version__ = "{version}"\n')
        wheel.writestr(
            f"{module_name}-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {package_name}\nVersion: {version}\n",
        )
        wheel.writestr(
            f"{module_name}-{version}.dist-info/WHEEL",
            "Wheel-Version: 1.0\n"
            "Generator: xinference-test\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n",
        )
        wheel.writestr(f"{module_name}-{version}.dist-info/RECORD", "")
    return wheel_path


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


class TestBuildUvSourceOptions:
    def test_reuses_all_configured_source_options(self):
        options = build_uv_source_options(
            {
                "index_url": "https://packages.example/simple",
                "extra_index_url": ["https://cuda.example/simple"],
                "find_links": ["/srv/wheels", "/opt/wheels"],
                "trusted_host": "packages.example",
                "index_strategy": "unsafe-best-match",
            },
            public_index_urls=["https://public.example/simple"],
        )

        assert options == [
            "--index",
            "https://cuda.example/simple",
            "--index",
            "https://packages.example/simple",
            "--default-index",
            "https://public.example/simple",
            "--find-links",
            "/srv/wheels",
            "--find-links",
            "/opt/wheels",
            "--trusted-host",
            "packages.example",
            "--index-strategy",
            "first-index",
        ]

    def test_configured_index_precedes_public_fallback(self):
        options = build_uv_source_options(
            {"index_url": "https://packages.example/simple"},
            public_index_urls=["https://public.example/simple"],
        )

        # uv prioritizes --index over --default-index. This verifies the
        # configured corporate mirror is effective before the public fallback,
        # rather than merely checking that both flags are present.
        assert options == [
            "--index",
            "https://packages.example/simple",
            "--default-index",
            "https://public.example/simple",
            "--index-strategy",
            "first-index",
        ]

    def test_multiple_public_fallbacks_follow_configured_sources(self):
        options = build_uv_source_options(
            {"index_url": "https://packages.example/simple"},
            public_index_urls=[
                "https://public-primary.example/simple",
                "https://public-default.example/simple",
            ],
        )

        assert options == [
            "--index",
            "https://packages.example/simple",
            "--index",
            "https://public-primary.example/simple",
            "--default-index",
            "https://public-default.example/simple",
            "--index-strategy",
            "first-index",
        ]

    def test_preserves_configured_strategy_without_public_fallback(self):
        options = build_uv_source_options(
            {
                "index_url": "https://packages.example/simple",
                "index_strategy": "unsafe-best-match",
            }
        )

        assert options == [
            "--default-index",
            "https://packages.example/simple",
            "--index-strategy",
            "unsafe-best-match",
        ]

    def test_offline_mode_omits_public_fallbacks(self):
        options = build_uv_source_options(
            {"find_links": "/srv/wheels"},
            public_index_urls=["https://public.example/simple"],
            allow_public_install=False,
        )

        assert options == ["--find-links", "/srv/wheels", "--no-index"]

    def test_configured_only_extra_index_replaces_implicit_public_default(self):
        options = build_uv_source_options(
            {
                "extra_index_url": "https://packages.example/simple",
                "index_strategy": "unsafe-best-match",
            },
            allow_public_install=False,
        )

        assert options == [
            "--default-index",
            "https://packages.example/simple",
            "--index-strategy",
            "unsafe-best-match",
        ]

    def test_deduplicates_public_fallback(self):
        options = build_uv_source_options(
            {"extra_index_url": "https://public.example/simple"},
            public_index_urls=["https://public.example/simple"],
        )

        assert options == [
            "--index",
            "https://public.example/simple",
        ]

    def test_resolver_does_not_consult_unavailable_public_fallback(self, tmp_path):
        uv_path = shutil.which("uv")
        if uv_path is None:
            pytest.skip("uv is required for the resolver-level source priority test")

        package_name = "xinference-source-priority-test"
        wheel_path = _create_test_wheel(tmp_path, package_name)
        wheel_name = wheel_path.name
        wheel_bytes = wheel_path.read_bytes()
        fallback_requests = []

        class SourceHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == f"/private/simple/{package_name}/":
                    body = (f'<a href="/files/{wheel_name}">{wheel_name}</a>').encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                elif self.path == f"/files/{wheel_name}":
                    body = wheel_bytes
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                elif self.path.startswith("/fallback/"):
                    fallback_requests.append(self.path)
                    self.send_error(503, "public fallback unavailable")
                    return
                else:
                    self.send_error(404)
                    return
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SourceHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            private_index = f"http://127.0.0.1:{port}/private/simple"
            public_fallback = f"http://127.0.0.1:{port}/fallback/simple"
            options = build_uv_source_options(
                {
                    "index_url": private_index,
                    "index_strategy": "unsafe-best-match",
                },
                public_index_urls=[public_fallback],
            )
            result = subprocess.run(
                [
                    uv_path,
                    "pip",
                    "install",
                    "--dry-run",
                    "--no-cache",
                    "--no-deps",
                    "--python",
                    sys.executable,
                    *options,
                    f"{package_name}==1.0.0",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        assert result.returncode == 0, result.stderr
        assert fallback_requests == []

    def test_find_links_miss_retries_with_public_fallback(self):
        configured_result = mock.MagicMock(returncode=1, stderr="package missing")
        fallback_result = mock.MagicMock(returncode=0)
        public_fallback = "https://public.example/simple"

        with mock.patch(
            "xinference.core.virtual_env_manager.subprocess.run",
            side_effect=[configured_result, fallback_result],
        ) as run_mock:
            result = _run_uv_install_with_source_fallback(
                ["uv", "pip", "install"],
                ["example-package==1.0.0"],
                {"find_links": "/srv/wheels"},
                public_index_urls=[public_fallback],
            )

        assert result is fallback_result
        assert run_mock.call_count == 2
        configured_cmd = run_mock.call_args_list[0][0][0]
        fallback_cmd = run_mock.call_args_list[1][0][0]
        assert "--no-index" in configured_cmd
        assert public_fallback not in configured_cmd
        assert "--no-index" not in fallback_cmd
        assert public_fallback in fallback_cmd
        assert "--find-links" not in fallback_cmd

    def test_configured_version_miss_uses_public_fallback(self, tmp_path):
        uv_path = shutil.which("uv")
        if uv_path is None:
            pytest.skip("uv is required for the resolver-level fallback test")

        package_name = "xinference-version-miss-fallback-test"
        private_dir = tmp_path / "private-files"
        public_dir = tmp_path / "public-files"
        private_dir.mkdir()
        public_dir.mkdir()
        private_wheel = _create_test_wheel(private_dir, package_name, "0.9.0")
        public_wheel = _create_test_wheel(public_dir, package_name, "1.0.0")
        private_requests = []
        public_requests = []

        class SourceHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == f"/private/simple/{package_name}/":
                    private_requests.append(self.path)
                    wheel_path = private_wheel
                elif self.path == f"/public/simple/{package_name}/":
                    public_requests.append(self.path)
                    wheel_path = public_wheel
                elif self.path == f"/files/{private_wheel.name}":
                    wheel_path = private_wheel
                    self._send_wheel(wheel_path)
                    return
                elif self.path == f"/files/{public_wheel.name}":
                    wheel_path = public_wheel
                    self._send_wheel(wheel_path)
                    return
                else:
                    self.send_error(404)
                    return

                body = (
                    f'<a href="/files/{wheel_path.name}">{wheel_path.name}</a>'
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_wheel(self, wheel_path):
                body = wheel_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SourceHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            result = _run_uv_install_with_source_fallback(
                [
                    uv_path,
                    "pip",
                    "install",
                    "--dry-run",
                    "--no-cache",
                    "--no-deps",
                    "--python",
                    sys.executable,
                ],
                [f"{package_name}==1.0.0"],
                {
                    "index_url": f"http://127.0.0.1:{port}/private/simple",
                    "index_strategy": "unsafe-best-match",
                },
                public_index_urls=[f"http://127.0.0.1:{port}/public/simple"],
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        assert result.returncode == 0, result.stderr
        assert len(private_requests) == 1
        assert len(public_requests) == 1

    def test_find_links_resolves_before_unavailable_public_fallback(self, tmp_path):
        uv_path = shutil.which("uv")
        if uv_path is None:
            pytest.skip("uv is required for the resolver-level source priority test")

        package_name = "xinference-find-links-priority-test"
        _create_test_wheel(tmp_path, package_name)
        fallback_requests = []

        class FallbackHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                fallback_requests.append(self.path)
                self.send_error(503, "public fallback unavailable")

            def log_message(self, format, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FallbackHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            public_fallback = (
                f"http://127.0.0.1:{server.server_address[1]}/fallback/simple"
            )
            result = _run_uv_install_with_source_fallback(
                [
                    uv_path,
                    "pip",
                    "install",
                    "--dry-run",
                    "--no-cache",
                    "--no-deps",
                    "--python",
                    sys.executable,
                ],
                [f"{package_name}==1.0.0"],
                {
                    "find_links": str(tmp_path),
                    "index_strategy": "unsafe-best-match",
                },
                public_index_urls=[public_fallback],
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        assert result.returncode == 0, result.stderr
        assert fallback_requests == []


class TestValidateVirtualEnvFindLinks:
    def test_accepts_canonical_paths_and_deduplicates(self, tmp_path):
        allowed = tmp_path / "allowed"
        wheels = allowed / "wheels"
        wheels.mkdir(parents=True)

        result = validate_virtual_env_find_links(
            [str(wheels), f"  {wheels}  ", ""],
            allowed_roots=(str(allowed),),
        )

        assert result == [str(wheels.resolve())]

    @pytest.mark.parametrize(
        "value, message",
        [
            ("relative/wheels", "must be absolute"),
            ("https://example.com/wheels", "absolute local directories"),
            ("file:///srv/wheels", "absolute local directories"),
        ],
    )
    def test_rejects_non_local_absolute_paths(self, tmp_path, value, message):
        with pytest.raises(ValueError, match=message):
            validate_virtual_env_find_links([value], allowed_roots=(str(tmp_path),))

    def test_rejects_non_list_and_non_string_entries(self, tmp_path):
        with pytest.raises(ValueError, match="must be a list"):
            validate_virtual_env_find_links(  # type: ignore[arg-type]
                str(tmp_path), allowed_roots=(str(tmp_path),)
            )
        with pytest.raises(ValueError, match="entries must be strings"):
            validate_virtual_env_find_links(  # type: ignore[list-item]
                [123], allowed_roots=(str(tmp_path),)
            )

    def test_rejects_missing_files_and_regular_files(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            validate_virtual_env_find_links(
                [str(tmp_path / "missing")], allowed_roots=(str(tmp_path),)
            )

        wheel = tmp_path / "package.whl"
        wheel.write_text("not a wheel")
        with pytest.raises(ValueError, match="not a directory"):
            validate_virtual_env_find_links(
                [str(wheel)], allowed_roots=(str(tmp_path),)
            )

    def test_rejects_path_outside_allowed_roots(self, tmp_path):
        allowed = tmp_path / "allowed"
        outside = tmp_path / "allowed-suffix"
        allowed.mkdir()
        outside.mkdir()

        with pytest.raises(ValueError, match="outside the configured allowed roots"):
            validate_virtual_env_find_links(
                [str(outside)], allowed_roots=(str(allowed),)
            )

    def test_rejects_symlink_escape(self, tmp_path):
        allowed = tmp_path / "allowed"
        outside = tmp_path / "outside"
        allowed.mkdir()
        outside.mkdir()
        link = allowed / "wheels"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("directory symlinks are not available")

        with pytest.raises(ValueError, match="outside the configured allowed roots"):
            validate_virtual_env_find_links([str(link)], allowed_roots=(str(allowed),))

    def test_rejects_unreadable_directory(self, tmp_path):
        with mock.patch(
            "xinference.core.virtual_env_manager.os.access", return_value=False
        ):
            with pytest.raises(ValueError, match="not readable"):
                validate_virtual_env_find_links(
                    [str(tmp_path)], allowed_roots=(str(tmp_path),)
                )

    def test_empty_allowed_roots_disables_feature(self, tmp_path):
        with pytest.raises(ValueError, match="disabled on this worker"):
            validate_virtual_env_find_links([str(tmp_path)], allowed_roots=())

    def test_merge_preserves_configured_sources(self):
        assert merge_virtual_env_find_links(
            ["https://packages.example/wheels", "/srv/wheels"],
            ["/srv/wheels", "/opt/wheels"],
        ) == [
            "https://packages.example/wheels",
            "/srv/wheels",
            "/opt/wheels",
        ]


class TestEnsureFlashinferCubinMatchesPostInstall:
    @pytest.fixture
    def fake_venv_manager(self):
        manager = mock.MagicMock()
        manager._get_uv_path.return_value = "/fake/uv"
        manager.env_path = "/fake/venv"
        return manager

    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        monkeypatch.delenv("FLASHINFER_DISABLE_VERSION_CHECK", raising=False)

    def test_matching_versions_are_noop(self, fake_venv_manager):
        with (
            mock.patch(
                "xinference.core.virtual_env_manager._get_virtualenv_distribution_version",
                side_effect=["0.6.14", "0.6.14"],
            ),
            mock.patch(
                "xinference.core.virtual_env_manager.subprocess.run"
            ) as run_mock,
        ):
            ensure_flashinfer_cubin_matches_post_install("vllm", fake_venv_manager)

        run_mock.assert_not_called()
        assert "FLASHINFER_DISABLE_VERSION_CHECK" not in os.environ

    def test_mismatched_cubin_is_synchronized(self, fake_venv_manager):
        result = mock.MagicMock(returncode=0)
        with (
            mock.patch(
                "xinference.core.virtual_env_manager._get_virtualenv_distribution_version",
                side_effect=["0.6.14", "0.6.6", "0.6.14"],
            ),
            mock.patch(
                "xinference.core.virtual_env_manager.subprocess.run",
                return_value=result,
            ) as run_mock,
        ):
            ensure_flashinfer_cubin_matches_post_install("vllm", fake_venv_manager)

        cmd = run_mock.call_args[0][0]
        assert "flashinfer-cubin==0.6.14" in cmd
        assert FLASHINFER_CUBIN_WHEEL_URL in cmd
        assert "FLASHINFER_DISABLE_VERSION_CHECK" not in os.environ

    def test_failed_sync_sets_version_check_bypass(self, fake_venv_manager):
        result = mock.MagicMock(returncode=1, stderr="wheel unavailable")
        with (
            mock.patch(
                "xinference.core.virtual_env_manager._get_virtualenv_distribution_version",
                side_effect=["0.6.14", "0.6.6"],
            ),
            mock.patch(
                "xinference.core.virtual_env_manager.subprocess.run",
                return_value=result,
            ),
        ):
            ensure_flashinfer_cubin_matches_post_install("vllm", fake_venv_manager)

        assert os.environ.get("FLASHINFER_DISABLE_VERSION_CHECK") == "1"

    def test_offline_mismatch_uses_version_check_bypass(self, fake_venv_manager):
        with (
            mock.patch(
                "xinference.core.virtual_env_manager._get_virtualenv_distribution_version",
                side_effect=["0.6.14", "0.6.6"],
            ),
            mock.patch(
                "xinference.core.virtual_env_manager.subprocess.run"
            ) as run_mock,
        ):
            ensure_flashinfer_cubin_matches_post_install(
                "vllm", fake_venv_manager, allow_public_install=False
            )

        run_mock.assert_not_called()
        assert os.environ.get("FLASHINFER_DISABLE_VERSION_CHECK") == "1"

    def test_offline_mismatch_uses_configured_find_links(self, fake_venv_manager):
        result = mock.MagicMock(returncode=0)
        with (
            mock.patch(
                "xinference.core.virtual_env_manager._get_virtualenv_distribution_version",
                side_effect=["0.6.14", "0.6.6", "0.6.14"],
            ),
            mock.patch(
                "xinference.core.virtual_env_manager.subprocess.run",
                return_value=result,
            ) as run_mock,
        ):
            ensure_flashinfer_cubin_matches_post_install(
                "vllm",
                fake_venv_manager,
                allow_public_install=False,
                conf={"find_links": "/srv/wheels"},
            )

        cmd = run_mock.call_args[0][0]
        assert ["--find-links", "/srv/wheels"] == cmd[7:9]
        assert "--no-index" in cmd
        assert FLASHINFER_CUBIN_WHEEL_URL not in cmd
        assert "FLASHINFER_DISABLE_VERSION_CHECK" not in os.environ

    def test_non_vllm_engine_is_noop(self, fake_venv_manager):
        with mock.patch(
            "xinference.core.virtual_env_manager._get_virtualenv_distribution_version"
        ) as version_mock:
            ensure_flashinfer_cubin_matches_post_install("sglang", fake_venv_manager)

        version_mock.assert_not_called()


class TestEnsureSGLangInheritedPackagesCompatiblePostInstall:
    @pytest.fixture
    def fake_venv_manager(self):
        manager = mock.MagicMock()
        manager._get_uv_path.return_value = "/fake/uv"
        manager.get_python_path.return_value = "/fake/venv/bin/python"
        manager.env_path = "/fake/venv"
        return manager

    def test_matching_versions_are_noop(self, fake_venv_manager):
        versions = {"numpy": "2.2.6", "pandas": "2.3.3"}
        with (
            mock.patch("importlib.metadata.version", side_effect=versions.__getitem__),
            mock.patch(
                "xinference.core.virtual_env_manager._get_virtualenv_distribution_version",
                side_effect=lambda _manager, name: versions[name],
            ),
            mock.patch(
                "xinference.core.virtual_env_manager.subprocess.run"
            ) as run_mock,
        ):
            ensure_sglang_inherited_packages_compatible_post_install(
                "sglang", fake_venv_manager
            )

        run_mock.assert_not_called()

    def test_cached_newer_numpy_is_removed(self, fake_venv_manager):
        result = mock.MagicMock(returncode=0, stderr="")
        versions = {"numpy": "2.2.6", "pandas": "2.3.3"}
        with (
            mock.patch("importlib.metadata.version", side_effect=versions.__getitem__),
            mock.patch(
                "xinference.core.virtual_env_manager._get_virtualenv_distribution_version",
                side_effect=["2.4.1", "2.2.6", "2.3.3"],
            ),
            mock.patch(
                "xinference.core.virtual_env_manager.subprocess.run",
                return_value=result,
            ) as run_mock,
        ):
            ensure_sglang_inherited_packages_compatible_post_install(
                "sglang", fake_venv_manager
            )

        assert run_mock.call_args[0][0] == [
            "/fake/uv",
            "pip",
            "uninstall",
            "--python",
            "/fake/venv/bin/python",
            "numpy",
        ]

    def test_cached_pandas_3_is_removed(self, fake_venv_manager):
        result = mock.MagicMock(returncode=0, stderr="")
        versions = {"numpy": "2.2.6", "pandas": "2.3.3"}
        with (
            mock.patch("importlib.metadata.version", side_effect=versions.__getitem__),
            mock.patch(
                "xinference.core.virtual_env_manager._get_virtualenv_distribution_version",
                side_effect=["2.2.6", "3.0.0", "2.3.3"],
            ),
            mock.patch(
                "xinference.core.virtual_env_manager.subprocess.run",
                return_value=result,
            ) as run_mock,
        ):
            ensure_sglang_inherited_packages_compatible_post_install(
                "sglang", fake_venv_manager
            )

        assert run_mock.call_args[0][0] == [
            "/fake/uv",
            "pip",
            "uninstall",
            "--python",
            "/fake/venv/bin/python",
            "pandas",
        ]

    def test_other_engines_are_untouched(self, fake_venv_manager):
        with mock.patch("importlib.metadata.version") as version_mock:
            ensure_sglang_inherited_packages_compatible_post_install(
                "vllm", fake_venv_manager
            )

        version_mock.assert_not_called()


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

    def test_extra_index_url_used_before_public_fallback(self, fake_venv_manager):
        configured_result = mock.MagicMock(returncode=1, stderr="package missing")
        fallback_result = mock.MagicMock(returncode=0)
        with mock.patch(
            "xinference.core.virtual_env_manager.subprocess.run",
            side_effect=[configured_result, fallback_result],
        ) as run_mock:
            apply_flashinfer_aot_post_install(
                "vllm",
                ["Qwen3_5MoeForConditionalGeneration"],
                fake_venv_manager,
                {"extra_index_url": ["https://wheels.vllm.ai/0.19.0/cu130"]},
                "13.0",
            )

        configured_cmd = run_mock.call_args_list[0][0][0]
        fallback_cmd = run_mock.call_args_list[1][0][0]
        assert "wheels.vllm.ai" in " ".join(configured_cmd)
        assert "flashinfer.ai" not in " ".join(configured_cmd)
        assert "wheels.vllm.ai" not in " ".join(fallback_cmd)
        assert "flashinfer.ai" in " ".join(fallback_cmd)

    def test_extra_index_url_string_used_before_public_fallback(
        self, fake_venv_manager
    ):
        configured_result = mock.MagicMock(returncode=1, stderr="package missing")
        fallback_result = mock.MagicMock(returncode=0)
        with mock.patch(
            "xinference.core.virtual_env_manager.subprocess.run",
            side_effect=[configured_result, fallback_result],
        ) as run_mock:
            apply_flashinfer_aot_post_install(
                "vllm",
                ["Qwen3_5MoeForConditionalGeneration"],
                fake_venv_manager,
                {"extra_index_url": "https://wheels.vllm.ai/0.19.0/cu130"},
                "13.0",
            )

        configured_cmd = run_mock.call_args_list[0][0][0]
        fallback_cmd = run_mock.call_args_list[1][0][0]
        assert "wheels.vllm.ai" in " ".join(configured_cmd)
        assert "flashinfer.ai" not in " ".join(configured_cmd)
        assert "wheels.vllm.ai" not in " ".join(fallback_cmd)
        assert "flashinfer.ai" in " ".join(fallback_cmd)

    def test_configured_source_success_skips_public_fallback(self, fake_venv_manager):
        result = mock.MagicMock(returncode=0)
        with mock.patch(
            "xinference.core.virtual_env_manager.subprocess.run", return_value=result
        ) as run_mock:
            apply_flashinfer_aot_post_install(
                "vllm",
                ["Qwen3_5MoeForConditionalGeneration"],
                fake_venv_manager,
                {"find_links": "/srv/wheels"},
                "13.0",
            )

        run_mock.assert_called_once()
        cmd = run_mock.call_args[0][0]
        assert ["--find-links", "/srv/wheels"] == cmd[8:10]
        assert "--no-index" in cmd
        assert FLASHINFER_AOT_WHEEL_URL not in cmd

    def test_reuses_configured_source_options_before_public_fallback(
        self, fake_venv_manager
    ):
        configured_result = mock.MagicMock(returncode=1, stderr="package missing")
        fallback_result = mock.MagicMock(returncode=0)
        conf = {
            "index_url": "https://packages.example/simple",
            "find_links": "/srv/wheels",
            "trusted_host": "packages.example",
            "index_strategy": "unsafe-best-match",
        }
        with mock.patch(
            "xinference.core.virtual_env_manager.subprocess.run",
            side_effect=[configured_result, fallback_result],
        ) as run_mock:
            apply_flashinfer_aot_post_install(
                "vllm",
                ["Qwen3_5MoeForConditionalGeneration"],
                fake_venv_manager,
                conf,
                "13.0",
            )

        configured_cmd = run_mock.call_args_list[0][0][0]
        configured_index_pos = configured_cmd.index("--default-index")
        assert (
            configured_cmd[configured_index_pos + 1]
            == "https://packages.example/simple"
        )
        assert "--find-links" in configured_cmd
        assert "/srv/wheels" in configured_cmd
        assert "--trusted-host" in configured_cmd
        assert "packages.example" in configured_cmd
        assert "--index-strategy" in configured_cmd
        assert "unsafe-best-match" in configured_cmd
        assert FLASHINFER_AOT_WHEEL_URL not in configured_cmd

        fallback_cmd = run_mock.call_args_list[1][0][0]
        fallback_index_pos = fallback_cmd.index("--default-index")
        assert fallback_cmd[fallback_index_pos + 1] == FLASHINFER_AOT_WHEEL_URL
        assert "https://packages.example/simple" not in fallback_cmd
        assert "--find-links" not in fallback_cmd
        assert "--trusted-host" not in fallback_cmd
        assert "--index-strategy" not in fallback_cmd

    def test_offline_mode_uses_configured_find_links(self, fake_venv_manager):
        result = mock.MagicMock(returncode=0)
        with mock.patch(
            "xinference.core.virtual_env_manager.subprocess.run", return_value=result
        ) as run_mock:
            apply_flashinfer_aot_post_install(
                "vllm",
                ["Qwen3_5MoeForConditionalGeneration"],
                fake_venv_manager,
                {"find_links": "/srv/wheels"},
                "13.0",
                allow_public_install=False,
            )

        cmd = run_mock.call_args[0][0]
        assert "--find-links" in cmd
        assert "/srv/wheels" in cmd
        assert "--no-index" in cmd
        assert FLASHINFER_AOT_WHEEL_URL not in cmd

    def test_offline_mode_without_source_skips_install(self, fake_venv_manager):
        with mock.patch(
            "xinference.core.virtual_env_manager.subprocess.run"
        ) as run_mock:
            apply_flashinfer_aot_post_install(
                "vllm",
                ["Qwen3_5MoeForConditionalGeneration"],
                fake_venv_manager,
                {},
                "13.0",
                allow_public_install=False,
            )

        run_mock.assert_not_called()
        assert "FLASHINFER_DISABLE_VERSION_CHECK" not in os.environ

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


class TestGetEngineCriticalDependencySpecs:
    """Tests for get_engine_critical_dependency_specs().

    Covers the skip_installed inheritance hole: when the parent env's engine
    copy satisfies the requested spec, the venv skips installing the engine,
    so nothing enforces the engine's own declared dependency requirements
    (e.g. sglang declares transformers==4.57.1 while the Docker image ships
    transformers 5.x, which breaks sglang.srt at import).
    """

    def _patch_metadata(self, versions, requires_map=None):
        requires_map = requires_map or {}

        def fake_version(name):
            try:
                return versions[name.lower()]
            except KeyError:
                raise importlib.metadata.PackageNotFoundError(name)

        def fake_requires(name):
            if name.lower() not in versions:
                raise importlib.metadata.PackageNotFoundError(name)
            return requires_map.get(name.lower(), [])

        return mock.patch.multiple(
            "importlib.metadata", version=fake_version, requires=fake_requires
        )

    def test_incompatible_parent_dependency_adds_declared_spec(self):
        with self._patch_metadata(
            {"sglang": "0.5.6", "transformers": "5.5.0"},
            {"sglang": ["transformers==4.57.1"]},
        ):
            specs = get_engine_critical_dependency_specs("sglang", ["sglang>=0.5.6"])
        assert specs == ["transformers==4.57.1"]

    def test_compatible_parent_dependency_is_noop(self):
        with self._patch_metadata(
            {"sglang": "0.5.6", "transformers": "4.57.1"},
            {"sglang": ["transformers==4.57.1"]},
        ):
            specs = get_engine_critical_dependency_specs("sglang", ["sglang>=0.5.6"])
        assert specs == []

    def test_engine_absent_from_parent_is_noop(self):
        """Without a parent copy the venv resolves the engine and its full
        dependency closure itself; nothing to compensate for."""
        with self._patch_metadata({"transformers": "5.5.0"}):
            specs = get_engine_critical_dependency_specs("sglang", ["sglang>=0.5.6"])
        assert specs == []

    def test_requested_spec_forcing_fresh_engine_install_is_noop(self):
        """A parent copy not satisfying the requested spec means the venv
        installs its own engine with full dependency resolution."""
        with self._patch_metadata(
            {"sglang": "0.5.6", "transformers": "5.5.0"},
            {"sglang": ["transformers==4.57.1"]},
        ):
            specs = get_engine_critical_dependency_specs("sglang", ["sglang>=0.5.7"])
        assert specs == []

    def test_explicit_dependency_spec_wins(self):
        with self._patch_metadata(
            {"sglang": "0.5.6", "transformers": "5.5.0"},
            {"sglang": ["transformers==4.57.1"]},
        ):
            specs = get_engine_critical_dependency_specs(
                "sglang", ["sglang>=0.5.6", "transformers==4.55.0"]
            )
        assert specs == []

    def test_missing_dependency_is_added(self):
        with self._patch_metadata(
            {"sglang": "0.5.6"},
            {"sglang": ["transformers==4.57.1"]},
        ):
            specs = get_engine_critical_dependency_specs("sglang", ["sglang>=0.5.6"])
        assert specs == ["transformers==4.57.1"]

    def test_extra_marker_requirements_ignored(self):
        with self._patch_metadata(
            {"sglang": "0.5.6", "transformers": "5.5.0"},
            {"sglang": ['transformers==4.57.1 ; extra == "srt"']},
        ):
            specs = get_engine_critical_dependency_specs("sglang", ["sglang>=0.5.6"])
        assert specs == []

    def test_non_critical_engine_is_noop(self):
        specs = get_engine_critical_dependency_specs("vllm", ["vllm>=0.11.2"])
        assert specs == []

    def test_no_engine_is_noop(self):
        assert get_engine_critical_dependency_specs(None, []) == []

    def test_unparseable_package_entries_are_skipped(self):
        with self._patch_metadata(
            {"sglang": "0.5.6", "transformers": "5.5.0"},
            {"sglang": ["transformers==4.57.1"]},
        ):
            specs = get_engine_critical_dependency_specs(
                "sglang",
                [
                    "#system_torch#",
                    'https://example.com/sgl_kernel-0.3.21+cu130-cp310-abi3-manylinux2014_x86_64.whl ; cuda_version == "13.0"',
                    "sglang>=0.5.6",
                ],
            )
        assert specs == ["transformers==4.57.1"]
