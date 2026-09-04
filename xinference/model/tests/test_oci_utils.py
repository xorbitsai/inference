# Copyright 2022-2023 XProbe Inc.
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
"""``oci://`` model URIs resolve to a local path."""

import json
import os
import tempfile
from unittest import mock

import pytest

from .. import llmman
from ..oci_utils import resolve_oci_model
from ..utils import parse_uri


def test_parse_uri_yields_a_bare_registry_reference():
    # The scheme-less remainder is exactly what llmman consumes.
    assert parse_uri("oci://ghcr.io/org/model:tag") == ("oci", "ghcr.io/org/model:tag")


def test_parse_uri_leaves_other_schemes_alone():
    assert parse_uri("file:///models/foo")[0] == "file"
    assert parse_uri("s3://bucket/key")[0] == "s3"


def test_parses_the_documented_contract():
    with tempfile.TemporaryDirectory() as path:
        line = json.dumps(
            {
                "reference": "ghcr.io/org/model:tag",
                "path": path,
                "format": "safetensors",
            }
        )
        assert llmman.parse_resolve_output(line, "ref") == path


def test_tolerates_trailing_newline_and_leaked_diagnostics():
    with tempfile.TemporaryDirectory() as path:
        out = "pulling blobs...\n" + json.dumps({"path": path}) + "\n"
        assert llmman.parse_resolve_output(out, "ref") == path


def test_ignores_unknown_fields_so_the_contract_can_grow():
    with tempfile.TemporaryDirectory() as path:
        line = json.dumps({"path": path, "format": "gguf", "mmproj": "/x", "future": 1})
        assert llmman.parse_resolve_output(line, "ref") == path


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   \n\n",
        "not json",
        '["a", "list"]',
        '{"no_path": 1}',
        '{"path": ""}',
        '{"path": 3}',
        '{"path": "/nonexistent/xyzzy"}',
    ],
)
def test_rejects_malformed_output(bad):
    with pytest.raises(RuntimeError):
        llmman.parse_resolve_output(bad, "ref")


def test_rejects_an_empty_reference_without_touching_the_daemon():
    for ref in ("", "   "):
        with pytest.raises(ValueError):
            resolve_oci_model(ref)


def test_hands_the_reference_to_the_daemon_with_progress_wired():
    with mock.patch(
        "xinference.model.oci_utils.llmman.pull_and_resolve", return_value="/resolved"
    ) as acquire:
        assert resolve_oci_model("ghcr.io/org/model:tag") == "/resolved"
    assert acquire.call_args[0][0] == "ghcr.io/org/model:tag"
    assert acquire.call_args[1]["progress"] is not None


def _oci_llm_family():
    from ..llm.llm_family import LLMFamilyV2, PytorchLLMSpecV2

    return LLMFamilyV2(
        version=2,
        context_length=2048,
        model_type="LLM",
        model_name="oci-modelpack-test",
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[
            PytorchLLMSpecV2(
                model_format="pytorch",
                model_size_in_billions=1,
                model_id="org/model",
                quantization="none",
                model_uri="oci://ghcr.io/org/model:tag",
            )
        ],
        chat_template=None,
        stop_token_ids=None,
        stop=None,
    )


def test_llm_cache_manager_caches_an_oci_uri(tmp_path, monkeypatch):
    """The production path: ``LLMCacheManager.cache()`` accepts ``oci://``."""
    from ... import constants
    from ..cache_manager import CacheManager
    from ..llm.cache_manager import LLMCacheManager

    store = tmp_path / "llmman" / "models" / "modelpack"
    store.mkdir(parents=True)
    monkeypatch.setattr(constants, "XINFERENCE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(CacheManager, "is_initialized", False)
    family = _oci_llm_family()

    with mock.patch(
        "xinference.model.oci_utils.llmman.pull_and_resolve", return_value=str(store)
    ) as acquire:
        cache_dir = LLMCacheManager(family).cache()

    # The daemon is handed the bare reference, and the cache entry is a symlink
    # into its store -- exactly the shape a file:// URI produces.
    assert acquire.call_args[0][0] == "ghcr.io/org/model:tag"
    assert os.path.islink(cache_dir)
    assert os.path.realpath(cache_dir) == os.path.realpath(store)

    # A second launch reuses the entry instead of going back to the daemon.
    with mock.patch("xinference.model.oci_utils.llmman.pull_and_resolve") as again:
        assert LLMCacheManager(family).cache() == cache_dir
    again.assert_not_called()


@pytest.mark.parametrize(
    "host,want",
    [
        ("", "http://127.0.0.1:17434"),
        ("1.2.3.4:9999", "http://1.2.3.4:9999"),
        ("1.2.3.4", "http://1.2.3.4:17434"),
        ("http://1.2.3.4:9999/ignored", "http://1.2.3.4:9999"),
        # An explicit scheme shifts the default port, as llmman does; the
        # origin stays http either way.
        ("http://example.com", "http://example.com:80"),
        ("https://example.com", "http://example.com:443"),
        ("https://example.com:9999", "http://example.com:9999"),
        ("ftp://example.com", "http://example.com:17434"),
        # An unusable port falls back to the default rather than being sent.
        ("1.2.3.4:70000", "http://1.2.3.4:17434"),
        ("https://1.2.3.4:70000", "http://1.2.3.4:443"),
        ("[::1]:70000", "http://[::1]:17434"),
        ("1.2.3.4:abc", "http://1.2.3.4:17434"),
        # A wildcard bind is meaningful to the server but not to a client.
        ("0.0.0.0:9999", "http://127.0.0.1:9999"),
        ("[::]:9999", "http://[::1]:9999"),
        ("[::1]", "http://[::1]:17434"),
    ],
)
def test_endpoint_parsing(host, want):
    with mock.patch.dict(os.environ, {llmman.HOST_ENV: host}):
        assert llmman.endpoint() == want


def test_binary_default_and_override():
    with mock.patch.dict(os.environ, {llmman.BIN_ENV: ""}):
        assert llmman.llmman_bin() == "llmman"
    with mock.patch.dict(os.environ, {llmman.BIN_ENV: "/opt/llmman"}):
        assert llmman.llmman_bin() == "/opt/llmman"


def test_reports_a_missing_binary():
    with mock.patch.dict(os.environ, {llmman.BIN_ENV: "/definitely/not/here"}):
        with pytest.raises(RuntimeError, match="not found"):
            llmman.resolve("ref")
