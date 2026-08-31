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

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from xinference.core.system_settings_store import SystemSettingsStore


def test_system_settings_path_can_be_overridden_by_environment(tmp_path):
    constants_path = Path(__file__).parents[2] / "constants.py"
    custom_path = tmp_path / "custom-system-settings.json"
    environ = os.environ.copy()
    environ["XINFERENCE_SYSTEM_SETTINGS_PATH"] = str(custom_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy, sys; "
                "values = runpy.run_path(sys.argv[1]); "
                "print(values['XINFERENCE_SYSTEM_SETTINGS_PATH'])"
            ),
            str(constants_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environ,
    )

    assert result.stdout.strip() == str(custom_path)


def test_defaults_without_saved_file(tmp_path):
    store = SystemSettingsStore(str(tmp_path / "system-settings.json"), environ={})

    assert store.get_public() == {
        "download_source": "auto",
        "hf_endpoint": "",
        "hf_token": "",
        "pip_index_url": "",
        "download_max_attempts": 3,
        "hub_detect_timeout": 3.0,
        "model_download_workers": 2,
    }


def test_startup_baseline_uses_environment(tmp_path):
    environ = {
        "XINFERENCE_MODEL_SRC": "modelscope",
        "HF_ENDPOINT": "https://hf.example.com",
        "HUGGING_FACE_HUB_TOKEN": "hf_abcdefgh12345678",
        "PIP_INDEX_URL": "https://pip.example.com/simple",
        "XINFERENCE_DOWNLOAD_MAX_ATTEMPTS": "5",
        "XINFERENCE_HUB_DETECT_TIMEOUT": "4.5",
        "XINFERENCE_MODEL_DOWNLOAD_WORKERS": "6",
    }
    store = SystemSettingsStore(str(tmp_path / "system-settings.json"), environ=environ)

    settings = store.get()
    assert settings.download_source == "modelscope"
    assert settings.hf_endpoint == "https://hf.example.com"
    assert settings.hf_token == "hf_abcdefgh12345678"
    assert settings.pip_index_url == "https://pip.example.com/simple"
    assert settings.download_max_attempts == 5
    assert settings.hub_detect_timeout == 4.5
    assert settings.model_download_workers == 6
    assert store.get_public()["hf_token"] == "hf_a********5678"


def test_startup_preserves_supported_download_sources(tmp_path):
    for source in ("openmind_hub", "csghub"):
        environ = {"XINFERENCE_MODEL_SRC": source}
        store = SystemSettingsStore(
            str(tmp_path / source / "system-settings.json"), environ=environ
        )

        store.apply_to_environment()

        assert store.get().download_source == source
        assert environ["XINFERENCE_MODEL_SRC"] == source


def test_save_writes_full_plaintext_snapshot(tmp_path):
    path = tmp_path / "system-settings.json"
    environ = {}
    store = SystemSettingsStore(str(path), environ=environ)
    payload = {
        "download_source": "huggingface",
        "hf_endpoint": "https://hf-mirror.example.com",
        "hf_token": "hf_abcdefgh12345678",
        "pip_index_url": "https://pip.example.com/simple",
        "download_max_attempts": 5,
        "hub_detect_timeout": 4.5,
        "model_download_workers": 6,
    }

    public = store.save_public(payload)

    assert public["hf_token"] == "hf_a********5678"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["version"] == 1
    assert saved["settings"] == payload
    assert environ["XINFERENCE_MODEL_SRC"] == "huggingface"
    assert environ["HF_ENDPOINT"] == payload["hf_endpoint"]
    assert environ["HUGGING_FACE_HUB_TOKEN"] == payload["hf_token"]
    assert environ["HF_TOKEN"] == payload["hf_token"]
    assert environ["PIP_INDEX_URL"] == payload["pip_index_url"]
    assert environ["XINFERENCE_DOWNLOAD_MAX_ATTEMPTS"] == "5"
    assert environ["XINFERENCE_HUB_DETECT_TIMEOUT"] == "4.5"
    assert environ["XINFERENCE_MODEL_DOWNLOAD_WORKERS"] == "6"


def test_masked_token_round_trip_preserves_secret(tmp_path):
    path = tmp_path / "system-settings.json"
    store = SystemSettingsStore(str(path), environ={})
    first = store.get_public()
    first["hf_token"] = "hf_abcdefgh12345678"
    store.save_public(first)

    update = store.get_public()
    update["download_max_attempts"] = 7
    result = store.save_public(update)

    assert result["hf_token"] == "hf_a********5678"
    assert store.get().hf_token == "hf_abcdefgh12345678"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["settings"]["hf_token"] == "hf_abcdefgh12345678"
    assert saved["settings"]["download_max_attempts"] == 7


def test_empty_token_clears_secret(tmp_path):
    environ = {"HUGGING_FACE_HUB_TOKEN": "hf_abcdefgh12345678"}
    store = SystemSettingsStore(
        str(tmp_path / "system-settings.json"),
        environ=environ,
    )
    payload = store.get_public()
    payload["hf_token"] = ""

    store.save_public(payload)

    assert store.get().hf_token == ""
    assert "HUGGING_FACE_HUB_TOKEN" not in environ
    assert "HF_TOKEN" not in environ


def test_reset_restores_immutable_startup_baseline(tmp_path):
    path = tmp_path / "system-settings.json"
    environ = {
        "XINFERENCE_MODEL_SRC": "modelscope",
        "XINFERENCE_DOWNLOAD_MAX_ATTEMPTS": "8",
    }
    store = SystemSettingsStore(str(path), environ=environ)
    payload = store.get_public()
    payload["download_source"] = "huggingface"
    payload["download_max_attempts"] = 2
    store.save_public(payload)
    assert environ["XINFERENCE_MODEL_SRC"] == "huggingface"
    assert environ["XINFERENCE_DOWNLOAD_MAX_ATTEMPTS"] == "2"

    restored = store.reset()

    assert restored["download_source"] == "modelscope"
    assert restored["download_max_attempts"] == 8
    assert store.get() == store.get_startup()
    assert environ["XINFERENCE_MODEL_SRC"] == "modelscope"
    assert environ["XINFERENCE_DOWNLOAD_MAX_ATTEMPTS"] == "8"
    assert not path.exists()


def test_saved_snapshot_wins_until_reset(tmp_path):
    path = tmp_path / "system-settings.json"
    first = SystemSettingsStore(
        str(path), environ={"XINFERENCE_MODEL_SRC": "huggingface"}
    )
    payload = first.get_public()
    payload["download_source"] = "modelscope"
    first.save_public(payload)

    restarted = SystemSettingsStore(
        str(path), environ={"XINFERENCE_MODEL_SRC": "huggingface"}
    )

    assert restarted.get().download_source == "modelscope"
    assert restarted.get_startup().download_source == "huggingface"
    assert restarted.reset()["download_source"] == "huggingface"


def test_saved_snapshot_is_applied_during_startup(tmp_path):
    path = tmp_path / "system-settings.json"
    first = SystemSettingsStore(str(path), environ={})
    payload = first.get_public()
    payload.update(
        {
            "download_source": "modelscope",
            "hf_endpoint": "https://hf.example.com",
            "hf_token": "hf_abcdefgh12345678",
            "pip_index_url": "https://pip.example.com/simple",
            "download_max_attempts": 5,
            "hub_detect_timeout": 4.5,
            "model_download_workers": 6,
        }
    )
    first.save_public(payload)

    startup_environ = {}
    restarted = SystemSettingsStore(str(path), environ=startup_environ)
    restarted.apply_to_environment()

    assert startup_environ == {
        "XINFERENCE_MODEL_SRC": "modelscope",
        "HF_ENDPOINT": "https://hf.example.com",
        "HUGGING_FACE_HUB_TOKEN": "hf_abcdefgh12345678",
        "HF_TOKEN": "hf_abcdefgh12345678",
        "PIP_INDEX_URL": "https://pip.example.com/simple",
        "XINFERENCE_DOWNLOAD_MAX_ATTEMPTS": "5",
        "XINFERENCE_HUB_DETECT_TIMEOUT": "4.5",
        "XINFERENCE_MODEL_DOWNLOAD_WORKERS": "6",
    }


def test_save_updates_loaded_download_consumers(monkeypatch, tmp_path):
    constants = SimpleNamespace()
    model_utils = SimpleNamespace(
        XINFERENCE_DOWNLOAD_MAX_ATTEMPTS=3,
        XINFERENCE_HUB_DETECT_TIMEOUT=3.0,
        XINFERENCE_MODEL_DOWNLOAD_WORKERS=2,
        _auto_detected_hub="huggingface",
        _auto_detect_hub_lock=threading.Lock(),
    )
    worker = SimpleNamespace(XINFERENCE_MODEL_DOWNLOAD_WORKERS=2)
    hf_constants = SimpleNamespace(
        _staging_mode=False,
        _HF_DEFAULT_ENDPOINT="https://huggingface.co",
        ENDPOINT="https://huggingface.co",
        HUGGINGFACE_CO_URL_TEMPLATE=(
            "https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
        ),
    )
    hf_file_download = SimpleNamespace(
        ENDPOINT="https://huggingface.co",
        HUGGINGFACE_CO_URL_TEMPLATE=(
            "https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
        ),
    )
    hf_api = SimpleNamespace(ENDPOINT="https://huggingface.co")
    monkeypatch.setitem(sys.modules, "xinference.constants", constants)
    monkeypatch.setitem(sys.modules, "xinference.model.utils", model_utils)
    monkeypatch.setitem(sys.modules, "xinference.core.worker", worker)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", hf_constants)
    monkeypatch.setitem(sys.modules, "huggingface_hub.file_download", hf_file_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub.hf_api", hf_api)

    store = SystemSettingsStore(str(tmp_path / "system-settings.json"), environ={})
    payload = store.get_public()
    payload.update(
        {
            "hf_endpoint": "https://hf.example.com/",
            "download_max_attempts": 5,
            "hub_detect_timeout": 4.5,
            "model_download_workers": 6,
        }
    )

    store.save_public(payload)

    assert constants.XINFERENCE_DOWNLOAD_MAX_ATTEMPTS == 5
    assert constants.XINFERENCE_HUB_DETECT_TIMEOUT == 4.5
    assert constants.XINFERENCE_MODEL_DOWNLOAD_WORKERS == 6
    assert model_utils.XINFERENCE_DOWNLOAD_MAX_ATTEMPTS == 5
    assert model_utils.XINFERENCE_HUB_DETECT_TIMEOUT == 4.5
    assert model_utils.XINFERENCE_MODEL_DOWNLOAD_WORKERS == 6
    assert model_utils._auto_detected_hub is None
    assert worker.XINFERENCE_MODEL_DOWNLOAD_WORKERS == 6
    assert hf_constants.ENDPOINT == "https://hf.example.com"
    assert hf_constants.HUGGINGFACE_CO_URL_TEMPLATE == (
        "https://hf.example.com/{repo_id}/resolve/{revision}/{filename}"
    )
    assert hf_file_download.ENDPOINT == "https://hf.example.com"
    assert hf_file_download.HUGGINGFACE_CO_URL_TEMPLATE == (
        "https://hf.example.com/{repo_id}/resolve/{revision}/{filename}"
    )
    assert hf_api.ENDPOINT == "https://hf.example.com"

    store.reset()

    assert constants.XINFERENCE_DOWNLOAD_MAX_ATTEMPTS == 3
    assert constants.XINFERENCE_HUB_DETECT_TIMEOUT == 3.0
    assert constants.XINFERENCE_MODEL_DOWNLOAD_WORKERS == 2
    assert model_utils.XINFERENCE_MODEL_DOWNLOAD_WORKERS == 2
    assert worker.XINFERENCE_MODEL_DOWNLOAD_WORKERS == 2
    assert hf_constants.ENDPOINT == "https://huggingface.co"
    assert hf_constants.HUGGINGFACE_CO_URL_TEMPLATE == (
        "https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    )
    assert hf_file_download.ENDPOINT == "https://huggingface.co"
    assert hf_file_download.HUGGINGFACE_CO_URL_TEMPLATE == (
        "https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    )
    assert hf_api.ENDPOINT == "https://huggingface.co"


def test_save_updates_imported_huggingface_url_builder(tmp_path):
    from huggingface_hub import hf_hub_url

    store = SystemSettingsStore(str(tmp_path / "system-settings.json"), environ={})
    payload = store.get_public()
    payload["hf_endpoint"] = "https://hf.example.com/"

    try:
        store.save_public(payload)

        assert hf_hub_url("org/model", "config.json") == (
            "https://hf.example.com/org/model/resolve/main/config.json"
        )
    finally:
        store.reset()


def test_invalid_saved_file_falls_back_to_startup(tmp_path):
    path = tmp_path / "system-settings.json"
    path.write_text('{"version": 1, "settings": {"unknown": "value"}}')

    store = SystemSettingsStore(
        str(path), environ={"XINFERENCE_MODEL_SRC": "modelscope"}
    )

    assert store.get().download_source == "modelscope"


def test_missing_saved_fields_use_startup_values(tmp_path):
    path = tmp_path / "system-settings.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "settings": {
                    "download_source": "modelscope",
                    "hf_endpoint": "https://hf.example.com",
                    "hf_token": "hf_abcdefgh12345678",
                    "pip_index_url": "https://pip.example.com/simple",
                    "download_max_attempts": 5,
                    "hub_detect_timeout": 4.5,
                },
            }
        )
    )

    store = SystemSettingsStore(
        str(path), environ={"XINFERENCE_MODEL_DOWNLOAD_WORKERS": "6"}
    )

    assert store.get().to_dict() == {
        "download_source": "modelscope",
        "hf_endpoint": "https://hf.example.com",
        "hf_token": "hf_abcdefgh12345678",
        "pip_index_url": "https://pip.example.com/simple",
        "download_max_attempts": 5,
        "hub_detect_timeout": 4.5,
        "model_download_workers": 6,
    }
