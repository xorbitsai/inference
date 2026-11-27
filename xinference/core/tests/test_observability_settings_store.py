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

import json
import sys
from types import SimpleNamespace

import pytest

from xinference.core.observability_settings_store import ObservabilitySettingsStore
from xinference.core.system_settings_store import SystemSettingsStore


def test_defaults_without_saved_section(tmp_path):
    store = ObservabilitySettingsStore(
        str(tmp_path / "system-settings.json"), environ={}
    )

    assert store.get_public() == {
        "opentelemetry": {
            "enable_otel": False,
            "otlp_base_endpoint": "http://localhost:4318",
            "otlp_trace_endpoint": "",
            "otlp_metric_endpoint": "",
            "otlp_api_key": "",
            "otel_exporter_otlp_protocol": "http/protobuf",
            "otel_sampling_rate": 0.1,
            "otel_batch_export_schedule_delay": 5000,
            "otel_max_queue_size": 2048,
            "otel_max_export_batch_size": 512,
            "otel_metric_export_interval": 60000,
            "otel_batch_export_timeout": 10000,
            "otel_metric_export_timeout": 30000,
        }
    }


def test_save_observability_only_uses_shared_document(tmp_path):
    path = tmp_path / "system-settings.json"
    environ = {}
    store = ObservabilitySettingsStore(str(path), environ=environ)
    payload = store.get_public()
    payload["opentelemetry"].update(
        {
            "enable_otel": True,
            "otlp_base_endpoint": "https://otel.example.com/",
            "otlp_api_key": "otel_abcdefgh12345678",
            "otel_sampling_rate": 0.5,
            "otel_max_queue_size": 1024,
            "otel_max_export_batch_size": 256,
        }
    )

    public = store.save_public(payload)

    assert public["opentelemetry"]["otlp_api_key"] == "otel********5678"
    expected_opentelemetry = {
        **payload["opentelemetry"],
        "otlp_base_endpoint": "https://otel.example.com",
    }
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "version": 1,
        "settings": {},
        "observability": {"opentelemetry": expected_opentelemetry},
    }
    assert environ["XINFERENCE_ENABLE_OTEL"] == "true"
    assert environ["XINFERENCE_OTLP_BASE_ENDPOINT"] == "https://otel.example.com"
    assert environ["XINFERENCE_OTLP_API_KEY"] == "otel_abcdefgh12345678"
    assert environ["XINFERENCE_OTEL_SAMPLING_RATE"] == "0.5"


def test_masked_secret_round_trip_and_reset(tmp_path):
    path = tmp_path / "system-settings.json"
    environ = {"XINFERENCE_ENABLE_OTEL": "false"}
    store = ObservabilitySettingsStore(str(path), environ=environ)
    first = store.get_public()
    first["opentelemetry"]["otlp_api_key"] = "otel_abcdefgh12345678"
    store.save_public(first)

    update = store.get_public()
    update["opentelemetry"]["otel_sampling_rate"] = 0.75
    store.save_public(update)

    assert store.get().opentelemetry.otlp_api_key == "otel_abcdefgh12345678"
    assert store.get().opentelemetry.otel_sampling_rate == 0.75

    restored = store.reset()

    assert restored["opentelemetry"]["enable_otel"] is False
    assert restored["opentelemetry"]["otlp_api_key"] == ""
    assert environ["XINFERENCE_ENABLE_OTEL"] == "false"
    assert "XINFERENCE_OTLP_API_KEY" not in environ
    assert not path.exists()


def test_sections_save_and_reset_independently_in_one_file(tmp_path):
    path = tmp_path / "system-settings.json"
    download_store = SystemSettingsStore(str(path), environ={})
    observability_store = ObservabilitySettingsStore(str(path), environ={})

    download_payload = download_store.get_public()
    download_payload["download_source"] = "modelscope"
    download_store.save_public(download_payload)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document == {"version": 1, "settings": download_payload}

    observability_payload = observability_store.get_public()
    observability_payload["opentelemetry"]["enable_otel"] = True
    observability_store.save_public(observability_payload)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["settings"] == download_payload
    saved_observability = document["observability"]

    download_payload["download_max_attempts"] = 7
    download_store.save_public(download_payload)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["settings"] == download_payload
    assert document["observability"] == saved_observability

    observability_store.reset()
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document == {"version": 1, "settings": download_payload}

    observability_store.save_public(observability_payload)
    download_store.reset()
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["settings"] == {}
    assert document["observability"]["opentelemetry"]["enable_otel"] is True


def test_observability_save_preserves_legacy_version_one_download_section(tmp_path):
    path = tmp_path / "system-settings.json"
    download_settings = {
        "download_source": "modelscope",
        "hf_endpoint": "https://hf.example.com",
        "hf_token": "hf_abcdefgh12345678",
        "pip_index_url": "https://pip.example.com/simple",
        "download_max_attempts": 5,
        "hub_detect_timeout": 4.5,
        "model_download_workers": 6,
    }
    path.write_text(
        json.dumps({"version": 1, "settings": download_settings}),
        encoding="utf-8",
    )
    store = ObservabilitySettingsStore(str(path), environ={})
    payload = store.get_public()
    payload["opentelemetry"]["enable_otel"] = True

    store.save_public(payload)

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["version"] == 1
    assert document["settings"] == download_settings
    assert document["observability"]["opentelemetry"]["enable_otel"] is True


def test_flat_observability_draft_is_loaded(tmp_path):
    path = tmp_path / "system-settings.json"
    baseline = ObservabilitySettingsStore(str(path), environ={}).get_public()[
        "opentelemetry"
    ]
    baseline["enable_otel"] = True
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "settings": {},
                "observability": baseline,
            }
        ),
        encoding="utf-8",
    )

    restarted = ObservabilitySettingsStore(str(path), environ={})

    assert restarted.get().opentelemetry.enable_otel is True


def test_opentelemetry_save_and_reset_preserve_other_observability_providers(
    tmp_path,
):
    path = tmp_path / "system-settings.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "settings": {},
                "observability": {"prometheus": {"enabled": True}},
            }
        ),
        encoding="utf-8",
    )
    store = ObservabilitySettingsStore(str(path), environ={})
    payload = store.get_public()
    payload["opentelemetry"]["enable_otel"] = True

    store.save_public(payload)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["observability"]["prometheus"] == {"enabled": True}
    assert document["observability"]["opentelemetry"]["enable_otel"] is True

    store.reset()
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["observability"] == {"prometheus": {"enabled": True}}


def test_save_updates_loaded_otel_constants(monkeypatch, tmp_path):
    constants = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "xinference.constants", constants)
    store = ObservabilitySettingsStore(
        str(tmp_path / "system-settings.json"), environ={}
    )
    payload = store.get_public()
    payload["opentelemetry"].update(
        {
            "enable_otel": True,
            "otlp_base_endpoint": "https://otel.example.com",
            "otel_sampling_rate": 0.5,
        }
    )

    store.save_public(payload)

    assert constants.XINFERENCE_ENABLE_OTEL is True
    assert constants.XINFERENCE_OTLP_TRACE_ENDPOINT == (
        "https://otel.example.com/v1/traces"
    )
    assert constants.XINFERENCE_OTLP_METRIC_ENDPOINT == (
        "https://otel.example.com/v1/metrics"
    )
    assert constants.XINFERENCE_OTEL_SAMPLING_RATE == 0.5


def test_runtime_snapshot_changes_only_after_restart(tmp_path):
    path = tmp_path / "system-settings.json"
    store = ObservabilitySettingsStore(
        str(path), environ={"XINFERENCE_ENABLE_OTEL": "false"}
    )
    payload = store.get_public()
    payload["opentelemetry"]["enable_otel"] = True

    store.save_public(payload)

    assert store.get_runtime().opentelemetry.enable_otel is False

    restarted = ObservabilitySettingsStore(
        str(path), environ={"XINFERENCE_ENABLE_OTEL": "false"}
    )
    assert restarted.get_runtime().opentelemetry.enable_otel is True


def test_export_batch_size_must_not_exceed_queue_size(tmp_path):
    store = ObservabilitySettingsStore(
        str(tmp_path / "system-settings.json"), environ={}
    )
    payload = store.get_public()
    payload["opentelemetry"]["otel_max_queue_size"] = 10
    payload["opentelemetry"]["otel_max_export_batch_size"] = 11

    with pytest.raises(ValueError, match="must not exceed"):
        store.save_public(payload)
