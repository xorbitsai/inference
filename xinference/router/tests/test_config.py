from pathlib import Path

import pytest

from xinference.router.config import load_config


def test_loads_config_with_environment_secret(tmp_path: Path, monkeypatch) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
backend:
  url: http://127.0.0.1:9997
model:
  logical_model: router-model
  tokenizer_path: {assets}
  tokenizer_asset_id: deepseek-v4-flash-0731
  tokenizer_asset_revision: "0731"
  tokenizer_asset_fingerprint: sha256:test-fingerprint
limits:
  short_threshold_tokens: 100
  short_max_model_len: 200
  long_max_model_len: 1000
pools:
  short:
    model_uid: short-model
    max_active: 2
  long:
    model_uid: long-model
    max_active: 1
"""
    )
    monkeypatch.setenv("XINFERENCE_API_KEY", "secret")
    loaded = load_config(config)
    assert loaded.logical_model == "router-model"
    assert loaded.short_pool.model_uid == "short-model"
    assert loaded.long_pool.model_uid == "long-model"
    assert loaded.backend_api_key == "secret"
    assert loaded.tokenization.max_workers == 2
    assert loaded.tokenization.max_active == 2
    assert loaded.tokenization.max_queue == 8
    assert loaded.tokenizer_asset_revision == "0731"
    assert loaded.tokenizer_asset_fingerprint == "sha256:test-fingerprint"


def control_plane_config(assets: Path) -> dict:
    return {
        "router_uid": "router-1",
        "revision": 3,
        "enabled": True,
        "virtual_model_uid": "router-model",
        "tokenizer_path": str(assets),
        "tokenizer_asset_id": "deepseek-v4-flash-0731",
        "tokenizer_asset_revision": "0731",
        "tokenizer_asset_fingerprint": "sha256:test-fingerprint",
        "backend_url": "http://supervisor:9997",
        "backends": {
            "short": {
                "model_uid": "short-model",
                "max_context_tokens": 200,
                "admission": {"max_active": 2},
            },
            "long": {
                "model_uid": "long-model",
                "max_context_tokens": 1000,
                "admission": {"max_active": 1},
            },
        },
        "routing": {
            "short_threshold_tokens": 100,
            "context_reserve_tokens": 0,
            "default_output_tokens": 8,
            "thinking_policy": "reject",
        },
        "tokenization": {
            "max_workers": 2,
            "max_active": 2,
            "max_queue": 8,
        },
    }


def test_control_plane_config_uses_authorization_passthrough(
    tmp_path: Path, monkeypatch
) -> None:
    from xinference.router.config import config_from_control_plane

    assets = tmp_path / "assets"
    assets.mkdir()
    monkeypatch.setenv("XINFERENCE_API_KEY", "must-not-be-used")

    loaded = config_from_control_plane(control_plane_config(assets))

    assert loaded.backend_api_key == ""
    assert loaded.require_auth is False
    assert loaded.thinking_pool == "reject"
    assert loaded.tokenizer_asset_revision == "0731"
    assert loaded.tokenizer_asset_fingerprint == "sha256:test-fingerprint"


def test_control_plane_config_is_fully_validated(tmp_path: Path) -> None:
    from xinference.router.config import ConfigError, config_from_control_plane

    assets = tmp_path / "assets"
    assets.mkdir()
    payload = control_plane_config(assets)
    payload["tokenization"]["max_active"] = 1

    with pytest.raises(ConfigError, match="greater than or equal"):
        config_from_control_plane(payload)


def typed_control_plane_config(assets: Path) -> dict:
    return {
        "config_version": 2,
        "route_profile": "llm_chat",
        "strategy": "typed_rules",
        "router_uid": "router-v2",
        "revision": 7,
        "enabled": True,
        "virtual_model_uid": "router-model-v2",
        "tokenizer_path": str(assets),
        "tokenizer_asset_id": "deepseek-v4-flash-0731",
        "tokenizer_asset_revision": "0731",
        "tokenizer_asset_fingerprint": "sha256:test-fingerprint",
        "backend_url": "http://supervisor:9997",
        "backends": [
            {
                "id": backend_id,
                "model_uid": f"{backend_id}-model",
                "max_context_tokens": context,
                "admission": {"max_active": 1, "max_queue": 2},
            }
            for backend_id, context in (
                ("fast", 32768),
                ("tools", 65536),
                ("reasoning", 131072),
                ("long", 1048576),
            )
        ],
        "routing": {
            "evaluation_mode": "first_match",
            "context_reserve_tokens": 64,
            "default_output_tokens": 512,
            "rules": [
                {
                    "id": "tools-route",
                    "priority": 400,
                    "match": {"tools_present": True},
                    "action": {"type": "route", "backend_id": "tools"},
                },
                {
                    "id": "thinking-route",
                    "priority": 300,
                    "match": {"thinking": True},
                    "action": {"type": "route", "backend_id": "reasoning"},
                },
                {
                    "id": "short-route",
                    "priority": 200,
                    "match": {"total_tokens_lte": 32768},
                    "action": {"type": "route", "backend_id": "fast"},
                },
                {
                    "id": "long-route",
                    "priority": 100,
                    "match": {"total_tokens_gte": 32769},
                    "action": {"type": "route", "backend_id": "long"},
                },
            ],
            "default_action": {"type": "reject", "reason": "no_matching_route"},
        },
        "tokenization": {
            "max_workers": 2,
            "max_active": 2,
            "max_queue": 8,
        },
    }


def test_control_plane_v2_loads_four_dynamic_backends(tmp_path: Path) -> None:
    from xinference.router.config import config_from_control_plane

    assets = tmp_path / "assets"
    assets.mkdir()
    loaded = config_from_control_plane(typed_control_plane_config(assets))

    assert loaded.config_version == 2
    assert loaded.route_profile == "llm_chat"
    assert loaded.strategy == "typed_rules"
    assert [backend.id for backend in loaded.backends] == [
        "fast",
        "tools",
        "reasoning",
        "long",
    ]
    assert [rule.id for rule in loaded.rules] == [
        "tools-route",
        "thinking-route",
        "short-route",
        "long-route",
    ]
    assert loaded.default_action.reason == "no_matching_route"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["backends"][1].update(id="fast"),
            "backend ids must be unique",
        ),
        (
            lambda payload: payload["routing"]["rules"][1].update(id="tools-route"),
            "routing rule ids must be unique",
        ),
        (
            lambda payload: payload["routing"]["rules"][1].update(priority=400),
            "routing rule priorities must be unique",
        ),
        (
            lambda payload: payload["routing"]["rules"][0]["action"].update(
                backend_id="missing"
            ),
            "references unknown backend",
        ),
        (
            lambda payload: payload["routing"]["rules"][0].update(match={}),
            "match cannot be empty",
        ),
        (
            lambda payload: payload["routing"]["rules"][0].update(
                match={"total_tokens_gte": 10, "total_tokens_lte": 9}
            ),
            "Invalid token range",
        ),
        (
            lambda payload: payload.update(config_version=3),
            "Unsupported config_version",
        ),
        (
            lambda payload: payload.update(
                tokenizer_asset_capabilities={
                    "chat": True,
                    "tools": False,
                    "thinking": True,
                }
            ),
            "requires tools but the Tokenizer asset",
        ),
        (
            lambda payload: payload.update(
                tokenizer_asset_capabilities={
                    "chat": True,
                    "tools": True,
                    "thinking": False,
                }
            ),
            "requires thinking but the Tokenizer asset",
        ),
    ],
)
def test_control_plane_v2_rejects_invalid_dynamic_config(
    tmp_path: Path, mutate, message: str
) -> None:
    from copy import deepcopy

    from xinference.router.config import ConfigError, config_from_control_plane

    assets = tmp_path / "assets"
    assets.mkdir()
    payload = deepcopy(typed_control_plane_config(assets))
    mutate(payload)

    with pytest.raises(ConfigError, match=message):
        config_from_control_plane(payload)
