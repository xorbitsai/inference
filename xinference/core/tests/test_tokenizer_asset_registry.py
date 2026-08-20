import hashlib
import json
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]
from tokenizers import Tokenizer, models, pre_tokenizers

from xinference.core.tokenizer_asset_registry import (
    TokenizerAssetError,
    TokenizerAssetRegistry,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_asset(root: Path, asset_id: str = "deepseek-v4-flash-0731") -> Path:
    path = root / asset_id
    path.mkdir(parents=True)
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    encoding = path / "encoding"
    encoding.mkdir()
    (encoding / "encoding_dsv4.py").write_text(
        "def encode_messages(messages, thinking_mode, reasoning_effort=None):\n"
        "    prefix = '<think> ' if thinking_mode == 'thinking' else ''\n"
        "    return prefix + ' '.join(str(m.get('content', '')) for m in messages)\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "asset_id": asset_id,
        "display_name": "DeepSeek-V4-Flash-0731",
        "model_family": "deepseek-v4",
        "model_name": "DeepSeek-V4-Flash-0731",
        "revision": "0731",
        "encoding_type": "deepseek_v4",
        "compatible_models": ["DeepSeek-V4-Flash-0731"],
        "required_files": ["tokenizer.json", "encoding/encoding_dsv4.py"],
        "checksums": {
            "tokenizer.json": f"sha256:{_sha256(path / 'tokenizer.json')}",
            "encoding/encoding_dsv4.py": f"sha256:{_sha256(encoding / 'encoding_dsv4.py')}",
        },
        "capabilities": {"chat": True, "tools": True, "thinking": True},
    }
    (path / "asset.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def make_registry_config(
    tmp_path: Path,
    asset_root: Path,
    *,
    assets: list[dict] | None = None,
    allow_custom_path: bool = False,
) -> Path:
    config = tmp_path / "tokenizer-assets.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "asset_roots": [str(asset_root)],
                "allow_custom_path": allow_custom_path,
                "assets": (
                    assets
                    if assets is not None
                    else [
                        {
                            "asset_id": "deepseek-v4-flash-0731",
                            "path": "deepseek-v4-flash-0731",
                            "enabled": True,
                        }
                    ]
                ),
            }
        ),
        encoding="utf-8",
    )
    return config


def test_registered_asset_list_resolve_and_validate(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    asset_path = make_asset(asset_root)
    registry = TokenizerAssetRegistry(str(make_registry_config(tmp_path, asset_root)))

    listed = registry.list_assets()
    assert listed["allow_custom_path"] is False
    assert listed["config_error"] == ""
    assert listed["items"][0]["status"] == "available"
    assert "path" not in listed["items"][0]

    resolved = registry.resolve("deepseek-v4-flash-0731")
    assert resolved["tokenizer_path"] == str(asset_path.resolve())
    assert resolved["tokenizer_asset_revision"] == "0731"
    assert resolved["tokenizer_asset_fingerprint"].startswith("sha256:")

    validated = registry.validate_asset("deepseek-v4-flash-0731")
    assert validated["valid"] is True
    assert validated["validated_at"]
    assert validated["checks"]["required_files"] == "ok"
    assert validated["checks"]["checksums"] == "ok"
    assert validated["checks"]["manifest"] == "ok"
    assert validated["capabilities"] == {
        "chat": True,
        "tools": True,
        "thinking": True,
    }
    assert validated["required_files"] == [
        "tokenizer.json",
        "encoding/encoding_dsv4.py",
    ]


def test_explicit_missing_config_fails_closed(tmp_path: Path) -> None:
    registry = TokenizerAssetRegistry(str(tmp_path / "missing.yaml"))
    assert registry.allow_custom_path is False
    assert "does not exist" in registry.config_error


def test_no_registry_keeps_legacy_custom_path_compatibility() -> None:
    registry = TokenizerAssetRegistry("")
    assert registry.allow_custom_path is True
    assert registry.list_assets()["items"] == []


def test_checksum_mismatch_marks_only_that_asset_invalid(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    first = make_asset(asset_root)
    make_asset(asset_root, "second")
    (first / "tokenizer.json").write_text("changed", encoding="utf-8")
    config = make_registry_config(
        tmp_path,
        asset_root,
        assets=[
            {"asset_id": "deepseek-v4-flash-0731", "path": first.name},
            {"asset_id": "second", "path": "second"},
        ],
    )

    items = {
        item["asset_id"]: item
        for item in TokenizerAssetRegistry(str(config)).list_assets()["items"]
    }
    assert items["deepseek-v4-flash-0731"]["status"] == "invalid"
    assert "SHA-256 mismatch" in items["deepseek-v4-flash-0731"]["errors"][0]
    assert items["second"]["status"] == "available"


def test_duplicate_asset_id_invalidates_registry_config(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    make_asset(asset_root)
    config = make_registry_config(
        tmp_path,
        asset_root,
        assets=[
            {"asset_id": "deepseek-v4-flash-0731", "path": "deepseek-v4-flash-0731"},
            {"asset_id": "deepseek-v4-flash-0731", "path": "deepseek-v4-flash-0731"},
        ],
    )
    registry = TokenizerAssetRegistry(str(config))
    assert registry.allow_custom_path is False
    assert registry.list_assets()["items"] == []
    assert "Duplicate Tokenizer asset_id" in registry.config_error


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    outside = tmp_path / "outside"
    make_asset(outside)
    asset_root.mkdir()
    (asset_root / "escaped").symlink_to(
        outside / "deepseek-v4-flash-0731", target_is_directory=True
    )
    config = make_registry_config(
        tmp_path,
        asset_root,
        assets=[{"asset_id": "deepseek-v4-flash-0731", "path": "escaped"}],
    )

    item = TokenizerAssetRegistry(str(config)).list_assets()["items"][0]
    assert item["status"] == "invalid"
    assert "escapes configured asset_roots" in item["errors"][0]


def test_disabled_asset_cannot_be_resolved(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    make_asset(asset_root)
    config = make_registry_config(
        tmp_path,
        asset_root,
        assets=[
            {
                "asset_id": "deepseek-v4-flash-0731",
                "path": "deepseek-v4-flash-0731",
                "enabled": False,
            }
        ],
    )
    registry = TokenizerAssetRegistry(str(config))
    assert registry.get_asset("deepseek-v4-flash-0731")["status"] == "disabled"
    with pytest.raises(TokenizerAssetError, match="not available"):
        registry.resolve("deepseek-v4-flash-0731")


def test_asset_id_and_path_must_resolve_to_same_directory(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    make_asset(asset_root)
    registry = TokenizerAssetRegistry(str(make_registry_config(tmp_path, asset_root)))
    with pytest.raises(TokenizerAssetError, match="different directories"):
        registry.resolve("deepseek-v4-flash-0731", str(tmp_path / "other"))


def test_registered_asset_requires_checksums_for_all_required_files(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    asset_path = make_asset(asset_root)
    manifest_path = asset_path / "asset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checksums"].pop("encoding/encoding_dsv4.py")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    item = TokenizerAssetRegistry(
        str(make_registry_config(tmp_path, asset_root))
    ).list_assets()["items"][0]

    assert item["status"] == "invalid"
    assert any("Missing SHA-256 checksum" in error for error in item["errors"])


def test_manifest_metadata_is_validated(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    asset_path = make_asset(asset_root)
    manifest_path = asset_path / "asset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["compatible_models"] = []
    manifest["capabilities"]["chat"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    item = TokenizerAssetRegistry(
        str(make_registry_config(tmp_path, asset_root))
    ).list_assets()["items"][0]

    assert item["status"] == "invalid"
    assert any("compatible_models" in error for error in item["errors"])
    assert any("capability chat must be true" in error for error in item["errors"])


def test_declared_capabilities_are_reported(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    asset_path = make_asset(asset_root)
    manifest_path = asset_path / "asset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["capabilities"] = {"chat": True, "tools": False, "thinking": False}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    registry = TokenizerAssetRegistry(str(make_registry_config(tmp_path, asset_root)))

    validated = registry.validate_asset("deepseek-v4-flash-0731")

    assert validated["valid"] is True
    assert validated["checks"]["required_files"] == "ok"
    assert validated["capabilities"] == {
        "chat": True,
        "tools": False,
        "thinking": False,
    }

    resolved = registry.resolve("deepseek-v4-flash-0731")
    assert resolved["tokenizer_asset_capabilities"] == {
        "chat": True,
        "tools": False,
        "thinking": False,
    }


def test_reload_picks_up_registry_changes(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    make_asset(asset_root)
    config = make_registry_config(tmp_path, asset_root)
    registry = TokenizerAssetRegistry(str(config))
    make_asset(asset_root, "second")
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "asset_roots": [str(asset_root)],
                "allow_custom_path": False,
                "assets": [
                    {
                        "asset_id": "deepseek-v4-flash-0731",
                        "path": "deepseek-v4-flash-0731",
                    },
                    {"asset_id": "second", "path": "second"},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert [item["asset_id"] for item in registry.list_assets()["items"]] == [
        "deepseek-v4-flash-0731"
    ]
    registry.reload()
    assert [item["asset_id"] for item in registry.list_assets()["items"]] == [
        "deepseek-v4-flash-0731",
        "second",
    ]
