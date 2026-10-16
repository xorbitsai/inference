import json
import os

import pytest

from xinference.model.flexible.utils import get_launcher
from xinference.model.llm.config_parser import (
    _ensure_within_allowed_roots,
    _load_chat_template_file,
    _load_tokenizer_config,
    _resolve_config_and_dir,
)

# --- Fix 1: path confinement (CWE-22) ---


def test_path_outside_roots_is_rejected():
    # A classic traversal target must be refused.
    with pytest.raises(ValueError, match="outside the allowed"):
        _ensure_within_allowed_roots("/etc/passwd")


def test_path_inside_allowed_root_is_accepted(tmp_path, monkeypatch):
    # Operator-allowlisted root should pass and resolve to a real path.
    monkeypatch.setenv("XINFERENCE_MODEL_REGISTER_ROOTS", str(tmp_path))
    model_dir = tmp_path / "victim_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"num_parameters": 13_000_000_000})
    )
    resolved = _ensure_within_allowed_roots(str(model_dir))
    assert resolved == os.path.realpath(str(model_dir))
    config_path, _ = _resolve_config_and_dir(str(model_dir))
    assert config_path.endswith("config.json")


def test_symlink_escape_is_rejected(tmp_path, monkeypatch):
    # A symlink inside an allowed root pointing outside must not escape.
    monkeypatch.setenv("XINFERENCE_MODEL_REGISTER_ROOTS", str(tmp_path))
    link = tmp_path / "escape"
    link.symlink_to("/etc")
    with pytest.raises(ValueError, match="outside the allowed"):
        _ensure_within_allowed_roots(str(link / "passwd"))


# --- Fix 2: launcher allowlist (CWE-470) ---


def test_launcher_not_allowlisted_is_rejected(monkeypatch):
    monkeypatch.delenv("XINFERENCE_FLEXIBLE_LAUNCHERS", raising=False)
    with pytest.raises(ValueError, match="not permitted"):
        get_launcher("os.system")


def test_launcher_allowlisted_loads(monkeypatch):
    monkeypatch.setenv("XINFERENCE_FLEXIBLE_LAUNCHERS", "os.getcwd")
    fn = get_launcher("os.getcwd")
    assert callable(fn)


@pytest.mark.parametrize("filename", ["config.json", "model.safetensors"])
def test_explicit_config_file_cannot_bypass_roots(tmp_path, monkeypatch, filename):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (outside / "config.json").write_text('{"audit": true}')
    monkeypatch.setenv("XINFERENCE_MODEL_REGISTER_ROOTS", str(allowed))
    with pytest.raises(ValueError, match="outside the allowed"):
        _resolve_config_and_dir(str(outside / filename))


@pytest.mark.parametrize(
    "filename", ["config.json", "tokenizer_config.json", "chat_template.jinja"]
)
def test_auxiliary_file_symlinks_cannot_escape_roots(tmp_path, monkeypatch, filename):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / filename
    outside.write_text('{"audit": true}')
    (allowed / filename).symlink_to(outside)
    monkeypatch.setenv("XINFERENCE_MODEL_REGISTER_ROOTS", str(allowed))
    reader = {
        "config.json": _resolve_config_and_dir,
        "tokenizer_config.json": _load_tokenizer_config,
        "chat_template.jinja": _load_chat_template_file,
    }[filename]
    with pytest.raises(ValueError, match="outside the allowed"):
        reader(str(allowed))


def test_hub_snapshot_directory_cannot_escape_roots(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    (outside / "revision").mkdir(parents=True)
    (outside / "revision/config.json").write_text("{}")
    (allowed / "snapshots").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("XINFERENCE_MODEL_REGISTER_ROOTS", str(allowed))
    with pytest.raises(ValueError, match="outside the allowed"):
        _resolve_config_and_dir(str(allowed))


def test_hub_ref_file_cannot_escape_roots(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    (allowed / "refs").mkdir(parents=True)
    (allowed / "snapshots/revision").mkdir(parents=True)
    (allowed / "snapshots/revision/config.json").write_text("{}")
    outside = tmp_path / "ref"
    outside.write_text("revision")
    (allowed / "refs/main").symlink_to(outside)
    monkeypatch.setenv("XINFERENCE_MODEL_REGISTER_ROOTS", str(allowed))
    with pytest.raises(ValueError, match="outside the allowed"):
        _resolve_config_and_dir(str(allowed))


def test_hub_blob_symlink_within_approved_repository_is_supported(
    tmp_path, monkeypatch
):
    (tmp_path / "snapshots/revision").mkdir(parents=True)
    (tmp_path / "blobs").mkdir()
    blob = tmp_path / "blobs/config"
    blob.write_text("{}")
    (tmp_path / "snapshots/revision/config.json").symlink_to(blob)
    monkeypatch.setenv("XINFERENCE_MODEL_REGISTER_ROOTS", str(tmp_path))
    config, directory = _resolve_config_and_dir(str(tmp_path))
    assert os.path.realpath(config) == str(blob)
    assert directory == str(tmp_path / "snapshots/revision")


@pytest.mark.parametrize(
    "name", ["transformers", "modelscope", "yolo", "image_process"]
)
def test_shipped_launchers_work_without_environment_allowlist(monkeypatch, name):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from xinference.model.flexible import utils

    monkeypatch.delenv("XINFERENCE_FLEXIBLE_LAUNCHERS", raising=False)
    launcher = Mock()
    importer = Mock(return_value=SimpleNamespace(**{name: launcher}))
    monkeypatch.setattr(utils.importlib, "import_module", importer)
    assert get_launcher("xinference.model.flexible.launchers." + name) is launcher
    importer.assert_called_once_with("xinference.model.flexible.launchers")
