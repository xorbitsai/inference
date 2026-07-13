import os
import json
import pytest

from xinference.model.llm.config_parser import (
    _ensure_within_allowed_roots,
    _resolve_config_and_dir,
)
from xinference.model.flexible.utils import get_launcher


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
    (model_dir / "config.json").write_text(json.dumps({"num_parameters": 13_000_000_000}))
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
