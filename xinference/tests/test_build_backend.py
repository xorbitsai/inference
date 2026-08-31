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

"""Unit tests for the in-tree build backend's revision recording."""

import importlib.util
import json
import os
import shutil
import subprocess

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
_BACKEND_PATH = os.path.join(_REPO_ROOT, "build_backend.py")

pytestmark = [
    pytest.mark.skipif(
        not os.path.exists(_BACKEND_PATH),
        reason="build backend only exists in a source tree",
    ),
    pytest.mark.skipif(shutil.which("git") is None, reason="requires the git binary"),
]


@pytest.fixture(scope="module")
def backend():
    import sys

    # build_backend imports its sibling build_web module from the repo root
    sys.path.insert(0, _REPO_ROOT)
    try:
        spec = importlib.util.spec_from_file_location(
            "_xinf_build_backend", _BACKEND_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(_REPO_ROOT)


def _read_recorded(root):
    path = os.path.join(root, "xinference", "_commit.py")
    if not os.path.exists(path):
        return None
    namespace: dict = {}
    with open(path) as f:
        exec(f.read(), namespace)
    return namespace["full_revisionid"]


def _make_source_tree(root):
    os.makedirs(os.path.join(root, "xinference"), exist_ok=True)
    with open(os.path.join(root, "xinference", "__init__.py"), "w") as f:
        f.write("")


def _write_llm_families(root, families):
    llm_dir = os.path.join(root, "xinference", "model", "llm")
    os.makedirs(llm_dir, exist_ok=True)
    with open(os.path.join(llm_dir, "llm_family.json"), "w") as f:
        json.dump(families, f)


def _git(cwd, *args):
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, env=env
    ).stdout.strip()


def test_records_head_of_own_checkout(backend, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_source_tree(str(repo))
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    head = _git(repo, "rev-parse", "HEAD")

    backend._record_full_revision(str(repo))

    assert _read_recorded(str(repo)) == head


def test_archive_node_used_when_not_a_checkout(backend, tmp_path):
    sha = "0123456789abcdef0123456789abcdef01234567"
    src = tmp_path / "archive"
    src.mkdir()
    _make_source_tree(str(src))
    (src / ".git_archival.txt").write_text(
        f"node: {sha}\nnode-date: 2026-01-01T00:00:00+00:00\n"
        "describe-name: v1.0.0-1-g0123456\n"
    )

    backend._record_full_revision(str(src))

    assert _read_recorded(str(src)) == sha


def test_ignores_enclosing_repository(backend, tmp_path):
    # an archive (no .git of its own) extracted inside an unrelated checkout
    # must not be attributed to the enclosing repository's commit
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "unrelated.txt").write_text("x")
    _git(outer, "init", "-q")
    _git(outer, "add", "-A")
    _git(outer, "commit", "-q", "-m", "outer")

    nested = outer / "extracted" / "archive"
    nested.mkdir(parents=True)
    _make_source_tree(str(nested))

    backend._record_full_revision(str(nested))

    assert _read_recorded(str(nested)) is None


def test_unexpanded_archival_placeholders_are_ignored(backend, tmp_path):
    # a plain checkout carries the archival file with $Format:...$ intact
    src = tmp_path / "srctree"
    src.mkdir()
    _make_source_tree(str(src))
    (src / ".git_archival.txt").write_text(
        "node: $Format:%H$\nnode-date: $Format:%cI$\n"
    )

    backend._record_full_revision(str(src))

    assert _read_recorded(str(src)) is None


def test_fallback_version_is_wheel_compatible():
    # regression: bdist_wheel does int() on the release parts of the version,
    # so a no-git build (e.g. the Dockerfile's dependency-skeleton layer)
    # crashes when fallback_version is not a numeric three-part release
    import re

    with open(os.path.join(_REPO_ROOT, "pyproject.toml")) as f:
        match = re.search(r'fallback_version\s*=\s*"([^"]+)"', f.read())
    assert match, "fallback_version must be configured for no-git builds"
    release = match.group(1).split("+")[0]
    parts = release.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


def test_existing_record_kept_without_revision_source(backend, tmp_path):
    # building a wheel from an sdist: no git, no expanded archival file --
    # the file recorded at sdist-build time must survive
    sha = "89abcdef0123456789abcdef0123456789abcdef"
    src = tmp_path / "sdist"
    src.mkdir()
    _make_source_tree(str(src))
    (src / "xinference" / "_commit.py").write_text(f'full_revisionid = "{sha}"\n')

    backend._record_full_revision(str(src))

    assert _read_recorded(str(src)) == sha


def test_validates_llm_family_specs(backend, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _write_llm_families(
        str(src),
        [{"model_name": "valid-family", "model_specs": []}],
    )

    backend._validate_builtin_model_specs(str(src))


def test_rejects_llm_family_without_model_specs(backend, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _write_llm_families(
        str(src),
        [
            {"model_name": "valid-family", "model_specs": []},
            {"model_name": "misplaced-audio-model", "model_family": "audio"},
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="misplaced-audio-model.*model_specs",
    ):
        backend._validate_builtin_model_specs(str(src))


def test_reports_entry_index_for_unnamed_invalid_llm_families(backend, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _write_llm_families(
        str(src),
        [
            {"model_name": None},
            {"model_name": ""},
            {},
        ],
    )

    with pytest.raises(RuntimeError) as exc_info:
        backend._validate_builtin_model_specs(str(src))

    message = str(exc_info.value)
    assert all(f"'entry {index}'" in message for index in range(3))


def test_pre_build_can_skip_model_validation_for_dependency_skeleton(
    backend, monkeypatch
):
    calls = []
    monkeypatch.setenv("XINFERENCE_SKIP_MODEL_SPEC_VALIDATION", "1")
    monkeypatch.setattr(
        backend, "_validate_builtin_model_specs", lambda: calls.append("validate")
    )
    monkeypatch.setattr(
        backend, "_record_full_revision", lambda: calls.append("record")
    )
    monkeypatch.setattr(backend, "build_web", lambda: calls.append("web"))

    backend._pre_build()

    assert calls == ["record", "web"]


def test_pre_build_validates_model_specs_by_default(backend, monkeypatch):
    calls = []
    monkeypatch.delenv("XINFERENCE_SKIP_MODEL_SPEC_VALIDATION", raising=False)
    monkeypatch.setattr(
        backend, "_validate_builtin_model_specs", lambda: calls.append("validate")
    )
    monkeypatch.setattr(
        backend, "_record_full_revision", lambda: calls.append("record")
    )
    monkeypatch.setattr(backend, "build_web", lambda: calls.append("web"))

    backend._pre_build()

    assert calls == ["validate", "record", "web"]


def test_project_declares_router_dependencies_and_all_includes_router():
    with open(os.path.join(_REPO_ROOT, "pyproject.toml"), "rb") as f:
        project = tomllib.load(f)

    metadata = project["project"]
    dependencies = metadata["dependencies"]
    optional_dependencies = metadata["optional-dependencies"]
    scripts = metadata["scripts"]

    assert "psutil>=5.9.0" in dependencies
    assert optional_dependencies["router"] == ["tokenizers>=0.21.0,<0.23.0"]
    assert "xinference[router]" in optional_dependencies["all"]

    assert scripts["xinference-router"] == "xinference.deploy.router:main"
    assert scripts["xinference-router-agent"] == "xinference.deploy.router_agent:main"
