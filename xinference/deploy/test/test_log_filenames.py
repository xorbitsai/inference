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

import logging
import subprocess
import sys

import pytest

from ..utils import (
    SafeRotatingFileHandler,
    SafeTimedAndSizeRotatingFileHandler,
    SafeTimedRotatingFileHandler,
)


def _log_filename(path_kind, tmp_path):
    if path_kind == "bare":
        return "xinference.log"
    if path_kind == "nested":
        return "logs/nested/xinference.log"
    return str(tmp_path / "absolute" / "nested" / "xinference.log")


@pytest.mark.parametrize("path_kind", ["bare", "nested", "absolute"])
@pytest.mark.parametrize(
    "handler_class,options",
    [
        (SafeRotatingFileHandler, {"maxBytes": 1024}),
        (SafeTimedRotatingFileHandler, {"when": "midnight"}),
        (
            SafeTimedAndSizeRotatingFileHandler,
            {"when": "midnight", "maxBytes": 1024},
        ),
    ],
)
def test_safe_handler_filename(
    handler_class, options, path_kind, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    filename = _log_filename(path_kind, tmp_path)
    handler = handler_class(filename, backupCount=2, encoding="utf8", **options)
    try:
        record = logging.makeLogRecord({"msg": "before rollover"})
        handler.emit(record)
        handler.flush()
        with open(filename) as log:
            assert "before rollover" in log.read()

        handler.doRollover()
        handler.emit(logging.makeLogRecord({"msg": "after rollover"}))
        handler.flush()
        with open(filename) as log:
            assert "after rollover" in log.read()
        assert list(tmp_path.rglob("*.lock"))
    finally:
        handler.close()


@pytest.mark.parametrize("path_kind", ["bare", "nested", "absolute"])
@pytest.mark.parametrize("rotation", ["size", "daily", "daily+size"])
def test_logging_config_filename(rotation, path_kind, tmp_path):
    filename = _log_filename(path_kind, tmp_path)
    # dictConfig closes existing handlers. Isolate it from pytest's logging state.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import logging
import logging.config
import sys
from pathlib import Path
from xinference.deploy.utils import get_config_dict

filename, rotation = sys.argv[1:]
try:
    logging.config.dictConfig(get_config_dict("INFO", filename, 2, 1024, rotation=rotation))
    logging.getLogger("xinference").info("configured log record")
    for handler in logging.getLogger("xinference").handlers:
        handler.flush()
    assert "configured log record" in Path(filename).read_text()
finally:
    logging.shutdown()
""",
            filename,
            rotation,
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("path_kind", ["bare", "nested", "absolute"])
@pytest.mark.parametrize(
    "handler_class,options",
    [
        (SafeRotatingFileHandler, {"maxBytes": 1024}),
        (SafeTimedRotatingFileHandler, {"when": "midnight"}),
        (
            SafeTimedAndSizeRotatingFileHandler,
            {"when": "midnight", "maxBytes": 1024},
        ),
    ],
)
def test_safe_handler_after_chdir(
    handler_class, options, path_kind, tmp_path, monkeypatch
):
    original_cwd = tmp_path / "original"
    original_cwd.mkdir()
    monkeypatch.chdir(original_cwd)
    filename = _log_filename(path_kind, original_cwd)
    handler = handler_class(filename, backupCount=2, encoding="utf8", **options)
    log_path = original_cwd / filename
    lock_path = log_path.with_name(log_path.name + ".rotate.lock")
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    try:
        assert lock_path.is_file()
        monkeypatch.chdir(other_cwd)
        handler.emit(logging.makeLogRecord({"msg": "before rollover"}))
        handler.doRollover()
        handler.emit(logging.makeLogRecord({"msg": "after chdir rollover"}))
        handler.flush()
        assert "after chdir rollover" in log_path.read_text()
        # Rollover must update the original lock's coordination state.
        assert lock_path.stat().st_size > 1
        assert not list(other_cwd.iterdir())
    finally:
        handler.close()
