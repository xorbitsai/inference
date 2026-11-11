# CI: linux
from types import SimpleNamespace

import pytest

from xinference.core.virtual_env_manager import apply_requested_virtualenv_indexes


def test_requested_indexes_override_and_merge():
    settings = SimpleNamespace(index_url="old", extra_index_url=["engine", "extra"])
    apply_requested_virtualenv_indexes(settings, " new ", ["extra", "custom"])
    assert settings.index_url == "new"
    assert settings.extra_index_url == ["extra", "custom", "engine"]


def test_unspecified_indexes_preserved():
    settings = SimpleNamespace(index_url="old", extra_index_url="engine")
    apply_requested_virtualenv_indexes(settings)
    assert settings.index_url == "old"
    assert settings.extra_index_url == "engine"


@pytest.mark.parametrize("index,extras", [([], None), (None, "bad"), (None, [1])])
def test_invalid_indexes(index, extras):
    with pytest.raises(ValueError):
        apply_requested_virtualenv_indexes(SimpleNamespace(), index, extras)


@pytest.mark.parametrize("index", [None, "", "   "])
def test_empty_request_preserves_worker_defaults(index):
    settings = SimpleNamespace(index_url="worker-default", extra_index_url=None)
    apply_requested_virtualenv_indexes(settings, index, [])
    assert settings.index_url == "worker-default"
    assert settings.extra_index_url is None


def test_request_sources_restored_after_engine_filtering():
    settings = SimpleNamespace(index_url="default", extra_index_url=["engine"])
    apply_requested_virtualenv_indexes(settings, "private", ["custom"])
    settings.extra_index_url = None
    apply_requested_virtualenv_indexes(settings, "private", ["custom"])
    assert settings.index_url == "private"
    assert settings.extra_index_url == ["custom"]
