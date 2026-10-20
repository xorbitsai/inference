from types import SimpleNamespace
from unittest.mock import Mock

from .. import core
from ..core import RerankModel


def make_model():
    family = SimpleNamespace(
        type="normal", model_specs=[SimpleNamespace(quantization="none")]
    )
    return RerankModel("test", "/model", family, "none")


def test_cleanup_keeps_cache_release_triggers_and_throttles_full_gc(monkeypatch):
    model = make_model()
    clock = [100.0]
    collect, release = Mock(), Mock()
    monkeypatch.setattr(core, "RERANK_EMPTY_CACHE_COUNT", 10)
    monkeypatch.setattr(core, "RERANK_GC_INTERVAL", 60)
    monkeypatch.setattr(core, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(core.gc, "collect", collect)
    monkeypatch.setattr(core, "empty_cache", release)
    for now in [100.0, 130.0, 159.0, 160.0]:
        clock[0] = now
        for _ in range(9):
            model._clean_cache_if_needed()
        previous = release.call_count
        model._clean_cache_if_needed()
        assert release.call_count == previous + 1
    assert collect.call_count == 2
    assert release.call_count == 4


def test_gc_interval_starts_after_collection_finishes(monkeypatch):
    model = make_model()
    clock = [100.0]
    collect = Mock(side_effect=lambda: clock.__setitem__(0, clock[0] + 5))
    monkeypatch.setattr(core, "RERANK_EMPTY_CACHE_COUNT", 1)
    monkeypatch.setattr(core, "RERANK_GC_INTERVAL", 60)
    monkeypatch.setattr(core, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(core.gc, "collect", collect)
    monkeypatch.setattr(core, "empty_cache", Mock())
    model._clean_cache_if_needed()
    clock[0] = 160.0
    model._clean_cache_if_needed()
    assert collect.call_count == 1
    clock[0] = 165.0
    model._clean_cache_if_needed()
    assert collect.call_count == 2


def test_zero_interval_collects_on_every_cleanup(monkeypatch):
    model = make_model()
    collect, release = Mock(), Mock()
    monkeypatch.setattr(core, "RERANK_EMPTY_CACHE_COUNT", 10)
    monkeypatch.setattr(core, "RERANK_GC_INTERVAL", 0)
    monkeypatch.setattr(core.gc, "collect", collect)
    monkeypatch.setattr(core, "empty_cache", release)
    for _ in range(30):
        model._clean_cache_if_needed()
    assert collect.call_count == release.call_count == 3


def test_gc_cooldown_is_per_model(monkeypatch):
    collect, release = Mock(), Mock()
    monkeypatch.setattr(core, "RERANK_EMPTY_CACHE_COUNT", 1)
    monkeypatch.setattr(core, "RERANK_GC_INTERVAL", 60)
    monkeypatch.setattr(core, "time", SimpleNamespace(monotonic=lambda: 100.0))
    monkeypatch.setattr(core.gc, "collect", collect)
    monkeypatch.setattr(core, "empty_cache", release)
    first, second = make_model(), make_model()
    for model in [first, second, first]:
        model._clean_cache_if_needed()
    assert collect.call_count == 2
    assert release.call_count == 3


def test_sync_vllm_rerank_uses_the_gc_cooldown(monkeypatch):
    from ..vllm.core import VLLMRerankModel

    family = SimpleNamespace(
        model_name="bge-reranker-v2-m3", model_specs=[None], type="normal"
    )
    model = VLLMRerankModel("test", "/model", family, "none")
    model._model = SimpleNamespace(score=Mock(return_value=["score"]))
    collect, release = Mock(), Mock()
    monkeypatch.setattr(core, "RERANK_EMPTY_CACHE_COUNT", 10)
    monkeypatch.setattr(core, "RERANK_GC_INTERVAL", 60)
    monkeypatch.setattr(core, "time", SimpleNamespace(monotonic=lambda: 100.0))
    monkeypatch.setattr(core.gc, "collect", collect)
    monkeypatch.setattr(core, "empty_cache", release)
    for _ in range(30):
        assert model._rerank(["document"], "query") == ["score"]
    assert model._counter == model._model.score.call_count == 30
    assert collect.call_count == 1
    assert release.call_count == 3
