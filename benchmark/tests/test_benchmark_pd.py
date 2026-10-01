import importlib.util
from pathlib import Path
from unittest.mock import Mock

import psutil
import pytest

spec = importlib.util.spec_from_file_location(
    "benchmark_pd", Path(__file__).parents[1] / "benchmark_pd.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_goodput_excludes_errors_and_missing_token_timing():
    records = [
        {"ttft_s": 0.1, "tpot_s": 0.02, "latency_s": 0.2, "output_tokens": 6},
        {"ttft_s": 3, "tpot_s": 0.01, "latency_s": 4, "output_tokens": 101},
        {"ttft_s": 0.1, "tpot_s": None, "latency_s": 0.1, "output_tokens": 1},
        {"error": "timeout", "latency_s": 5},
    ]
    result = module.summarize(records, 10, 2, 0.05)
    assert result["successful"] == 3
    assert result["slo_eligible_requests"] == 2
    assert result["goodput_req_s"] == 0.1
    assert result["output_tokens_s"] == 10.8
    assert result["ttft_s"]["p99"] == 3


@pytest.mark.parametrize("stage", ["parent", "children", "memory"])
@pytest.mark.parametrize("error", [psutil.NoSuchProcess, psutil.AccessDenied])
def test_rss_sampling_recovers_from_process_races(monkeypatch, stage, error):
    child = Mock()
    child.memory_info.return_value.rss = 20
    parent = Mock()
    parent.memory_info.return_value.rss = 100
    parent.children.return_value = [child]
    lookup = Mock(return_value=parent)
    monkeypatch.setattr(psutil, "Process", lookup)
    target = {
        "parent": lookup,
        "children": parent.children,
        "memory": child.memory_info,
    }[stage]
    target.side_effect = error(123)
    assert module.process_tree_rss(123) is None
    target.side_effect = None
    assert module.process_tree_rss(123) == 120
