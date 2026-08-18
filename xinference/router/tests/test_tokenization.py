import asyncio
import os
import time
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from xinference.router.admission import AdmissionRejected
from xinference.router.metrics import RouterMetrics
from xinference.router.tokenization import TokenizationService
from xinference.router.tokenizer import TokenizationError


def make_assets(tmp_path: Path) -> Path:
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    encoding = tmp_path / "encoding"
    encoding.mkdir()
    (encoding / "encoding_dsv4.py").write_text(
        "import time\n"
        "\n"
        "def encode_messages(messages, thinking_mode, reasoning_effort=None):\n"
        "    content = ' '.join(str(m.get('content', '')) for m in messages)\n"
        "    if content.startswith('__cpu__:'):\n"
        "        duration = float(content.split(':', 1)[1])\n"
        "        deadline = time.perf_counter() + duration\n"
        "        while time.perf_counter() < deadline:\n"
        "            pass\n"
        "        return 'hello'\n"
        "    if content == '__error__':\n"
        "        raise RuntimeError('synthetic rendering failure')\n"
        "    return content\n"
    )
    return tmp_path


def payload(content: str = "hello") -> dict:
    return {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def make_service(
    tmp_path: Path,
    *,
    max_workers: int = 1,
    max_active: int = 1,
    max_queue: int = 1,
    queue_timeout_seconds: float = 1,
    retry_after_seconds: int = 1,
) -> tuple[TokenizationService, RouterMetrics]:
    metrics = RouterMetrics()
    service = TokenizationService(
        make_assets(tmp_path),
        metrics,
        reserve_tokens=0,
        default_output_tokens=1,
        max_workers=max_workers,
        max_active=max_active,
        max_queue=max_queue,
        queue_timeout_seconds=queue_timeout_seconds,
        retry_after_seconds=retry_after_seconds,
    )
    return service, metrics


async def wait_for_active(service: TokenizationService, expected: int = 1) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if (await service.snapshot()).active == expected:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"tokenization active count did not become {expected}")


@pytest.mark.asyncio
async def test_spawn_workers_are_prestarted_and_remove_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XINFERENCE_API_KEY", "parent-secret")
    service, _ = make_service(
        tmp_path,
        max_workers=2,
        max_active=2,
    )
    try:
        await service.start()
        assert len(service.worker_pids) == 2
        assert os.getpid() not in service.worker_pids
        assert os.environ["XINFERENCE_API_KEY"] == "parent-secret"
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_process_tokenization_does_not_block_event_loop(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path)
    try:
        await service.start()
        task = asyncio.create_task(
            service.estimate(payload("__cpu__:0.3"), input_bytes=123)
        )
        await wait_for_active(service)

        ticks = 0
        deadline = time.monotonic() + 0.15
        while time.monotonic() < deadline:
            await asyncio.sleep(0.01)
            ticks += 1

        assert ticks >= 8
        assert not task.done()
        result = await task
        assert result.prompt_tokens == 1
        assert (await service.snapshot()).active == 0
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_tokenization_queue_full_is_rejected_and_observed(
    tmp_path: Path,
) -> None:
    service, metrics = make_service(
        tmp_path,
        max_queue=0,
        queue_timeout_seconds=0,
        retry_after_seconds=3,
    )
    try:
        await service.start()
        first = asyncio.create_task(
            service.estimate(payload("__cpu__:0.2"), input_bytes=10)
        )
        await wait_for_active(service)
        with pytest.raises(AdmissionRejected) as exc_info:
            await service.estimate(payload(), input_bytes=20)
        assert exc_info.value.reason == "queue_full"
        assert exc_info.value.retry_after_seconds == 3
        await first
        rendered = await metrics.render()
        assert 'reason="queue_full"} 1' in rendered
        assert "xinference_token_router_tokenization_input_bytes_sum 10" in rendered
        assert "xinference_token_router_tokenization_active 0" in rendered
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_cancelled_caller_keeps_slot_until_worker_finishes(
    tmp_path: Path,
) -> None:
    service, metrics = make_service(
        tmp_path,
        max_queue=0,
        queue_timeout_seconds=0,
    )
    try:
        await service.start()
        task = asyncio.create_task(
            service.estimate(payload("__cpu__:0.25"), input_bytes=10)
        )
        await wait_for_active(service)
        task.cancel()
        await asyncio.sleep(0.03)
        assert (await service.snapshot()).active == 1
        with pytest.raises(AdmissionRejected):
            await service.estimate(payload(), input_bytes=20)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await service.snapshot()).active == 0
        rendered = await metrics.render()
        assert 'outcome="cancelled"} 1' in rendered
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_tokenization_error_crosses_process_boundary(tmp_path: Path) -> None:
    service, metrics = make_service(tmp_path)
    try:
        await service.start()
        with pytest.raises(TokenizationError, match="synthetic rendering failure"):
            await service.estimate(payload("__error__"), input_bytes=11)
        assert (await service.snapshot()).active == 0
        rendered = await metrics.render()
        assert 'outcome="failed"} 1' in rendered
    finally:
        await service.aclose()
