# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Opaque signatures for reasoning produced by this Xinference endpoint."""

import hashlib
import hmac
import json
from functools import lru_cache


@lru_cache(maxsize=1)
def _key() -> bytes:
    from ...constants import _get_or_create_persisted_secret

    return _get_or_create_persisted_secret(
        "XINFERENCE_ANTHROPIC_THINKING_SECRET", "anthropic_thinking_key"
    ).encode()


def sign_thinking(text: str, model: str) -> str:
    payload = json.dumps([model, text], ensure_ascii=False).encode()
    return "xinference_v1:" + hmac.new(_key(), payload, hashlib.sha256).hexdigest()


def verify_thinking(text: str, model: str, signature: str) -> bool:
    return hmac.compare_digest(sign_thinking(text, model), signature)
