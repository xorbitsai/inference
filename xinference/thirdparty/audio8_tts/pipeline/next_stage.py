# SPDX-License-Identifier: Apache-2.0
from typing import Any

from xinference.thirdparty.audio8_tts.pipeline.state_io import load_state


def preprocessing_next(request_id: str, output: Any) -> str:
    del request_id, output
    return "tts_engine"


def tts_engine_next(request_id: str, output: Any) -> str:
    del request_id
    state = load_state(output)
    if state.response_format in {"codes", "codec", "npy"}:
        return None
    return "vocoder"


def vocoder_next(request_id: str, output: Any) -> None:
    del request_id, output
    return None
