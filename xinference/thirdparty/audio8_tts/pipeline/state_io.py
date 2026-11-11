# SPDX-License-Identifier: Apache-2.0
from xinference.thirdparty.audio8_tts.io import Audio8TTSState
from sglang_omni.proto import StagePayload


def load_state(payload: StagePayload) -> Audio8TTSState:
    return Audio8TTSState.from_dict(payload.data)


def store_state(payload: StagePayload, state: Audio8TTSState) -> StagePayload:
    payload.data = state.to_dict()
    return payload
