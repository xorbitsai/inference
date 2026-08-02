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


from types import SimpleNamespace


def _model():
    from ..core import SGLANGModel

    model = object.__new__(SGLANGModel)
    model.model_uid = "test-model-0"
    model.model_family = SimpleNamespace(model_name="qwen2")
    model.model_spec = SimpleNamespace(model_size_in_billions=7)
    return model


def _meta_info_with_logprobs():
    # sglang meta_info when return_logprob=True + return_text_in_logprobs=True:
    # each sampled position is a (logprob, token_id, token_text) 3-tuple, and
    # each top position is None or a list of the same 3-tuples.
    return {
        "finish_reason": {"type": "stop", "status_code": 200},
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "output_token_logprobs": [
            (-0.5, 4398, "Hello"),
            (-1.2, 290, " world"),
        ],
        "output_top_logprobs": [
            [(-0.5, 4398, "Hello"), (-3.0, 912, "Hi")],
            [(-1.2, 290, " world"), (-2.5, 818, " there")],
        ],
    }


def test_non_stream_logprobs_surfaced():
    # red on master: _convert_state_to_completion hardcodes logprobs=None, so
    # this assertion fails there and turns green on the branch.
    completion = _model()._convert_state_to_completion(
        "req-1", "test-model-0", "Hello world", _meta_info_with_logprobs()
    )
    lp = completion["choices"][0]["logprobs"]

    assert lp is not None
    assert lp["tokens"] == ["Hello", " world"]
    assert lp["text_offset"] == [0, 5]
    assert lp["token_logprobs"] == [-0.5, -1.2]
    assert lp["top_logprobs"][0] == {"Hello": -0.5, "Hi": -3.0}
    assert lp["top_logprobs"][1] == {" world": -1.2, " there": -2.5}


def test_stream_chunk_logprobs_surfaced():
    # a single-token streaming chunk's meta_info; mirrors the non-stream shape
    meta = {
        "finish_reason": None,
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "output_token_logprobs": [(-0.5, 4398, "Hello")],
        "output_top_logprobs": [[(-0.5, 4398, "Hello"), (-3.0, 912, "Hi")]],
    }
    chunk = _model()._convert_state_to_completion_chunk(
        "req-1", "test-model-0", "Hello", meta
    )
    lp = chunk["choices"][0]["logprobs"]

    assert lp is not None
    assert lp["tokens"] == ["Hello"]
    assert lp["text_offset"] == [0]
    assert lp["token_logprobs"] == [-0.5]
    assert lp["top_logprobs"][0] == {"Hello": -0.5, "Hi": -3.0}


def test_logprob_floor_applied():
    # mirror vllm: logprobs below -9999.0 are floored, never fabricated
    meta = {
        "finish_reason": {"type": "stop"},
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "output_token_logprobs": [(-15000.0, 9, "x")],
        "output_top_logprobs": [[(-20000.0, 9, "x")]],
    }
    lp = _model()._convert_state_to_completion("r", "m", "x", meta)["choices"][0][
        "logprobs"
    ]

    assert lp is not None
    assert lp["token_logprobs"] == [-9999.0]
    assert lp["top_logprobs"][0] == {"x": -9999.0}


def test_no_logprob_data_degrades_to_none():
    # defensive: caller did not request return_logprob -> meta_info carries no
    # output_token_logprobs -> None, no crash, no fabricated probabilities
    meta = {"finish_reason": {"type": "stop"}, "prompt_tokens": 2, "completion_tokens": 1}
    lp = _model()._convert_state_to_completion("r", "m", "x", meta)["choices"][0][
        "logprobs"
    ]
    assert lp is None


def test_empty_logprob_list_degrades_to_none():
    meta = {
        "finish_reason": {"type": "stop"},
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "output_token_logprobs": [],
    }
    lp = _model()._convert_state_to_completion("r", "m", "x", meta)["choices"][0][
        "logprobs"
    ]
    assert lp is None


def test_malformed_entries_degrade_gracefully():
    meta = {
        "finish_reason": {"type": "stop"},
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "output_token_logprobs": [(-0.5, 1, "a"), None],
        "output_top_logprobs": [[(-0.5, 1, "a")], None],
    }
    lp = _model()._build_logprobs(meta)

    assert lp is not None
    assert lp["token_logprobs"][0] == -0.5
    assert lp["token_logprobs"][1] is None
    assert lp["tokens"][1] == ""
    assert lp["top_logprobs"][0] == {"a": -0.5}
    assert lp["top_logprobs"][1] is None


def test_older_two_tuple_and_dict_shapes_read_defensively():
    # older sglang emitted (token_id, logprob) pairs without decoded text, and
    # the top field was named output_topk_logprobs holding a decoded_text ->
    # logprob dict. The adapter recovers the logprob by type and degrades only
    # the missing text, without crashing or fabricating.
    meta = {
        "finish_reason": {"type": "stop"},
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "output_token_logprobs": [(4398, -0.5)],
        "output_topk_logprobs": [{"Hello": -0.5}],
    }
    lp = _model()._build_logprobs(meta)

    assert lp is not None
    assert lp["token_logprobs"] == [-0.5]
    assert lp["tokens"] == [""]
    assert lp["top_logprobs"][0] == {"Hello": -0.5}


def test_sanitize_maps_chat_logprobs_to_sglang_params():
    from ..core import SGLANGModel

    # chat completions: logprobs is a bool flag, top_logprobs holds the count
    cfg = SGLANGModel._sanitize_generate_config(
        {"logprobs": True, "top_logprobs": 3}
    )
    assert cfg["return_logprob"] is True
    assert cfg["top_logprobs_num"] == 3
    assert cfg["return_text_in_logprobs"] is True
    # raw OpenAI keys must not survive (#3553: they crash sglang SamplingParams)
    assert "logprobs" not in cfg
    assert "top_logprobs" not in cfg


def test_sanitize_maps_legacy_completion_logprobs_count():
    from ..core import SGLANGModel

    # legacy /v1/completions: logprobs is the requested count directly
    cfg = SGLANGModel._sanitize_generate_config({"logprobs": 5})
    assert cfg["return_logprob"] is True
    assert cfg["top_logprobs_num"] == 5
    assert cfg["return_text_in_logprobs"] is True
    assert "logprobs" not in cfg


def test_sanitize_without_logprob_request_adds_no_sglang_params():
    from ..core import SGLANGModel

    cfg = SGLANGModel._sanitize_generate_config({"temperature": 0.7})
    assert "return_logprob" not in cfg
    assert "top_logprobs_num" not in cfg
    assert "return_text_in_logprobs" not in cfg


def test_lift_logprob_request_params_moves_to_top_level():
    from ..core import SGLANGModel

    sampling_params = {
        "temperature": 0.7,
        "return_logprob": True,
        "top_logprobs_num": 3,
        "return_text_in_logprobs": True,
    }
    top = SGLANGModel._lift_logprob_request_params(sampling_params)

    # the logprob request fields are lifted out of sampling_params so sglang
    # reads them as top-level GenerateReqInput fields, not SamplingParams (#3553)
    assert sampling_params == {"temperature": 0.7}
    assert top == {
        "return_logprob": True,
        "top_logprobs_num": 3,
        "return_text_in_logprobs": True,
    }


def test_lift_logprob_request_params_is_empty_when_unrequested():
    from ..core import SGLANGModel

    sampling_params = {"temperature": 0.7}
    assert SGLANGModel._lift_logprob_request_params(sampling_params) == {}
    assert sampling_params == {"temperature": 0.7}
