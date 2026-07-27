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

import pytest


def _model(model_name="gemma-4", model_size=26):
    from ..core import SGLANGModel

    model = object.__new__(SGLANGModel)
    model.model_uid = "test-model-0"
    model.model_family = SimpleNamespace(model_name=model_name)
    model.model_spec = SimpleNamespace(model_size_in_billions=model_size)
    return model


def test_drafter_becomes_nextn_server_args():
    model_config = {"draft_model_path": "/cache/gemma-4-draft", "tp_size": 1}

    _model()._apply_draft_model(model_config)

    assert model_config["speculative_algorithm"] == "NEXTN"
    assert model_config["speculative_draft_model_path"] == "/cache/gemma-4-draft"
    assert model_config["speculative_num_draft_tokens"] == 4
    assert model_config["speculative_num_steps"] == 3
    assert model_config["speculative_eagle_topk"] == 1
    # the whole model_config is splatted into the engine, so the neutral
    # options must not survive
    assert "draft_model_path" not in model_config
    assert "num_speculative_tokens" not in model_config
    assert model_config["tp_size"] == 1


@pytest.mark.parametrize(
    ("model_size", "expected"),
    [
        (2, 2),
        (4, 4),
        (12, 4),
        (26, 4),
        (31, 4),
    ],
)
def test_gemma_4_recipe_default_by_size(model_size, expected):
    model_config = {"draft_model_path": "/cache/draft"}

    _model(model_size=model_size)._apply_draft_model(model_config)

    assert model_config["speculative_num_draft_tokens"] == expected
    assert model_config["speculative_num_steps"] == max(1, expected - 1)


def test_other_nextn_family_keeps_generic_default():
    model_config = {"draft_model_path": "/cache/draft"}

    _model(model_name="other-nextn")._apply_draft_model(model_config)

    assert model_config["speculative_num_draft_tokens"] == 6
    assert model_config["speculative_num_steps"] == 5


def test_num_speculative_tokens_drives_the_step_count():
    # the Web UI submits additional parameters as strings
    model_config = {"draft_model_path": "/cache/draft", "num_speculative_tokens": "4"}

    _model()._apply_draft_model(model_config)

    assert model_config["speculative_num_draft_tokens"] == 4
    assert model_config["speculative_num_steps"] == 3


def test_explicit_draft_token_count_wins():
    # The guard above only looks at speculative_algorithm, so an explicitly
    # supplied count must survive here too, like the two keys beside it.
    model_config = {
        "draft_model_path": "/cache/draft",
        "speculative_num_draft_tokens": 3,
    }

    _model()._apply_draft_model(model_config)

    assert model_config["speculative_num_draft_tokens"] == 3
    # the step count follows the value actually in effect
    assert model_config["speculative_num_steps"] == 2


def test_explicit_draft_token_count_is_coerced():
    # the Web UI submits strings, and the engine expects an int
    model_config = {
        "draft_model_path": "/cache/draft",
        "speculative_num_draft_tokens": "3",
    }

    _model()._apply_draft_model(model_config)

    assert model_config["speculative_num_draft_tokens"] == 3
    assert model_config["speculative_num_steps"] == 2


@pytest.mark.parametrize("invalid", [0, "0", -2, 2.7, "abc"])
def test_invalid_native_draft_token_count_is_rejected(invalid):
    # zero draft tokens with NEXTN on is self-contradictory, and 2.7 truncating
    # to 2 is the silent substitution the shared validator exists to prevent
    model_config = {
        "draft_model_path": "/cache/draft",
        "speculative_num_draft_tokens": invalid,
    }

    with pytest.raises(ValueError, match="positive integer"):
        _model()._apply_draft_model(model_config)


@pytest.mark.parametrize("invalid", [0, "0", -1, 1.5, "abc"])
def test_invalid_neutral_token_count_is_rejected(invalid):
    model_config = {
        "draft_model_path": "/cache/draft",
        "num_speculative_tokens": invalid,
    }

    with pytest.raises(ValueError, match="positive integer"):
        _model()._apply_draft_model(model_config)


def test_explicit_speculative_algorithm_wins():
    model_config = {
        "draft_model_path": "/cache/draft",
        "speculative_algorithm": "EAGLE",
        "speculative_draft_model_path": "/cache/eagle",
    }

    _model()._apply_draft_model(model_config)

    assert model_config["speculative_algorithm"] == "EAGLE"
    assert model_config["speculative_draft_model_path"] == "/cache/eagle"
    assert "draft_model_path" not in model_config


def test_no_drafter_is_a_noop():
    model_config = {"tp_size": 2}

    _model()._apply_draft_model(model_config)

    assert model_config == {"tp_size": 2}
