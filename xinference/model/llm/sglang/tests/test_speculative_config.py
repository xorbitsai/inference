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


def _model():
    from ..core import SGLANGModel

    model = object.__new__(SGLANGModel)
    model.model_uid = "test-model-0"
    return model


def test_drafter_becomes_nextn_server_args():
    model_config = {"draft_model_path": "/cache/gemma-4-draft", "tp_size": 1}

    _model()._apply_draft_model(model_config)

    assert model_config["speculative_algorithm"] == "NEXTN"
    assert model_config["speculative_draft_model_path"] == "/cache/gemma-4-draft"
    assert model_config["speculative_num_draft_tokens"] == 6
    assert model_config["speculative_num_steps"] == 5
    assert model_config["speculative_eagle_topk"] == 1
    # the whole model_config is splatted into the engine, so the neutral
    # options must not survive
    assert "draft_model_path" not in model_config
    assert "num_speculative_tokens" not in model_config
    assert model_config["tp_size"] == 1


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
