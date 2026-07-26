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


def _model(model_config):
    from ..core import XllamaCppModel

    model = object.__new__(XllamaCppModel)
    model.model_uid = "test-model-0"
    model._llamacpp_model_config = model_config
    return model


def _params():
    # mirrors the shape of xllamacpp's CommonParams.speculative
    return SimpleNamespace(
        speculative=SimpleNamespace(
            types=[],
            draft=SimpleNamespace(n_max=3, mparams=SimpleNamespace(path="")),
        )
    )


def test_drafter_file_is_wired_into_params(tmp_path):
    drafter = tmp_path / "mtp-gemma-4-12b-it-BF16.gguf"
    drafter.write_text("gguf")
    model_config = {"draft_model_path": str(drafter), "n_ctx": 4096}
    params = _params()

    model = _model(model_config)
    model._apply_draft_model(params, *model._draft_options())

    assert params.speculative.draft.mparams.path == str(drafter)
    from xllamacpp import common_speculative_type

    assert params.speculative.types == [
        common_speculative_type.COMMON_SPECULATIVE_TYPE_DRAFT_MTP
    ]
    # they stay in the config so a load retry still finds them; the passthrough
    # loop skips them instead
    assert model_config["n_ctx"] == 4096


def test_cache_dir_resolves_to_the_single_gguf(tmp_path):
    # the drafter cache dir mirrors the repo layout, MTP/ subdirectory included
    cache_dir = tmp_path / "gemma-4-ggufv2-12b-Q4_K_M-draft-BF16"
    (cache_dir / "MTP").mkdir(parents=True)
    drafter = cache_dir / "MTP" / "mtp-gemma-4-12b-it-BF16.gguf"
    drafter.write_text("gguf")
    params = _params()

    model = _model({"draft_model_path": str(cache_dir)})
    model._apply_draft_model(params, *model._draft_options())

    assert params.speculative.draft.mparams.path == str(drafter)


def test_cache_dir_resolves_a_flat_gguf(tmp_path):
    # a drafter may also sit directly in the cache dir; `**` without
    # recursive=True collapses to a single level and would miss it
    cache_dir = tmp_path / "gemma-4-ggufv2-2b-Q4_K_M-draft-BF16"
    cache_dir.mkdir()
    drafter = cache_dir / "mtp-gemma-4-E2B-it-BF16.gguf"
    drafter.write_text("gguf")
    params = _params()

    model = _model({"draft_model_path": str(cache_dir)})
    model._apply_draft_model(params, *model._draft_options())

    assert params.speculative.draft.mparams.path == str(drafter)


def test_num_speculative_tokens_sets_n_max(tmp_path):
    drafter = tmp_path / "d.gguf"
    drafter.write_text("gguf")
    params = _params()

    # the Web UI submits additional parameters as strings
    model = _model({"draft_model_path": str(drafter), "num_speculative_tokens": "5"})
    model._apply_draft_model(params, *model._draft_options())

    assert params.speculative.draft.n_max == 5


def test_empty_cache_dir_is_reported(tmp_path):
    params = _params()

    with pytest.raises(ValueError, match="No gguf drafter found"):
        model = _model({"draft_model_path": str(tmp_path)})
        model._apply_draft_model(params, *model._draft_options())


def test_tuning_knobs_do_not_disable_the_drafter(tmp_path):
    # Only the mode selectors mean "the user is driving this themselves"; a
    # tuning knob must not silently turn speculative decoding off.
    drafter = tmp_path / "d.gguf"
    drafter.write_text("gguf")
    model_config = {
        "draft_model_path": str(drafter),
        "speculative.draft.n_min": 2,
        "speculative.draft.p_min": 0.5,
    }
    params = _params()

    model = _model(model_config)
    model._apply_draft_model(params, *model._draft_options())

    assert params.speculative.draft.mparams.path == str(drafter)
    assert len(params.speculative.types) == 1


def test_draft_options_survive_a_second_load(tmp_path):
    # The load retry reuses the instance and its config, so reading these must
    # not consume them.
    drafter = tmp_path / "d.gguf"
    drafter.write_text("gguf")
    model_config = {"draft_model_path": str(drafter), "num_speculative_tokens": 5}

    model = _model(model_config)

    assert model._draft_options() == (str(drafter), 5)
    assert model._draft_options() == (str(drafter), 5)
    assert model_config["draft_model_path"] == str(drafter)


def test_explicit_speculative_params_win(tmp_path):
    # A user driving llama.cpp's speculative decoding through the dotted-key
    # passthrough must not have it clobbered, matching vLLM and SGLang.
    drafter = tmp_path / "d.gguf"
    drafter.write_text("gguf")
    model_config = {
        "draft_model_path": str(drafter),
        "speculative.draft.mparams.path": "/somewhere/mine.gguf",
    }
    params = _params()

    model = _model(model_config)
    model._apply_draft_model(params, *model._draft_options())

    assert params.speculative.types == []
    assert params.speculative.draft.mparams.path == ""


def test_neutral_options_are_skipped_by_the_passthrough_loop():
    # They are not CommonParams attributes, so the loop must skip them or it
    # logs a failure for each one on every speculative launch.
    from ..core import XllamaCppModel

    assert set(XllamaCppModel.DRAFT_OPTION_KEYS) == {
        "draft_model_path",
        "num_speculative_tokens",
    }


def test_no_drafter_is_a_noop():
    model_config = {"n_ctx": 4096}
    params = _params()

    model = _model(model_config)
    model._apply_draft_model(params, *model._draft_options())

    assert params.speculative.types == []
    assert params.speculative.draft.mparams.path == ""
    assert model_config == {"n_ctx": 4096}
