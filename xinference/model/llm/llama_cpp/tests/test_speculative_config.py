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

    _model(model_config)._apply_draft_model(params)

    assert params.speculative.draft.mparams.path == str(drafter)
    from xllamacpp import common_speculative_type

    assert params.speculative.types == [
        common_speculative_type.COMMON_SPECULATIVE_TYPE_DRAFT_MTP
    ]
    # the engine-neutral options must not stay in the passthrough config
    assert "draft_model_path" not in model_config
    assert model_config == {"n_ctx": 4096}


def test_cache_dir_resolves_to_the_single_gguf(tmp_path):
    # the drafter cache dir mirrors the repo layout, MTP/ subdirectory included
    cache_dir = tmp_path / "gemma-4-ggufv2-12b-Q4_K_M-draft-BF16"
    (cache_dir / "MTP").mkdir(parents=True)
    drafter = cache_dir / "MTP" / "mtp-gemma-4-12b-it-BF16.gguf"
    drafter.write_text("gguf")
    params = _params()

    _model({"draft_model_path": str(cache_dir)})._apply_draft_model(params)

    assert params.speculative.draft.mparams.path == str(drafter)


def test_num_speculative_tokens_sets_n_max(tmp_path):
    drafter = tmp_path / "d.gguf"
    drafter.write_text("gguf")
    params = _params()

    # the Web UI submits additional parameters as strings
    _model(
        {"draft_model_path": str(drafter), "num_speculative_tokens": "5"}
    )._apply_draft_model(params)

    assert params.speculative.draft.n_max == 5


def test_empty_cache_dir_is_reported(tmp_path):
    params = _params()

    with pytest.raises(ValueError, match="No gguf drafter found"):
        _model({"draft_model_path": str(tmp_path)})._apply_draft_model(params)


def test_no_drafter_is_a_noop():
    model_config = {"n_ctx": 4096}
    params = _params()

    _model(model_config)._apply_draft_model(params)

    assert params.speculative.types == []
    assert params.speculative.draft.mparams.path == ""
    assert model_config == {"n_ctx": 4096}
