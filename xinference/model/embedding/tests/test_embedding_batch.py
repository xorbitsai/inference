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

from xoscar.batch import _ExtensibleWrapper

from ....types import Embedding, EmbeddingData, EmbeddingUsage
from ..core import EmbeddingModel, EmbeddingModelFamilyV2, TransformersEmbeddingSpecV1

_STUB_SPEC = EmbeddingModelFamilyV2(
    version=2,
    model_name="stub-embedding",
    dimensions=8,
    max_tokens=512,
    language=["en"],
    model_specs=[
        TransformersEmbeddingSpecV1(
            model_format="pytorch",
            model_id="stub/stub-embedding",
            model_revision="0000000000000000000000000000000000000000",
            quantization="none",
        )
    ],
)


class _StubEmbeddingModel(EmbeddingModel):
    """EmbeddingModel whose ``_create_embedding`` needs no model download.

    It mimics the real engines (sentence_transformers, flag, vllm, ...) by
    numbering ``EmbeddingData.index`` with ``enumerate`` over the *entire* input
    list it receives. Under batching that list is the concatenation of every
    caller's inputs, so the produced indices are global to the batch — exactly
    the behaviour the splitter in ``EmbeddingModel.create_embedding.batch`` must
    re-normalize back to ``[0, n)`` per caller.
    """

    @classmethod
    def check_lib(cls):
        return True

    @classmethod
    def match_json(cls, model_family, model_spec, quantization):
        return True

    def load(self):
        pass

    def _create_embedding(self, sentences, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        data = [
            EmbeddingData(index=i, object="embedding", embedding=[float(i)])
            for i in range(len(sentences))
        ]
        return Embedding(
            object="list",
            model=self._model_uid,
            model_replica=self._model_uid,
            data=data,
            usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),
        )


def _make_stub_model() -> _StubEmbeddingModel:
    return _StubEmbeddingModel("stub-uid", "/tmp/stub-embedding", _STUB_SPEC)


def test_create_embedding_batch_normalizes_index_per_request():
    """Regression for intermittent invalid embedding indices under concurrency.

    ``create_embedding.batch`` concatenates inputs from concurrent calls, runs
    ``_create_embedding`` once over the full list, then slices the result back
    per call. Each caller must receive indices covering ``[0, n)``; a call whose
    inputs land at a non-zero offset would otherwise leak the global batch index
    (e.g. ``[1, 2]`` instead of ``[0, 1]``) — the production defect captured in
    ``optimize/report``.
    """
    model = _make_stub_model()
    results = model.create_embedding.batch(
        _ExtensibleWrapper.delay(["a"]),  # call 0: 1 input, offset 0
        _ExtensibleWrapper.delay(["b", "c"]),  # call 1: 2 inputs, offset 1
    )

    assert [d["index"] for d in results[0]["data"]] == [0]
    # call 1's slice is data[1:3]; without re-normalization this is [1, 2]
    assert [d["index"] for d in results[1]["data"]] == [0, 1]
    # the vectors themselves must be unchanged — only the index label is fixed
    assert [d["embedding"] for d in results[1]["data"]] == [[1.0], [2.0]]


def test_create_embedding_batch_single_input_at_nonzero_offset():
    """A single-input call batched behind another request sits at offset > 0.

    Its response must still report ``index == [0]`` rather than the global
    ``[1]``. This guards against a fix that only handles multi-input calls.
    """
    model = _make_stub_model()
    results = model.create_embedding.batch(
        _ExtensibleWrapper.delay(["first"]),  # offset 0
        _ExtensibleWrapper.delay(["second"]),  # offset 1, single input
    )

    assert [d["index"] for d in results[0]["data"]] == [0]
    assert [d["index"] for d in results[1]["data"]] == [0]


def test_create_embedding_batch_treats_dict_as_one_multimodal_input():
    model = _make_stub_model()
    results = model.create_embedding.batch(
        _ExtensibleWrapper.delay({"image": "one.jpg", "text": "first"}),
        _ExtensibleWrapper.delay({"video": "two.mp4", "text": "second"}),
    )

    assert len(results) == 2
    assert [len(result["data"]) for result in results] == [1, 1]
    assert [result["data"][0]["index"] for result in results] == [0, 0]
