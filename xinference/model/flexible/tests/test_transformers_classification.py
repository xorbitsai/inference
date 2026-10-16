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

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("task", ["text-classification", "zero-shot-classification"])
@pytest.mark.parametrize("device", [None, "cpu", 0])
@pytest.mark.parametrize("enable_virtual_env", [None, False, True])
def test_pipeline_device(monkeypatch, task, device, enable_virtual_env):
    from ..core import FlexibleModelSpec
    from ..utils import get_launcher

    launcher = get_launcher("xinference.model.flexible.launchers.transformers")
    pipeline = Mock()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=pipeline))
    config = {"task": task, "enable_virtual_env": enable_virtual_env}
    if device is not None:
        config["device"] = device
    model = launcher(
        "classifier",
        FlexibleModelSpec(
            model_name="classifier",
            model_uri="/local/model",
            launcher="xinference.model.flexible.launchers.transformers",
        ),
        **config,
    )
    model.load()
    pipeline.assert_called_once_with(model="/local/model", device=device, task=task)
    assert model.config == config
    model.infer(["first", "second"], top_k=None, truncation=True)
    pipeline.return_value.assert_called_once_with(
        ["first", "second"], top_k=None, truncation=True
    )


@pytest.mark.parametrize(
    "problem_type", ["single_label_classification", "multi_label_classification"]
)
def test_local_bert_classification(tmp_path, problem_type):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        BertTokenizerFast,
    )

    from ..core import FlexibleModelSpec, create_flexible_model_instance
    from ..custom import register_flexible_model, unregister_flexible_model

    # A complete local model exercises loading without downloading any weights.
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nbad\n", encoding="utf-8"
    )
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
    tokenizer.save_pretrained(tmp_path)
    config = BertConfig(
        vocab_size=7,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=16,
        num_labels=3,
        id2label={0: "first", 1: "second", 2: "third"},
        label2id={"first": 0, "second": 1, "third": 2},
        problem_type=problem_type,
    )
    reference = BertForSequenceClassification(config).eval()
    with torch.no_grad():
        reference.classifier.weight.zero_()
        reference.classifier.bias.copy_(torch.tensor([-1.0, 0.0, 1.0]))
    reference.save_pretrained(tmp_path)
    spec = FlexibleModelSpec(
        model_name="test-local-bert-classifier",
        model_uri=str(tmp_path),
        launcher="xinference.model.flexible.launchers.transformers",
        launcher_args=json.dumps({"task": "text-classification"}),
    )
    register_flexible_model(spec, persist=False)
    try:
        model = create_flexible_model_instance(
            "classifier", spec.model_name, device="cpu", enable_virtual_env=False
        )
        model.load()
        texts = ["good", "bad good", "good " * 30]
        with torch.inference_mode():
            logits = reference(
                **tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=16,
                    return_tensors="pt",
                )
            ).logits
        expected = (
            logits.sigmoid()
            if problem_type == "multi_label_classification"
            else logits.softmax(dim=-1)
        )
        results = model.infer(
            texts, top_k=None, batch_size=2, truncation=True, max_length=16
        )
        assert len(results) == len(texts)
        for row, scores in zip(results, expected.tolist()):
            assert {item["label"]: item["score"] for item in row} == pytest.approx(
                dict(zip(["first", "second", "third"], scores))
            )
        single = model.infer("good", top_k=None)
        assert single == results[0]
        assert model.infer("good")[0]["label"] == "third"
    finally:
        unregister_flexible_model(spec.model_name)
