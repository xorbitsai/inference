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

from ....utils import ModelArtifactSource
from .. import core


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_prepare_modelscope_jina_clip_v2(monkeypatch, tmp_path):
    model_path = tmp_path / "jina-clip-v2"
    clip_impl_path = tmp_path / "jina-clip-implementation"
    text_model_path = tmp_path / "jina-embeddings-v3"
    text_impl_path = tmp_path / "xlm-roberta-flash-implementation"
    for directory in (
        model_path,
        clip_impl_path,
        text_model_path,
        text_impl_path,
    ):
        directory.mkdir()

    _write_json(model_path / "config.json", {"text_config": {}})
    _write_json(model_path / "preprocessor_config.json", {})
    (model_path / "custom_st.py").write_text(
        "from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer\n"
        "        self.tokenizer = AutoTokenizer.from_pretrained(\n"
        "            tokenizer_name_or_path or model_name_or_path,\n"
        "            **tokenizer_kwargs,\n"
        "        )\n",
        encoding="utf-8",
    )
    (clip_impl_path / "modeling_clip.py").write_text(
        "from transformers import (\n"
        "    AutoTokenizer,\n"
        ")\n"
        "            self.tokenizer = AutoTokenizer.from_pretrained(\n"
        "                self.config._name_or_path, trust_remote_code=True\n"
        "            )\n",
        encoding="utf-8",
    )
    (clip_impl_path / "processing_clip.py").write_text(
        "PROCESSOR = True\n", encoding="utf-8"
    )
    _write_json(text_model_path / "config.json", {})
    (text_impl_path / "modeling_xlm_roberta.py").write_text(
        "        self.tokenizer = AutoTokenizer.from_pretrained(\n"
        "            self.name_or_path, trust_remote_code=True\n"
        "        )\n",
        encoding="utf-8",
    )
    (text_impl_path / "modeling_lora.py").write_text("MODEL = True\n", encoding="utf-8")

    snapshots = {
        "jinaai/jina-clip-implementation": clip_impl_path,
        "jinaai/jina-embeddings-v3": text_model_path,
        "jinaai/xlm-roberta-flash-implementation": text_impl_path,
    }
    calls = []

    def snapshot_download(self, model_id, **kwargs):
        calls.append((self.hub, model_id, kwargs))
        return str(snapshots[model_id])

    cached = []
    monkeypatch.setattr(ModelArtifactSource, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        core,
        "_copy_python_sources_to_transformers_cache",
        lambda source_dir: cached.append(source_dir),
    )

    source = ModelArtifactSource("modelscope")
    core._prepare_modelscope_jina_clip_v2(str(model_path), source)
    core._prepare_modelscope_jina_clip_v2(str(model_path), source)

    clip_config = json.loads((model_path / "config.json").read_text())
    processor_config = json.loads((model_path / "preprocessor_config.json").read_text())
    text_config = json.loads((text_model_path / "config.json").read_text())
    assert clip_config["auto_map"] == {
        "AutoConfig": "configuration_clip.JinaCLIPConfig",
        "AutoModel": "modeling_clip.JinaCLIPModel",
    }
    assert clip_config["text_config"]["hf_model_name_or_path"] == str(text_model_path)
    assert processor_config["auto_map"]["AutoProcessor"].startswith("processing_clip.")
    assert text_config["_name_or_path"] == str(text_model_path)
    assert text_config["auto_map"]["AutoModel"] == "modeling_lora.XLMRobertaLoRA"
    assert "XLMRobertaTokenizerFast" in (model_path / "custom_st.py").read_text()
    assert "XLMRobertaTokenizerFast" in (model_path / "modeling_clip.py").read_text()
    assert (
        "self.tokenizer = None"
        in (text_model_path / "modeling_xlm_roberta.py").read_text()
    )
    assert cached == [str(text_model_path), str(text_model_path)]
    assert [model_id for _, model_id, _ in calls[:3]] == list(snapshots)
    assert all(hub == "modelscope" for hub, _, _ in calls)
