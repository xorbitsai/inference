import json

import pytest

from xinference.model.llm.config_parser import (
    _resolve_config_and_dir,
    build_llm_registration_from_local_config,
)


def _write_config(model_dir, config):
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config))


@pytest.mark.parametrize("ref_name", ["main", "master"])
def test_auto_register_huggingface_cache_root(tmp_path, ref_name):
    cache_root = tmp_path / "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B"
    revision = "916b56a44061fd5cd7d6a8fb632557ed4f724f60"
    snapshot_dir = cache_root / "snapshots" / revision
    _write_config(
        snapshot_dir,
        {
            "architectures": ["Qwen2ForCausalLM"],
            "text_config": {
                "hidden_size": 3584,
                "model_size_in_billions": 7,
                "num_hidden_layers": 28,
                "max_position_embeddings": 131072,
            },
        },
    )
    (cache_root / "refs").mkdir()
    (cache_root / "refs" / ref_name).write_text(revision)

    result = build_llm_registration_from_local_config(
        str(cache_root), "deepseek-r1-distill-qwen"
    )

    assert result["context_length"] == 131072
    assert result["model_specs"] == [
        {
            "model_uri": str(snapshot_dir),
            "model_format": "pytorch",
            "model_size_in_billions": "7",
            "quantization": "none",
        }
    ]


def test_auto_register_uses_config_before_architecture_estimate(tmp_path):
    model_dir = tmp_path / "custom-qwen"
    _write_config(
        model_dir,
        {
            "max_position_embeddings": 4096,
            "text_config": {
                "hidden_size": 2048,
                "model_size_in_billions": 0.8,
                "num_hidden_layers": 24,
                "max_position_embeddings": 262144,
            },
        },
    )

    result = build_llm_registration_from_local_config(str(model_dir), "qwen3.5")

    assert result["context_length"] == 4096
    assert result["model_specs"][0]["model_size_in_billions"] == "0_8"


@pytest.mark.parametrize(
    "model_dir_name",
    [
        "ERNIE-4.5-300B-47B-PT-4bit",
        "TinyLlama-1.1B-step-50K-105b-GGUF",
    ],
)
def test_auto_register_does_not_infer_size_from_path(tmp_path, model_dir_name):
    model_dir = tmp_path / model_dir_name
    _write_config(
        model_dir,
        {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_parameters_in_billions": 6.6,
        },
    )

    result = build_llm_registration_from_local_config(str(model_dir), "custom")

    assert result["model_specs"][0]["model_size_in_billions"] == "6_6"


def test_huggingface_cache_root_requires_unambiguous_snapshot(tmp_path):
    cache_root = tmp_path / "models--org--model-7B"
    _write_config(cache_root / "snapshots" / "revision-a", {})
    _write_config(cache_root / "snapshots" / "revision-b", {})

    with pytest.raises(ValueError, match="Multiple Hugging Face snapshots"):
        _resolve_config_and_dir(str(cache_root))
