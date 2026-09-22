"""Catalog authoring checks run without importing model engines or Xinference."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

CATALOG_PATH = Path(__file__).parents[1] / "_model_catalog.py"
MODEL_ROOT = CATALOG_PATH.parent / "model"
_spec = importlib.util.spec_from_file_location("catalog_schema_checks", CATALOG_PATH)
assert _spec is not None and _spec.loader is not None
catalog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(catalog)


def model_record(kind):
    source = {"model_id": "org/demo", "model_revision": "main"}
    record = {"version": 2, "model_name": "demo"}
    if kind in ("llm", "embedding", "rerank"):
        source["quantizations"] = ["none"]
        spec = {"model_format": "pytorch", "model_src": {"huggingface": source}}
        record["model_specs"] = [spec]
        if kind == "llm":
            record.update(
                context_length=4096, model_lang=["en"], model_ability=["chat"]
            )
            spec["model_size_in_billions"] = "1_8"
        else:
            record.update(language=["en"], max_tokens=512)
            if kind == "embedding":
                record["dimensions"] = 768
    else:
        record.update(model_family="demo", model_src={"huggingface": source})
        abilities = {
            "audio": "audio2text",
            "image": "text2image",
            "video": "text2video",
            "world": "image2world",
        }
        record["model_ability"] = [abilities[kind]]
        if kind == "audio":
            record["multilingual"] = True
        elif kind == "video":
            record.update(engine="diffusers", model_format="diffusers")
        elif kind == "world":
            record.update(
                version=1,
                model_format="pytorch",
                source_url="https://example.org/demo.git",
                source_revision="main",
            )
    return record


def validate(tmp_path, kind, records):
    directory = tmp_path / "models"
    directory.mkdir(exist_ok=True)
    (directory / "demo.json").write_text(json.dumps(records), encoding="utf-8")
    catalog.validate_model_catalog(directory, kind)


@pytest.mark.parametrize("kind", catalog.MODEL_KINDS)
def test_all_builtin_catalogs_match_schema(kind):
    catalog.validate_model_catalog(MODEL_ROOT / kind / "models", kind)


@pytest.mark.parametrize("source_level", [False, True])
@pytest.mark.parametrize(
    "invalid", [{"head_dim": 0}, {"num_key_value_heads": "8"}, {"unknown": 1}]
)
def test_model_metadata_schema(tmp_path, source_level, invalid):
    record = model_record("llm")
    spec = record["model_specs"][0]
    owner = spec["model_src"]["huggingface"] if source_level else spec
    metadata = dict(
        vocab_size=32000,
        num_attention_heads=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
    )
    owner["model_metadata"] = metadata
    validate(tmp_path, "llm", [record])
    metadata.update(invalid)
    with pytest.raises(ValueError, match="model_metadata"):
        validate(tmp_path, "llm", [record])


@pytest.mark.parametrize("path", sorted((MODEL_ROOT / "schemas").glob("*.json")))
def test_schema_definitions_are_valid(path):
    Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("kind", catalog.MODEL_KINDS)
def test_rejects_unknown_record_fields(tmp_path, kind):
    record = model_record(kind)
    validate(tmp_path, kind, [record])
    record["model_nmae"] = "typo"
    with pytest.raises(ValueError, match=r"demo.json#/0:.*model_nmae"):
        validate(tmp_path, kind, [record])


@pytest.mark.parametrize("kind", catalog.MODEL_KINDS)
@pytest.mark.parametrize("field,value", [("version", "2"), ("version", 9)])
def test_rejects_wrong_version_without_coercion(tmp_path, kind, field, value):
    record = model_record(kind)
    record[field] = value
    with pytest.raises(ValueError, match="version"):
        validate(tmp_path, kind, [record])


@pytest.mark.parametrize("kind", catalog.MODEL_KINDS)
def test_rejects_missing_source_id_and_unknown_hubs(tmp_path, kind):
    record = model_record(kind)
    owner = record["model_specs"][0] if "model_specs" in record else record
    source = owner["model_src"]["huggingface"]
    del source["model_id"]
    with pytest.raises(ValueError, match="model_id"):
        validate(tmp_path, kind, [record])
    source["model_id"] = "org/demo"
    owner["model_src"] = {"hugingface": source}
    with pytest.raises(ValueError, match="hugingface"):
        validate(tmp_path, kind, [record])


@pytest.mark.parametrize("kind", catalog.MODEL_KINDS)
def test_rejects_nested_source_typos(tmp_path, kind):
    record = model_record(kind)
    owner = record["model_specs"][0] if "model_specs" in record else record
    owner["model_src"]["huggingface"]["model_revison"] = "main"
    with pytest.raises(ValueError, match=r"model_src/huggingface:.*model_revison"):
        validate(tmp_path, kind, [record])


@pytest.mark.parametrize("kind", ["llm", "embedding", "rerank"])
@pytest.mark.parametrize("value", [[], ["none", "none"], [4], "none", None])
def test_quantizations_are_nonempty_unique_string_arrays(tmp_path, kind, value):
    record = model_record(kind)
    record["model_specs"][0]["model_src"]["huggingface"]["quantizations"] = value
    with pytest.raises(ValueError, match="quantizations"):
        validate(tmp_path, kind, [record])


@pytest.mark.parametrize("kind", ["llm", "embedding", "rerank"])
def test_gguf_requires_file_template_for_every_hub(tmp_path, kind):
    record = model_record(kind)
    spec = record["model_specs"][0]
    spec["model_format"] = "ggufv2"
    source = spec["model_src"]["huggingface"]
    spec["model_src"]["modelscope"] = copy.deepcopy(source)
    source["model_file_name_template"] = "demo-{quantization}.gguf"
    with pytest.raises(ValueError, match=r"modelscope:.*model_file_name_template"):
        validate(tmp_path, kind, [record])
    spec["model_src"]["modelscope"]["model_file_name_template"] = "demo.gguf"
    validate(tmp_path, kind, [record])


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_specs", []),
        ("model_ability", ["cht"]),
        ("context_length", 0),
        ("context_length", True),
        ("virtualenv", {"packages": [], "package_typo": []}),
        ("cache_config", {"ignore_paterns": []}),
    ],
)
def test_rejects_invalid_llm_metadata(tmp_path, field, value):
    record = model_record("llm")
    record[field] = value
    with pytest.raises(ValueError, match=field):
        validate(tmp_path, "llm", [record])


def test_rejects_missing_required_fields_and_spec_typos(tmp_path):
    record = model_record("llm")
    del record["context_length"]
    with pytest.raises(ValueError, match="context_length"):
        validate(tmp_path, "llm", [record])
    record["context_length"] = 4096
    record["model_specs"][0]["model_szie_in_billions"] = 7
    with pytest.raises(ValueError, match="model_szie_in_billions"):
        validate(tmp_path, "llm", [record])


@pytest.mark.parametrize("kind", ["audio", "image", "video", "world"])
def test_engine_parameter_dictionaries_remain_extensible(tmp_path, kind):
    record = model_record(kind)
    record["default_model_config"] = {"future_engine_option": {"value": [1, True]}}
    validate(tmp_path, kind, [record])
    record["default_model_config"] = []
    with pytest.raises(ValueError, match="default_model_config"):
        validate(tmp_path, kind, [record])


def test_multiple_audio_engine_variants_in_one_file(tmp_path):
    first = model_record("audio")
    first.update(engine="PyTorch", model_format="pytorch")
    second = copy.deepcopy(first)
    second.update(engine="MLX", model_format="mlx", quantization="4bit")
    validate(tmp_path, "audio", [first, second])


@pytest.mark.parametrize("kind", ["image", "world"])
def test_direct_and_grouped_sources_are_mutually_exclusive(tmp_path, kind):
    record = model_record(kind)
    source = record.pop("model_src")
    variant = {"model_src": source}
    if kind == "world":
        record["model_family"] = record.pop("model_name")
        variant["model_name"] = "demo-1B"
    else:
        variant["model_format"] = "pytorch"
    record["model_specs"] = [variant]
    validate(tmp_path, kind, [record])
    record["model_src"] = source
    with pytest.raises(ValueError):
        validate(tmp_path, kind, [record])
    record["model_name"] = "demo"
    with pytest.raises(ValueError):
        validate(tmp_path, kind, [record])


def test_empty_catalog_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Empty model catalog"):
        validate(tmp_path, "llm", [])


def test_empty_gguf_template_requires_split_metadata(tmp_path):
    record = model_record("llm")
    spec = record["model_specs"][0]
    spec["model_format"] = "ggufv2"
    source = spec["model_src"]["huggingface"]
    source["model_file_name_template"] = ""
    with pytest.raises(ValueError, match="model_file_name_split_template"):
        validate(tmp_path, "llm", [record])
    source["model_file_name_split_template"] = "demo-{quantization}-{part}.gguf"
    source["quantization_parts"] = {"Q4_K_M": ["00001-of-00002", "00002-of-00002"]}
    validate(tmp_path, "llm", [record])
    source["quantization_parts"] = {}
    with pytest.raises(ValueError, match="quantization_parts"):
        validate(tmp_path, "llm", [record])


def test_optional_gguf_download_configuration_is_complete(tmp_path):
    record = model_record("image")
    source = record["model_src"]["huggingface"]
    source["gguf_model_id"] = "org/demo-gguf"
    with pytest.raises(ValueError, match="gguf_quantizations"):
        validate(tmp_path, "image", [record])
    source.update(
        gguf_quantizations=["Q4_K_M"],
        gguf_model_file_name_template="demo-{quantization}.gguf",
    )
    validate(tmp_path, "image", [record])
    source["gguf_quantizations"].append("Q4_K_M")
    with pytest.raises(ValueError, match="non-unique"):
        validate(tmp_path, "image", [record])


def test_catalog_structure_checks_still_apply(tmp_path):
    (tmp_path / "wrong.json").write_text(json.dumps([model_record("llm")]))
    with pytest.raises(ValueError, match="Model name does not match"):
        catalog.validate_model_catalog(tmp_path, "llm")


def test_schema_validation_is_opt_in_for_legacy_loading(tmp_path):
    path = tmp_path / "legacy.json"
    records = [{"model_name": "demo", "legacy_extension": True}]
    path.write_text(json.dumps(records))
    assert catalog.load_model_catalog(path) == records
    result = subprocess.run(
        [sys.executable, "-S", str(CATALOG_PATH), "validate", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(CATALOG_PATH),
            "validate",
            str(path),
            "--model-type",
            "llm",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "pip install" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_all_catalogs_runs_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, str(CATALOG_PATH), "validate", "--all"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cli_schema_failure_reports_file_and_pointer(tmp_path):
    record = model_record("embedding")
    record["dimensions"] = "768"
    path = tmp_path / "demo.json"
    path.write_text(json.dumps([record]))
    result = subprocess.run(
        [
            sys.executable,
            str(CATALOG_PATH),
            "validate",
            str(path),
            "--model-type",
            "embedding",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert f"{path}#/0/dimensions" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "args",
    [
        ["validate"],
        ["validate", "--all", "extra"],
        ["validate", "--all", "--model-type", "llm"],
        ["export", "--all"],
        ["validate", "source", "destination"],
    ],
)
def test_cli_rejects_ambiguous_arguments(args):
    result = subprocess.run(
        [sys.executable, str(CATALOG_PATH), *args], capture_output=True, text=True
    )
    assert result.returncode == 2
