import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from xinference._model_catalog import load_model_catalog, split_model_catalog

MODEL_ROOT = Path(__file__).parents[1] / "model"
KINDS = ("llm", "embedding", "rerank", "image", "audio", "video", "world")


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def catalog(tmp_path):
    source = tmp_path / "audio_models.json"
    records = [
        {"model_name": "voice", "engine": "PyTorch", "updated_at": 1},
        {"model_name": "other", "engine": "PyTorch"},
        {"model_name": "voice", "engine": "MLX", "updated_at": 2},
    ]
    write_json(source, records)
    destination = tmp_path / "models"
    split_model_catalog(source, destination)
    return source, destination, records


def test_roundtrip_preserves_interleaved_engines_and_fresh_records(catalog):
    source, destination, records = catalog
    assert sorted(p.name for p in destination.iterdir()) == [
        "index.json",
        "other.json",
        "voice.json",
    ]
    assert load_model_catalog(source) == load_model_catalog(destination) == records
    changed = load_model_catalog(destination)
    changed[0]["engine"] = "changed"
    assert load_model_catalog(destination) == records


@pytest.mark.parametrize("kind", KINDS)
def test_all_builtin_catalogs_roundtrip(kind, tmp_path):
    directory = MODEL_ROOT / kind / "models"
    records = load_model_catalog(directory)
    assert records
    aggregate = tmp_path / f"{kind}_models.json"
    write_json(aggregate, records)
    split_model_catalog(aggregate, tmp_path / "models")
    assert load_model_catalog(tmp_path / "models") == records
    assert (tmp_path / "models" / "index.json").read_bytes() == (
        directory / "index.json"
    ).read_bytes()


@pytest.mark.parametrize(
    "reference",
    [
        ["../voice.json", 0],
        ["/voice.json", 0],
        ["voice.json", -1],
        ["voice.json", True],
        ["voice.json", 99],
        ["missing.json", 0],
        ["index.json", 0],
        "voice.json",
        ["voice.json"],
    ],
)
def test_invalid_reference_is_rejected(catalog, reference):
    _, directory, _ = catalog
    write_json(directory / "index.json", [reference])
    with pytest.raises(ValueError, match="models"):
        load_model_catalog(directory)


def test_duplicate_reference_is_rejected(catalog):
    _, directory, _ = catalog
    write_json(directory / "index.json", [["voice.json", 0], ["voice.json", 0]])
    with pytest.raises(ValueError, match="Duplicate"):
        load_model_catalog(directory)


@pytest.mark.parametrize("missing_variant", [True, False])
def test_unindexed_file_or_variant_is_rejected(catalog, missing_variant):
    _, directory, _ = catalog
    if missing_variant:
        index = json.loads((directory / "index.json").read_text())
        write_json(directory / "index.json", index[:-1])
    else:
        write_json(directory / "extra.json", [{"model_name": "extra"}])
    with pytest.raises(ValueError, match="Unindexed"):
        load_model_catalog(directory)


def test_model_file_cannot_mix_names(catalog):
    _, directory, _ = catalog
    write_json(directory / "voice.json", [{"model_name": "wrong"}])
    with pytest.raises(ValueError, match="voice.json"):
        load_model_catalog(directory)


def test_invalid_json_reports_filename(catalog):
    _, directory, _ = catalog
    (directory / "voice.json").write_text("{broken")
    with pytest.raises(ValueError, match="voice.json"):
        load_model_catalog(directory)


def test_split_refuses_to_overwrite(catalog):
    source, directory, records = catalog
    with pytest.raises(FileExistsError):
        split_model_catalog(source, directory)
    assert load_model_catalog(directory) == records


@pytest.mark.parametrize("names", [["Index"], ["voice", "Voice"], ["../voice"]])
def test_split_rejects_unsafe_or_colliding_names_before_writing(tmp_path, names):
    source = tmp_path / "models.json"
    write_json(source, [{"model_name": name} for name in names])
    destination = tmp_path / "models"
    with pytest.raises(ValueError):
        split_model_catalog(source, destination)
    assert not destination.exists()


def test_world_family_without_top_level_name(tmp_path):
    source = tmp_path / "world_models.json"
    records = [{"model_family": "world", "model_specs": [{"model_name": "world-1B"}]}]
    write_json(source, records)
    split_model_catalog(source, tmp_path / "models")
    assert load_model_catalog(tmp_path / "models") == records


def test_export_works_without_runtime_dependencies(catalog, tmp_path):
    _, directory, records = catalog
    helper = MODEL_ROOT.parent / "_model_catalog.py"
    output = tmp_path / "export.json"
    subprocess.run(
        [sys.executable, "-S", str(helper), "export", str(directory), str(output)],
        check=True,
    )
    assert json.loads(output.read_text()) == records


@pytest.fixture
def workflow():
    path = MODEL_ROOT.parents[1] / ".github/scripts/validate_gen_docs_pr.py"
    spec = importlib.util.spec_from_file_location("catalog_workflow", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "path,allowed",
    [
        ("xinference/model/audio/models/voice.json", True),
        ("xinference/model/world/models/index.json", True),
        ("xinference/model/audio/models/run.py", False),
        ("xinference/model/audio/models/sub/voice.json", False),
        ("xinference/model/audio/models/../voice.json", False),
        ("xinference/model/audio/model_spec.json", False),
    ],
)
def test_workflow_only_accepts_catalog_data(workflow, path, allowed):
    assert workflow.is_model_spec_file(path) is allowed


def test_workflow_copy_includes_deletions_and_validates(workflow, tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    for relative in workflow.MODEL_SPEC_DIRS:
        directory = source / relative
        directory.mkdir(parents=True)
        write_json(directory / "index.json", [])
        old = target / relative
        old.mkdir(parents=True)
        write_json(old / "stale.json", [])
    workflow.copy_catalogs(source, target)
    assert workflow.validate_model_names(target) == 0
    assert not list(target.rglob("stale.json"))


def test_workflow_rejects_symlinked_parent(workflow, tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    try:
        (source / "xinference").symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlink"):
        list(workflow.catalog_files(source))
