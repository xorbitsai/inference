"""Dependency-free readers and conversion tools for built-in model catalogs.

Run this file directly when preparing Models Hub exports, without importing
Xinference or installing any model engines.
"""

import sys

if __name__ == "__main__":
    # Direct execution must not let xinference/types.py shadow stdlib types.
    sys.path.pop(0)

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Union

_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.json\Z")


def _read_json(path: Path):
    if path.is_symlink():
        raise ValueError(f"Catalog files must not be symlinks: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read model catalog {path}: {exc}") from exc


def _records(path: Path) -> List[Dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, list) or any(not isinstance(x, dict) for x in data):
        raise ValueError(f"{path} must contain a list of model records")
    return data


def _model_name(record: Dict[str, Any]) -> str:
    # World models may contain named variants under one family.
    name = record.get("model_name") or record.get("model_family")
    if not isinstance(name, str) or not _FILENAME.fullmatch(f"{name}.json"):
        raise ValueError(f"Invalid catalog model name: {name!r}")
    return name


def load_model_catalog(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a split directory or an unchanged Hub/custom aggregate JSON file.

    Discover model files in stable, case-insensitive filename order. Preserve
    record order within each file, where engine/spec preference can matter.
    Return fresh records: runtime loaders normalize/mutate their input.
    """
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Catalog paths must not be symlinks: {path}")
    if not path.is_dir():
        return _records(path)
    seen = set()
    result = []
    for file in sorted(path.glob("*.json"), key=lambda p: (p.name.casefold(), p.name)):
        if not _FILENAME.fullmatch(file.name):
            raise ValueError(f"Invalid catalog filename: {file}")
        if file.name.casefold() in seen:
            raise ValueError(f"Catalog filename collision: {file}")
        seen.add(file.name.casefold())
        records = _records(file)
        for record in records:
            if _model_name(record) + ".json" != file.name:
                raise ValueError(f"Model name does not match {file}")
        result.extend(records)
    return result


def split_model_catalog(source: Union[str, Path], destination: Union[str, Path]):
    """Convert a full Hub catalog to a new directory (never overwrite a source)."""
    records = load_model_catalog(source)
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    case_names: Dict[str, str] = {}
    for record in records:
        name = _model_name(record) + ".json"
        if case_names.setdefault(name.casefold(), name) != name:
            raise ValueError(f"Catalog filename collision: {name}")
        grouped[name].append(record)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name, rows in grouped.items():
        (destination / name).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("split", "export", "validate"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", nargs="?", type=Path)
    args = parser.parse_args()
    if args.command == "validate":
        load_model_catalog(args.source)
    elif args.destination is None:
        parser.error("split/export requires a destination")
    elif args.command == "split":
        split_model_catalog(args.source, args.destination)
    else:
        data = load_model_catalog(args.source)
        with args.destination.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
