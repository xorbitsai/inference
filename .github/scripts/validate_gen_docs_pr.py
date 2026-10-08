#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path

MODEL_SPEC_DIRS = tuple(
    f"xinference/model/{kind}/models"
    for kind in ("llm", "embedding", "rerank", "image", "audio", "video", "world")
)

SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def is_model_spec_file(filename):
    path = Path(filename)
    return (
        path.parent.as_posix() in MODEL_SPEC_DIRS
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.json", path.name) is not None
    )


def catalog_files(workspace):
    for directory in MODEL_SPEC_DIRS:
        root = workspace / directory
        for parent in (root, *root.parents):
            if parent.is_symlink():
                raise ValueError(f"Refusing symlinked catalog path: {parent}")
        if not root.is_dir():
            raise ValueError(f"Missing catalog directory: {root}")
        for path in sorted(root.iterdir()):
            relative = path.relative_to(workspace)
            if (
                path.is_symlink()
                or not path.is_file()
                or not is_model_spec_file(str(relative))
            ):
                raise ValueError(f"Unexpected catalog path: {path}")
            yield relative


def copy_catalogs(source, destination):
    files = list(catalog_files(source))  # Validate before touching the destination.
    for directory in MODEL_SPEC_DIRS:
        target = destination / directory
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
    for relative in files:
        shutil.copyfile(source / relative, destination / relative)


def _next_link(link_header):
    links = {}
    for part in link_header.split(","):
        if 'rel="' not in part:
            continue
        url_part, rel_part = part.split(";", 1)
        rel = rel_part.split('rel="', 1)[1].split('"', 1)[0]
        links[rel] = url_part.strip()[1:-1]
    return links.get("next")


def validate_changed_files():
    token = os.environ["GITHUB_TOKEN"]
    next_url = os.environ["PR_FILES_URL"] + "?per_page=100"
    changed = []

    while next_url:
        req = urllib.request.Request(
            next_url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(req) as resp:
            for item in json.load(resp):
                changed.append(item["filename"])
                if "previous_filename" in item:
                    changed.append(item["previous_filename"])
            next_url = _next_link(resp.headers.get("Link", ""))

    unexpected = sorted(
        filename for filename in set(changed) if not is_model_spec_file(filename)
    )
    if unexpected:
        print("Unexpected PR file changes are not allowed:")
        for filename in unexpected:
            print(f"- {filename}")
        return 1

    if not changed:
        print("No changed files found for PR.")
        return 1

    print("Changed files are limited to the allowed model spec JSON files.")
    return 0


def _visit_model_names(value, source, unsafe):
    if isinstance(value, dict):
        model_name = value.get("model_name")
        if model_name is not None:
            if not isinstance(model_name, str) or not SAFE_MODEL_NAME.fullmatch(
                model_name
            ):
                unsafe.append((source, model_name))
        for child in value.values():
            _visit_model_names(child, source, unsafe)
    elif isinstance(value, list):
        for child in value:
            _visit_model_names(child, source, unsafe)


def validate_model_names(workspace):
    unsafe = []
    for spec_file in catalog_files(workspace):
        path = workspace / spec_file
        with path.open() as fp:
            _visit_model_names(json.load(fp), spec_file, unsafe)

    # Load the validator from the trusted base checkout, never from PR code.
    helper = Path(__file__).resolve().parents[2] / "xinference" / "_model_catalog.py"
    spec = importlib.util.spec_from_file_location("_model_catalog", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for directory in MODEL_SPEC_DIRS:
        module.load_model_catalog(workspace / directory)

    if unsafe:
        print("Unsafe model_name values are not allowed:")
        for source, model_name in unsafe:
            print(f"- {source}: {model_name!r}")
        return 1

    print("Model names are safe for generated documentation paths.")
    return 0


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("changed-files")
    model_names = subparsers.add_parser("model-names")
    model_names.add_argument("workspace", type=Path)
    copy = subparsers.add_parser("copy-catalogs")
    copy.add_argument("source", type=Path)
    copy.add_argument("destination", type=Path)
    args = parser.parse_args()

    if args.command == "changed-files":
        return validate_changed_files()
    if args.command == "model-names":
        return validate_model_names(args.workspace)
    if args.command == "copy-catalogs":
        copy_catalogs(args.source, args.destination)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
