#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path


MODEL_SPEC_FILES = (
    "xinference/model/llm/llm_family.json",
    "xinference/model/embedding/model_spec.json",
    "xinference/model/rerank/model_spec.json",
    "xinference/model/image/model_spec.json",
    "xinference/model/audio/model_spec.json",
    "xinference/model/video/model_spec.json",
)

SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


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

    unexpected = sorted(set(changed) - set(MODEL_SPEC_FILES))
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
    for spec_file in MODEL_SPEC_FILES:
        path = workspace / spec_file
        with path.open() as fp:
            _visit_model_names(json.load(fp), spec_file, unsafe)

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
    args = parser.parse_args()

    if args.command == "changed-files":
        return validate_changed_files()
    if args.command == "model-names":
        return validate_model_names(args.workspace)
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
