"""Select cross-platform CI coverage without dropping the Linux version matrix."""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

PYTHON_VERSIONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
PLATFORMS = ("macos-latest", "windows-latest")
INFRASTRUCTURE_FILES = {
    ".github/workflows/python.yaml",
    ".github/platform-smoke-tests.txt",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "MANIFEST.in",
    "build_backend.py",
    "build_web.py",
}


def infrastructure_changed(path: str) -> bool:
    return path in INFRASTRUCTURE_FILES or path.startswith(
        (".github/scripts/", ".github/requirements/")
    )


def platform_changed(path: str) -> bool:
    if path.startswith("xinference/ui/"):
        return False
    return path.startswith(("xinference/", "scripts/"))


def metal_changed(path: str) -> bool:
    if path.startswith("xinference/ui/"):
        return False
    if path.startswith("xinference/thirdparty/"):
        return True
    if not path.startswith("xinference/model/"):
        return path.startswith("xinference/")
    # MLX's P-D path and Metal tests use the shared SGLang Xavier settings
    # and directory, even though the rest of the SGLang engine is separate.
    if path.startswith("xinference/model/llm/sglang/xavier/"):
        return True
    if path.startswith("xinference/model/llm/"):
        # These engines own their implementations. Everything else under
        # llm/ includes MLX or shared code such as tool parsers and metadata.
        return not path.startswith(
            tuple(
                f"xinference/model/llm/{engine}/"
                for engine in (
                    "llama_cpp",
                    "lmdeploy",
                    "sglang",
                    "transformers",
                    "vllm",
                )
            )
        )
    if path.startswith(
        (
            "xinference/model/audio/",
            "xinference/model/scheduler/",
        )
    ):
        return True
    return Path(path).parent.as_posix() == "xinference/model"


def build_matrix(event_name: str, paths: Optional[List[str]]) -> Dict[str, list]:
    # Missing/unavailable diffs must never silently remove platform coverage.
    full = (
        event_name in ("schedule", "workflow_dispatch")
        or paths is None
        or any(infrastructure_changed(path) for path in paths)
    )
    paths = paths or []
    matrix = [
        {
            "os": "ubuntu-latest",
            "python-version": version,
            "module": "xinference",
            "suite": "full",
        }
        for version in PYTHON_VERSIONS
    ]
    if full or any(platform_changed(path) for path in paths):
        matrix.extend(
            {
                "os": platform,
                "python-version": version,
                "module": "xinference",
                "suite": "full" if full else "smoke",
            }
            for platform in PLATFORMS
            for version in ("3.10", "3.14")
        )
    if full or any(metal_changed(path) for path in paths):
        matrix.append(
            {
                "os": "macos-latest",
                "python-version": "3.10",
                "module": "metal",
                "suite": "metal",
            }
        )
    return {"include": matrix}


def changed_paths(event_name: str, event: dict) -> Optional[List[str]]:
    if event_name == "pull_request":
        base = event["pull_request"]["base"]["sha"]
        head = event["pull_request"]["head"]["sha"]
        diff_range = f"{base}...{head}"
    elif event_name == "push":
        base, head = event.get("before"), event.get("after")
        diff_range = f"{base}..{head}"
    else:
        return None
    if not base or not head or not base.strip("0"):
        return None
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "-z", diff_range, "--"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    return [path for path in result.stdout.split("\0") if path]


if __name__ == "__main__":
    event_name = os.environ["GITHUB_EVENT_NAME"]
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    matrix = json.dumps(build_matrix(event_name, changed_paths(event_name, event)))
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"matrix={matrix}\n")
    print(f"Platform test matrix: {matrix}")
