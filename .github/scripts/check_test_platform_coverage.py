"""Validate CI platform coverage for changed tests without importing Xinference."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import yaml


def workflow_test_paths(root: Path) -> List[str]:
    workflow = yaml.safe_load(
        (root / ".github/workflows/python.yaml").read_text(encoding="utf-8")
    )
    step = next(
        step
        for step in workflow["jobs"]["build_test_job"]["steps"]
        if step.get("name") == "Test with pytest"
    )
    return (
        step["env"]["PLATFORM_TEST_PATHS"].split()
        + step["env"]["LINUX_TEST_PATHS"].split()
    )


def classified_test(path: str, content: str, targets: Sequence[str]) -> bool:
    if any(path == target or path.startswith(target + "/") for target in targets):
        return True
    if path.startswith("xinference/model/llm/mlx/tests/"):
        return True  # Discovered by the macOS unit/Metal jobs.
    return re.search(r"^# CI: linux$", content, re.MULTILINE) is not None


def git_paths(root: Path, *args: str) -> List[str]:
    output = subprocess.check_output(["git", *args, "-z"], cwd=root, text=True)
    return [path for path in output.split("\0") if path]


def changed_paths(
    root: Path,
    filenames: Sequence[str] = (),
    base: Optional[str] = None,
    head: str = "HEAD",
) -> List[str]:
    diff_args = ("--no-renames", "--diff-filter=AM", "--name-only")
    if base:
        return git_paths(root, "diff", *diff_args, f"{base}...{head}")

    paths = set()
    # Remote refs are read locally. No network or model dependencies are needed.
    for ref in ("upstream/main", "origin/main"):
        if (
            subprocess.run(
                ["git", "rev-parse", "--verify", "--quiet", ref],
                cwd=root,
                stdout=subprocess.DEVNULL,
            ).returncode
            == 0
        ):
            paths.update(git_paths(root, "diff", *diff_args, f"{ref}...HEAD"))
            break
    paths.update(git_paths(root, "diff", "--cached", *diff_args))
    paths.update(git_paths(root, "diff", *diff_args))
    if filenames:
        paths.update(git_paths(root, "ls-files", "--others", "--exclude-standard"))
        paths.intersection_update(filenames)
    return sorted(paths)


def coverage_errors(root: Path, paths: Iterable[str]) -> List[str]:
    targets = workflow_test_paths(root)
    errors = []
    for name in sorted(set(paths)):
        path = Path(name)
        if (
            not name.startswith("xinference/")
            or name.startswith("xinference/thirdparty/")
            or not path.name.startswith("test_")
            or path.suffix != ".py"
            or not (root / path).is_file()
        ):
            continue
        if not classified_test(
            name, (root / path).read_text(encoding="utf-8"), targets
        ):
            errors.append(
                f"{name}: add it to PLATFORM_TEST_PATHS, LINUX_TEST_PATHS "
                "(or the macOS scope), or declare '# CI: linux' for a "
                "portable/Linux-only regression."
            )
    return errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="*")
    parser.add_argument(
        "--base",
        default=os.environ.get("TEST_BASE_SHA")
        or os.environ.get("PRE_COMMIT_FROM_REF"),
    )
    parser.add_argument(
        "--head",
        default=os.environ.get("TEST_HEAD_SHA")
        or os.environ.get("PRE_COMMIT_TO_REF")
        or "HEAD",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    try:
        base = args.base if args.base and args.base.strip("0") else None
        paths = changed_paths(root, args.filenames, base, args.head)
        errors = coverage_errors(root, paths)
    except (
        OSError,
        subprocess.CalledProcessError,
        KeyError,
        StopIteration,
        yaml.YAMLError,
    ) as exc:
        print(f"Cannot validate test platform coverage: {exc}", file=sys.stderr)
        return 1
    for error in errors:
        print(error, file=sys.stderr)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
