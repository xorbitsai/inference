#!/usr/bin/env python3
"""Regroup GitHub generated release notes by conventional-commit type.

Faithful port of ``refactor_release_notes`` from the internal xprobebot
(xorbitsai/bot, ``bot/github/release.py``). Stdlib only.

Usage:
    format_release_notes.py --repo-name inference --date 2026-11-01 [notes.md]

The raw notes are read from the given file, or from stdin when omitted or
``-``. The formatted body is written to stdout.
"""

import argparse
import re
import sys
import warnings
from collections import defaultdict

# Support both uppercase (FEAT) and lowercase (feat) conventional commit types
LABELS_UPPERCASE = [
    "FEAT",
    "FEATURE",
    "ENH",
    "BUG",
    "DOC",
    "TST",
    "BLD",
    "PERF",
    "TYP",
    "CLN",
    "REF",
    "CHORE",
]

# Include both uppercase and lowercase versions
LABELS = LABELS_UPPERCASE + [x.lower() for x in LABELS_UPPERCASE] + ["fix"]


def refactor_release_notes(original_md: str, repo_name: str, date: str) -> str:
    md = ""
    label_to_commits = defaultdict(list)
    all_lines = original_md.strip().splitlines()
    version = all_lines[-1][all_lines[-1].rfind("v") + 1 :]
    changes_end_line = 1
    # find all commits
    for line in all_lines[1:]:
        if not line.startswith("* "):
            break
        changes_end_line += 1
        # Parse conventional commit format
        # Supports: "FEAT:", "feat:", "FEAT(scope):", "feat(scope):", etc.
        commit_content = line[2:]  # Remove "* " prefix
        colon_position = commit_content.find(":")

        if colon_position < 0:
            warnings.warn(f"Commit is not labelled: {line}")
            label_to_commits["OTHERS"].append(line)
            continue

        # Extract label (before colon)
        label_part = commit_content[:colon_position].strip()

        # Match "type" or "type(scope)" and keep just the type
        conventional_match = re.match(r"^([a-zA-Z]+)(?:\([^)]*\))?$", label_part)

        if conventional_match:
            label = conventional_match.group(1).upper()
            if label == "FIX":
                label = "BUG"
        else:
            label = label_part.upper()

        if label not in LABELS_UPPERCASE:
            label_to_commits["OTHERS"].append(line)
        else:
            label_to_commits[label].append(line)
    # title
    md += f"# What's new in {version} ({date})\n"
    # first line
    md += f"These are the changes in {repo_name} v{version}.\n"
    # features
    features = label_to_commits["FEAT"] + label_to_commits["FEATURE"]
    if features:
        md += "## New features\n"
        for commit in features:
            md += f"{commit}\n"
    # enhancements
    enhancements = (
        label_to_commits["ENH"]
        + label_to_commits["BLD"]
        + label_to_commits["PERF"]
        + label_to_commits["TYP"]
        + label_to_commits["CLN"]
        + label_to_commits["REF"]
    )
    if enhancements:
        md += "## Enhancements\n"
        for commit in enhancements:
            md += f"{commit}\n"
    # bug fixes
    if label_to_commits["BUG"]:
        md += "## Bug fixes\n"
        for commit in label_to_commits["BUG"]:
            md += f"{commit}\n"
    # tests
    if label_to_commits["TST"]:
        md += "## Tests\n"
        for commit in label_to_commits["TST"]:
            md += f"{commit}\n"
    # documentation
    if label_to_commits["DOC"]:
        md += "## Documentation\n"
        for commit in label_to_commits["DOC"]:
            md += f"{commit}\n"
    # others
    others = label_to_commits["OTHERS"] + label_to_commits["CHORE"]
    if others:
        md += "## Others\n"
        for commit in others:
            md += f"{commit}\n"
    for line in all_lines[changes_end_line:]:
        md += line + "\n"
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-name", required=True, help="e.g. inference")
    parser.add_argument("--date", required=True, help="release date, YYYY-MM-DD")
    parser.add_argument("input", nargs="?", default="-", help="notes file or '-'")
    args = parser.parse_args()
    if args.input == "-":
        original = sys.stdin.read()
    else:
        with open(args.input, encoding="utf-8") as f:
            original = f.read()
    if not original.strip():
        sys.exit("error: empty release notes")
    sys.stdout.write(refactor_release_notes(original, args.repo_name, args.date))


if __name__ == "__main__":
    main()
