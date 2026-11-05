"""Contract tests for .github/scripts/format_release_notes.py.

The fixtures are the raw GitHub generated notes for v3.2.0...v3.3.0 and the
body that XprobeBot published for v3.3.0, so the release workflow keeps
producing the same output as the bot it replaces.

GitHub no longer lists one new-contributor line (@AmirF194) in freshly
generated notes because that user has contributed since; it was copied back
from the published body into the generated fixture. The formatter passes that
trailing section through unchanged.
"""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".github" / "scripts" / "format_release_notes.py"
FIXTURES = Path(__file__).parent / "resources" / "release_notes"

_spec = importlib.util.spec_from_file_location("format_release_notes", SCRIPT)
assert _spec and _spec.loader
fmt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fmt)


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


RAW = """## What's Changed
* BUG: upper bug by @a in https://x/pull/1
* fix(scope): lower fix by @a in https://x/pull/2
* feature: long feature by @a in https://x/pull/3
* perf: speed by @a in https://x/pull/4
* test: not a label by @a in https://x/pull/5
* chore(ci): chore by @a in https://x/pull/6
* no label here by @a in https://x/pull/7
* doc: docs by @a in https://x/pull/8
* tst: tests by @a in https://x/pull/9
* enh: enhance by @a in https://x/pull/10
* BLD: build by @a in https://x/pull/11
* typ: types by @a in https://x/pull/12
* cln: cleanup by @a in https://x/pull/13
* ref(core): refactor by @a in https://x/pull/14

**Full Changelog**: https://github.com/o/r/compare/v1.0.0...v1.0.1
"""


class TestFormatReleaseNotes(unittest.TestCase):
    def test_matches_published_v3_3_0_body(self):
        out = fmt.refactor_release_notes(
            _read("v3.3.0_generated.txt"), "inference", "2026-08-30"
        )
        self.assertEqual(out, _read("v3.3.0_published.txt"))

    def test_cli_reads_stdin_and_file(self):
        expected = _read("v3.3.0_published.txt")
        args = [
            sys.executable,
            str(SCRIPT),
            "--repo-name",
            "inference",
            "--date",
            "2026-08-30",
        ]
        from_file = subprocess.run(
            [*args, str(FIXTURES / "v3.3.0_generated.txt")],
            capture_output=True,
            text=True,
            check=True,
        )
        from_stdin = subprocess.run(
            args,
            input=_read("v3.3.0_generated.txt"),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(from_file.stdout, expected)
        self.assertEqual(from_stdin.stdout, expected)

    def test_grouping_and_order(self):
        out = fmt.refactor_release_notes(RAW, "r", "2026-01-02")
        self.assertEqual(
            out.splitlines(),
            [
                "# What's new in 1.0.1 (2026-01-02)",
                "These are the changes in r v1.0.1.",
                "## New features",
                "* feature: long feature by @a in https://x/pull/3",
                "## Enhancements",
                "* enh: enhance by @a in https://x/pull/10",
                "* BLD: build by @a in https://x/pull/11",
                "* perf: speed by @a in https://x/pull/4",
                "* typ: types by @a in https://x/pull/12",
                "* cln: cleanup by @a in https://x/pull/13",
                "* ref(core): refactor by @a in https://x/pull/14",
                "## Bug fixes",
                "* BUG: upper bug by @a in https://x/pull/1",
                "* fix(scope): lower fix by @a in https://x/pull/2",
                "## Tests",
                "* tst: tests by @a in https://x/pull/9",
                "## Documentation",
                "* doc: docs by @a in https://x/pull/8",
                "## Others",
                "* test: not a label by @a in https://x/pull/5",
                "* no label here by @a in https://x/pull/7",
                "* chore(ci): chore by @a in https://x/pull/6",
                "",
                "**Full Changelog**: https://github.com/o/r/compare/v1.0.0...v1.0.1",
            ],
        )


if __name__ == "__main__":
    unittest.main()
