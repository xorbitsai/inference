"""Regression checks for PR coverage and the full cross-platform fallback."""

import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "ci_platform_matrix.py"
spec = importlib.util.spec_from_file_location("ci_platform_matrix", SCRIPT)
assert spec is not None and spec.loader is not None
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


class TestPlatformMatrix(unittest.TestCase):
    def matrix(self, paths, event="pull_request"):
        entries = ci.build_matrix(event, paths)["include"]
        linux = [entry for entry in entries if entry["os"] == "ubuntu-latest"]
        self.assertEqual(
            [entry["python-version"] for entry in linux], list(ci.PYTHON_VERSIONS)
        )
        self.assertTrue(all(entry["suite"] == "full" for entry in linux))
        return [entry for entry in entries if entry not in linux]

    def assert_full(self, entries):
        self.assertEqual(len(entries), 5)
        for platform in ci.PLATFORMS:
            ordinary = [
                entry
                for entry in entries
                if entry["os"] == platform and entry["module"] == "xinference"
            ]
            self.assertEqual(
                [entry["python-version"] for entry in ordinary], ["3.10", "3.14"]
            )
            self.assertTrue(all(entry["suite"] == "full" for entry in ordinary))
        self.assertEqual(sum(entry["module"] == "metal" for entry in entries), 1)

    def test_docs_and_frontend_skip_platform_jobs(self):
        for paths in (
            [],
            ["doc/source/getting_started.rst"],
            ["frontend/src/app/page.tsx", "xinference/ui/web/dist/index.html"],
            [".github/workflows/assign.yaml", "README.md"],
        ):
            with self.subTest(paths=paths):
                self.assertEqual(self.matrix(paths), [])

    def test_backend_runs_both_python_versions_on_each_platform(self):
        for path in (
            "xinference/core/worker.py",
            "xinference/api/restful_api.py",
            "xinference/client/restful/restful_client.py",
            "xinference/deploy/local.py",
            "xinference/conftest.py",
        ):
            with self.subTest(path=path):
                entries = self.matrix([path])
                ordinary = [e for e in entries if e["module"] == "xinference"]
                self.assertEqual(
                    [(e["os"], e["python-version"]) for e in ordinary],
                    [
                        (platform, version)
                        for platform in ci.PLATFORMS
                        for version in ("3.10", "3.14")
                    ],
                )
                self.assertTrue(all(e["suite"] == "smoke" for e in ordinary))
                self.assertEqual(len(entries), 5)

    def test_python_endpoints_always_have_the_same_conditions_and_suite(self):
        for event in ("pull_request", "push", "schedule", "workflow_dispatch"):
            for paths in (
                None,
                [],
                ["README.md"],
                ["frontend/src/app/page.tsx"],
                ["xinference/core/worker.py"],
                ["xinference/model/llm/vllm/core.py"],
                ["xinference/model/llm/mlx/core.py"],
                ["pyproject.toml"],
            ):
                with self.subTest(event=event, paths=paths):
                    entries = self.matrix(paths, event)
                    for platform in ci.PLATFORMS:
                        suites = {
                            e["python-version"]: e["suite"]
                            for e in entries
                            if e["os"] == platform and e["module"] == "xinference"
                        }
                        self.assertIn(set(suites), (set(), {"3.10", "3.14"}))
                        self.assertLessEqual(len(set(suites.values())), 1)

    def test_metal_follows_engine_and_shared_dependencies(self):
        for path in (
            "xinference/model/llm/mlx/core.py",
            "xinference/model/llm/mlx/tests/test_mlx.py",
            "xinference/model/llm/xavier/contract.py",
            "xinference/model/llm/sglang/xavier/settings.py",
            "xinference/model/llm/sglang/xavier/directory.py",
            "xinference/model/llm/llm_family.py",
            "xinference/model/llm/models/qwen.json",
            "xinference/model/llm/tool_parsers/glm4_tool_parser.py",
            "xinference/model/llm/new_shared_package/utils.py",
            "xinference/model/utils.py",
            "xinference/model/scheduler/core.py",
            "xinference/model/audio/whisper_mlx.py",
            "xinference/thirdparty/f5_tts/module.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.matrix([path])[-1]["module"], "metal")

    def test_other_engines_keep_smoke_without_metal(self):
        for path in (
            "xinference/model/llm/vllm/core.py",
            "xinference/model/llm/sglang/core.py",
            "xinference/model/llm/transformers/core.py",
            "xinference/model/embedding/core.py",
            "xinference/model/image/core.py",
            "xinference/model/video/core.py",
            "scripts/install.ps1",
        ):
            with self.subTest(path=path):
                entries = self.matrix([path])
                self.assertEqual(len(entries), 4)
                self.assertTrue(all(e["suite"] == "smoke" for e in entries))

    def test_mixed_changes_preserve_metal(self):
        entries = self.matrix(
            ["frontend/src/app/page.tsx", "xinference/model/llm/mlx/core.py"]
        )
        self.assertEqual(len(entries), 5)

    def test_infrastructure_changes_run_full_matrix(self):
        for path in (
            *ci.INFRASTRUCTURE_FILES,
            ".github/requirements/model-ci-constraints.txt",
            ".github/scripts/ci_platform_matrix.py",
        ):
            with self.subTest(path=path):
                self.assert_full(self.matrix([path]))

    def test_scheduled_manual_and_missing_diffs_run_full_matrix(self):
        for event, paths in (
            ("schedule", []),
            ("workflow_dispatch", ["README.md"]),
            ("pull_request", None),
            ("push", None),
        ):
            with self.subTest(event=event, paths=paths):
                self.assert_full(self.matrix(paths, event))

    def test_push_uses_the_same_path_selection(self):
        self.assertEqual(
            self.matrix(["xinference/core/worker.py"], "push"),
            self.matrix(["xinference/core/worker.py"]),
        )


class TestChangedPaths(unittest.TestCase):
    @patch.object(ci.subprocess, "run")
    def test_pull_request_uses_merge_base(self, run):
        run.return_value.stdout = "xinference/core/worker.py\0a path with spaces.py\0"
        event = {"pull_request": {"base": {"sha": "abc"}, "head": {"sha": "def"}}}
        self.assertEqual(
            ci.changed_paths("pull_request", event),
            ["xinference/core/worker.py", "a path with spaces.py"],
        )
        self.assertEqual(
            run.call_args.args[0],
            ["git", "diff", "--name-only", "-z", "abc...def", "--"],
        )

    @patch.object(ci.subprocess, "run")
    def test_push_uses_previous_head(self, run):
        run.return_value.stdout = "README.md\0"
        self.assertEqual(
            ci.changed_paths("push", {"before": "abc", "after": "def"}),
            ["README.md"],
        )
        self.assertIn("abc..def", run.call_args.args[0])

    @patch.object(ci.subprocess, "run")
    def test_unusable_diff_runs_full_matrix(self, run):
        run.side_effect = subprocess.CalledProcessError(128, "git")
        self.assertIsNone(ci.changed_paths("push", {"before": "abc", "after": "def"}))
        self.assertIsNone(
            ci.changed_paths("push", {"before": "0" * 40, "after": "def"})
        )
        self.assertIsNone(ci.changed_paths("push", {}))
        self.assertIsNone(ci.changed_paths("schedule", {}))
        run.assert_called_once()

    def test_smoke_manifest_is_valid(self):
        root = SCRIPT.parents[2]
        manifest = root / ".github/platform-smoke-tests.txt"
        selectors = [
            line
            for line in manifest.read_text().splitlines()
            if line and not line.startswith("#")
        ]
        self.assertTrue(selectors)
        self.assertEqual(len(selectors), len(set(selectors)))
        self.assertEqual(selectors, sorted(selectors))
        for line in selectors:
            self.assertTrue((root / line.split("::")[0]).is_file(), line)


if __name__ == "__main__":
    unittest.main()
