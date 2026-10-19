"""Check static platform matrices and their path selection in Python CI."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/python.yaml").read_text())
JOBS = WORKFLOW["jobs"]
PLATFORM_STEP = next(
    step for step in JOBS["changes"]["steps"] if step.get("id") == "platform"
)


class TestPythonWorkflow(unittest.TestCase):
    def select(self, paths, *, event="pull_request", diff_available=True):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "changed-files.txt").write_text("\n".join(paths) + "\n")
            output = work / "outputs"
            script = (
                PLATFORM_STEP["run"]
                .replace("${{ github.event_name }}", event)
                .replace(
                    "${{ steps.gpu.outputs.diff_available }}",
                    str(diff_available).lower(),
                )
            )
            subprocess.run(
                ["bash", "-e", "-o", "pipefail", "-c", script],
                cwd=work,
                env={**os.environ, "GITHUB_OUTPUT": str(output)},
                check=True,
                capture_output=True,
                text=True,
            )
            return dict(line.split("=", 1) for line in output.read_text().splitlines())

    def test_matrices_are_static_and_preserve_python_endpoints(self):
        for name in ("build_test_job", "platform_test_job", "metal_test_job"):
            matrix = JOBS[name]["strategy"]["matrix"]
            self.assertIsInstance(matrix, dict)
            for values in matrix.values():
                self.assertIsInstance(values, list)
                self.assertTrue(all("${{" not in value for value in values))
        linux = JOBS["build_test_job"]["strategy"]["matrix"]
        self.assertEqual(linux["os"], ["ubuntu-latest"])
        self.assertEqual(
            linux["python-version"], ["3.10", "3.11", "3.12", "3.13", "3.14"]
        )
        platform = JOBS["platform_test_job"]["strategy"]["matrix"]
        self.assertEqual(platform["os"], ["macos-latest", "windows-latest"])
        self.assertEqual(platform["python-version"], ["3.10", "3.14"])

    def test_platform_versions_share_one_condition_and_test_step(self):
        platform = JOBS["platform_test_job"]
        self.assertEqual(
            platform["if"], "${{ needs.changes.outputs.platform == 'true' }}"
        )
        self.assertEqual(platform["steps"], JOBS["build_test_job"]["steps"])
        self.assertEqual(platform["steps"], JOBS["metal_test_job"]["steps"])
        self.assertEqual(
            platform["env"]["TEST_SUITE"], "${{ needs.changes.outputs.platform_suite }}"
        )

    def test_workflow_has_no_schedule(self):
        # PyYAML's YAML 1.1 loader treats the key `on` as a boolean.
        triggers = WORKFLOW.get("on", WORKFLOW.get(True))
        self.assertNotIn("schedule", triggers)
        self.assertIn("workflow_dispatch", triggers)

    def test_docs_frontend_and_unrelated_changes_skip_platforms(self):
        for paths in (
            [],
            ["README.md", "doc/source/getting_started.rst"],
            ["frontend/src/app/page.tsx", "xinference/ui/web/dist/index.html"],
            [".github/workflows/assign.yaml"],
        ):
            with self.subTest(paths=paths):
                self.assertEqual(
                    self.select(paths),
                    {"platform": "false", "metal": "false", "suite": "smoke"},
                )

    def test_shared_runtime_and_mlx_changes_run_platforms_and_metal(self):
        for path in (
            "xinference/core/worker.py",
            "xinference/api/restful_api.py",
            "xinference/client/restful/restful_client.py",
            "xinference/deploy/local.py",
            "xinference/conftest.py",
            "xinference/model/utils.py",
            "xinference/model/llm/mlx/core.py",
            "xinference/model/llm/llm_family.py",
            "xinference/model/llm/models/qwen.json",
            "xinference/model/llm/tool_parsers/glm4_tool_parser.py",
            "xinference/model/llm/sglang/xavier/settings.py",
            "xinference/model/llm/xavier/contract.py",
            "xinference/model/audio/whisper_mlx.py",
            "xinference/model/scheduler/core.py",
            "xinference/thirdparty/f5_tts/module.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    self.select([path]),
                    {"platform": "true", "metal": "true", "suite": "smoke"},
                )

    def test_other_engines_and_scripts_do_not_trigger_metal(self):
        for path in (
            "xinference/model/llm/vllm/core.py",
            "xinference/model/llm/sglang/core.py",
            "xinference/model/llm/transformers/core.py",
            "xinference/model/llm/llama_cpp/core.py",
            "xinference/model/embedding/core.py",
            "xinference/model/image/core.py",
            "xinference/model/video/core.py",
            "scripts/install.ps1",
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    self.select([path]),
                    {"platform": "true", "metal": "false", "suite": "smoke"},
                )

    def test_infrastructure_changes_require_full_coverage(self):
        for path in (
            ".github/workflows/python.yaml",
            ".github/platform-smoke-tests.txt",
            ".github/requirements/model-ci-constraints.txt",
            ".github/scripts/tests/test_python_workflow.py",
            "pyproject.toml",
            "build_backend.py",
            "build_web.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    self.select([path]),
                    {"platform": "true", "metal": "true", "suite": "full"},
                )

    def test_manual_checks_missing_diffs_and_mixed_changes(self):
        for options in ({"event": "workflow_dispatch"}, {"diff_available": False}):
            with self.subTest(options=options):
                self.assertEqual(
                    self.select(["README.md"], **options),
                    {"platform": "true", "metal": "true", "suite": "full"},
                )
        paths = ["xinference/model/llm/vllm/core.py", "xinference/core/worker.py"]
        for event in ("pull_request", "push"):
            for ordered in (paths, list(reversed(paths))):
                with self.subTest(event=event, paths=ordered):
                    self.assertEqual(
                        self.select(ordered, event=event),
                        {"platform": "true", "metal": "true", "suite": "smoke"},
                    )

    def test_smoke_manifest_is_valid(self):
        selectors = [
            line
            for line in (ROOT / ".github/platform-smoke-tests.txt")
            .read_text()
            .splitlines()
            if line and not line.startswith("#")
        ]
        self.assertTrue(selectors)
        self.assertEqual(selectors, sorted(set(selectors)))
        for selector in selectors:
            self.assertTrue((ROOT / selector.split("::")[0]).is_file(), selector)


if __name__ == "__main__":
    unittest.main()
