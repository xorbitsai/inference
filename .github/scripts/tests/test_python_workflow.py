"""Check static platform matrices and their path selection in Python CI."""

import ast
import json
import os
import subprocess
import sys
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
GPU_STEP = next(step for step in JOBS["changes"]["steps"] if step.get("id") == "gpu")


class TestPythonWorkflow(unittest.TestCase):
    def commands(self, platform, version="3.10", suite="fast"):
        step = next(
            step
            for step in JOBS["build_test_job"]["steps"]
            if step.get("name") == "Test with pytest"
        )
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "bin").mkdir()
            fake_pytest = work / "bin/pytest"
            fake_pytest.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['TRACE'], 'a') as trace:\n"
                "    trace.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )
            fake_pytest.chmod(0o755)
            trace = work / "trace"
            subprocess.run(
                [
                    "bash",
                    "-e",
                    "-o",
                    "pipefail",
                    "-c",
                    step["run"].replace("${{ matrix.python-version }}", version),
                ],
                cwd=work,
                env={
                    **os.environ,
                    "PATH": str(work / "bin") + os.pathsep + os.environ["PATH"],
                    "TRACE": str(trace),
                    "MODULE": "xinference",
                    "SUITE": suite,
                    "TEST_PLATFORM": platform,
                },
                check=True,
                capture_output=True,
                text=True,
            )
            return [json.loads(line) for line in trace.read_text().splitlines()]

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
                    {"platform": "false", "metal": "false", "suite": "fast"},
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
                    {"platform": "true", "metal": "true", "suite": "fast"},
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
                    {"platform": "true", "metal": "false", "suite": "fast"},
                )

    def test_infrastructure_changes_require_full_coverage(self):
        for path in (
            ".github/workflows/python.yaml",
            ".github/requirements/model-ci-constraints.txt",
            ".github/scripts/tests/test_python_workflow.py",
            ".pre-commit-config.yaml",
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
                        {
                            "platform": "true",
                            "metal": "true",
                            "suite": "full" if event == "push" else "fast",
                        },
                    )

    def test_main_pushes_run_full_platform_tests_without_a_schedule(self):
        for path, metal in (
            ("xinference/core/worker.py", "true"),
            ("xinference/model/llm/llama_cpp/core.py", "false"),
            ("scripts/install.ps1", "false"),
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    self.select([path], event="push"),
                    {"platform": "true", "metal": metal, "suite": "full"},
                )
        self.assertEqual(self.select(["README.md"], event="push")["platform"], "false")

    def test_renames_classify_both_source_and_destination(self):
        for source, destination in (
            ("xinference/core/worker.py", "doc/worker.py"),
            ("xinference/model/llm/mlx/core.py", "xinference/model/llm/vllm/core.py"),
        ):
            with (
                self.subTest(source=source),
                tempfile.TemporaryDirectory() as directory,
            ):
                work = Path(directory)

                def git(*args):
                    return subprocess.run(
                        [
                            "git",
                            "-c",
                            "core.hooksPath=/dev/null",
                            "-c",
                            "commit.gpgsign=false",
                            "-c",
                            "user.name=CI",
                            "-c",
                            "user.email=ci@example.invalid",
                            *args,
                        ],
                        cwd=work,
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()

                git("init", "-q")
                old = work / source
                old.parent.mkdir(parents=True)
                old.write_text("# unchanged content\n")
                git("add", ".")
                git("commit", "-qm", "before")
                base = git("rev-parse", "HEAD")
                new = work / destination
                new.parent.mkdir(parents=True)
                old.rename(new)
                git("add", "-A")
                git("commit", "-qm", "rename")
                head = git("rev-parse", "HEAD")
                self.assertEqual(
                    git("diff", "-M", "--name-only", base, head), destination
                )
                script = (
                    GPU_STEP["run"]
                    .replace("${{ github.event_name }}", "pull_request")
                    .replace("${{ github.event.pull_request.base.sha }}", base)
                    .replace("${{ github.event.pull_request.head.sha }}", head)
                    .replace("${{ github.event.before }}", base)
                    .replace("${{ github.sha }}", head)
                )
                subprocess.run(
                    ["bash", "-e", "-o", "pipefail", "-c", script],
                    cwd=work,
                    env={**os.environ, "GITHUB_OUTPUT": str(work / "outputs")},
                    check=True,
                    capture_output=True,
                    text=True,
                )
                paths = (work / "changed-files.txt").read_text().splitlines()
                self.assertEqual(set(paths), {source, destination})
                self.assertEqual(
                    self.select(paths),
                    {"platform": "true", "metal": "true", "suite": "fast"},
                )

    def test_engine_suites_run_only_on_supported_platforms(self):
        windows = self.commands("windows-latest")
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0][-1], "xinference")
        for path in (
            "xinference/model/llm/mlx",
            "xinference/model/llm/vllm",
            "xinference/model/llm/sglang",
            "xinference/model/llm/lmdeploy",
            "xinference/model/embedding/vllm",
            "xinference/model/rerank/vllm",
        ):
            index = windows[0].index(path)
            self.assertEqual(windows[0][index - 1], "--ignore")
        self.assertNotIn("xinference/model/llm/xavier", windows[0])
        macos = self.commands("macos-latest")
        self.assertEqual(len(macos), 2)
        self.assertEqual(macos[1][-1], "xinference/model/llm/mlx/tests")
        linux = self.commands("ubuntu-latest", suite="full")
        self.assertEqual(len(linux), 2)
        self.assertIn("benchmark/tests/test_benchmark_pd.py", linux[1])
        self.assertNotIn("xinference/model/llm/mlx/tests", linux[1])
        for name in (
            "test_anthropic_runtime.py",
            "test_context_length.py",
            "test_structured.py",
        ):
            self.assertIn(f"xinference/model/llm/mlx/tests/{name}", linux[1])

    def test_both_python_versions_use_identical_selection(self):
        for platform in ("windows-latest", "macos-latest"):
            for suite in ("fast", "full"):
                with self.subTest(platform=platform, suite=suite):
                    self.assertEqual(
                        self.commands(platform, "3.10", suite),
                        self.commands(platform, "3.14", suite),
                    )
        for command in self.commands("windows-latest", suite="full"):
            self.assertNotIn("-m", command)

    def test_new_engine_test_directories_require_platform_classification(self):
        # Shared directories discover new tests automatically. A new engine
        # directory needs an explicit platform decision before CI can pass.
        shared = {
            "llm": {"llama_cpp", "tests", "tool_parsers", "transformers", "xavier"},
            "embedding": {"flag", "llama_cpp", "sentence_transformers", "tests"},
            "rerank": {"llama_cpp", "sentence_transformers", "tests"},
        }
        restricted = {
            "llm": {"lmdeploy", "mlx", "sglang", "vllm"},
            "embedding": {"vllm"},
            "rerank": {"vllm"},
        }
        windows = self.commands("windows-latest")[0]
        ignores = {
            value for option, value in zip(windows, windows[1:]) if option == "--ignore"
        }
        for family, shared_names in shared.items():
            root = ROOT / "xinference/model" / family
            actual = {
                path.relative_to(root).parts[0] for path in root.rglob("test_*.py")
            }
            self.assertLessEqual(
                actual,
                shared_names | restricted[family],
                f"Classify new {family} test directories and update platform selection",
            )
            for name in restricted[family]:
                self.assertIn(f"xinference/model/{family}/{name}", ignores)
            for name in shared_names:
                self.assertNotIn(f"xinference/model/{family}/{name}", ignores)

    def test_new_files_and_cases_run_without_registration(self):
        command = self.commands("windows-latest")[0]
        marker = command[command.index("-m") + 1]
        self.assertEqual(marker, "not platform_slow")
        ignores = [
            value for option, value in zip(command, command[1:]) if option == "--ignore"
        ]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            package = work / "xinference"
            package.mkdir()
            (work / "pytest.ini").write_text(
                "[pytest]\nmarkers = platform_slow: slow model integration\n"
            )
            existing = package / "test_existing.py"
            existing.write_text(
                "import pytest\n"
                "def test_existing_unit(): pass\n"
                "@pytest.mark.platform_slow\n"
                "def test_download(): raise AssertionError('expensive integration')\n"
            )
            # A new case in an existing module and a completely new module
            # both run; even sharing a slow test's name prefix is safe.
            with existing.open("a") as output:
                output.write("def test_download_new_unit(): pass\n")
            (package / "test_new.py").write_text("def test_new_unit(): pass\n")
            args = [
                sys.executable,
                "-m",
                "pytest",
                "--noconftest",
                "-q",
                "-m",
                marker,
            ]
            for path in ignores:
                args.extend(["--ignore", path])
            result = subprocess.run(
                [*args, command[-1]],
                cwd=work,
                env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("3 passed, 1 deselected", result.stdout)

    def test_slow_marker_cannot_exclude_whole_modules_or_classes(self):
        for path in (ROOT / "xinference").rglob("test_*.py"):
            if "thirdparty" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "pytestmark":
                            self.assertNotIn(
                                "platform_slow", ast.unparse(node.value), path
                            )
                if isinstance(node, ast.ClassDef):
                    for decorator in node.decorator_list:
                        self.assertNotIn("platform_slow", ast.unparse(decorator), path)


if __name__ == "__main__":
    unittest.main()
