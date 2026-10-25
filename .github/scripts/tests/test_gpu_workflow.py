"""Exercise GPU selection and installer commands without installing model stacks."""

import itertools
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
JOBS = yaml.safe_load((ROOT / ".github/workflows/python.yaml").read_text())["jobs"]
GPU_STEP = next(step for step in JOBS["changes"]["steps"] if step.get("id") == "gpu")
GROUPS = ("llm", "embedding", "image", "audio")


class TestGPUWorkflow(unittest.TestCase):
    def select(self, paths, event="pull_request", base="base"):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "bin").mkdir()
            git = work / "bin/git"
            git.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$CHANGED_PATHS\"\n")
            git.chmod(0o755)
            script = GPU_STEP["run"]
            for key, value in {
                "github.event_name": event,
                "github.event.pull_request.base.sha": base,
                "github.event.pull_request.head.sha": "head",
                "github.event.before": base,
                "github.sha": "head",
            }.items():
                script = script.replace("${{ " + key + " }}", value)
            subprocess.run(
                ["bash", "-e", "-o", "pipefail", "-c", script],
                cwd=work,
                env={
                    **os.environ,
                    "PATH": str(work / "bin") + os.pathsep + os.environ["PATH"],
                    "CHANGED_PATHS": "\n".join(paths),
                    "GITHUB_OUTPUT": str(work / "outputs"),
                },
                check=True,
                capture_output=True,
                text=True,
            )
            return dict(
                line.split("=", 1)
                for line in (work / "outputs").read_text().splitlines()
            )

    def install(self, groups):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            trace = work / "trace"
            for command in ("python", "conda", "sudo"):
                tool = work / command
                tool.write_text(
                    f"#!{sys.executable}\n"
                    "import json, os, sys\n"
                    "from pathlib import Path\n"
                    "with open(os.environ['TRACE'], 'a') as out:\n"
                    "    out.write(json.dumps([Path(sys.argv[0]).name, sys.argv[1:], "
                    "os.environ.get('PIP_CONSTRAINT')]) + '\\n')\n"
                )
                tool.chmod(0o755)
            result = subprocess.run(
                [
                    "bash",
                    str(ROOT / ".github/scripts/install_gpu_test_dependencies.sh"),
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PATH": str(work) + os.pathsep + os.environ["PATH"],
                    "TRACE": str(trace),
                    "GPU_TEST_GROUPS": groups,
                },
                capture_output=True,
                text=True,
            )
            calls = (
                [json.loads(line) for line in trace.read_text().splitlines()]
                if trace.exists()
                else []
            )
            return result, calls

    def test_unrelated_changes_skip_gpu_and_model_changes_select_their_family(self):
        for path in (
            "README.md",
            "doc/source/index.rst",
            "frontend/page.tsx",
            "xinference/ui/index.html",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.select([path])["gpu"], "false")
        for group, path in zip(
            GROUPS,
            (
                "xinference/model/llm/transformers/core.py",
                "xinference/model/rerank/core.py",
                "xinference/model/image/tests/test_stable_diffusion.py",
                "xinference/model/audio/tests/test_whisper.py",
            ),
        ):
            with self.subTest(group=group):
                self.assertEqual(self.select([path])["groups"], group)
        self.assertEqual(
            self.select(
                ["xinference/model/audio/core.py", "xinference/model/image/core.py"]
            )["groups"],
            "image audio",
        )

    def test_unrelated_engine_and_cpu_test_changes_skip_gpu(self):
        paths = [
            "xinference/model/llm/vllm/core.py",
            "xinference/model/llm/sglang/core.py",
            "xinference/model/llm/mlx/core.py",
            "xinference/model/llm/llama_cpp/core.py",
            "xinference/model/llm/xavier/contract.py",
            "xinference/model/embedding/vllm/core.py",
            "xinference/model/rerank/vllm/core.py",
            "xinference/model/audio/whisper_mlx.py",
            "xinference/model/audio/f5tts_mlx.py",
            "xinference/model/audio/cosyvoice.py",
            "xinference/model/audio/qwen3_tts.py",
            "xinference/model/llm/tests/test_context_length.py",
            "xinference/model/audio/tests/test_audio_batching.py",
            "xinference/model/image/tests/test_got_ocr2.py",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(self.select([path])["gpu"], "false")
        self.assertEqual(
            self.select(paths + ["xinference/model/llm/transformers/core.py"])[
                "groups"
            ],
            "llm",
        )

    def test_all_group_combinations_install_only_selected_optional_stacks(self):
        for size in range(1, len(GROUPS) + 1):
            for groups in itertools.combinations(GROUPS, size):
                with self.subTest(groups=groups):
                    result, calls = self.install(" ".join(groups))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    installs = [
                        args
                        for tool, args, _ in calls
                        if tool == "python" and args[:3] == ["-m", "pip", "install"]
                    ]
                    packages = installs[0]
                    self.assertIn("torch==2.7.0", packages)
                    self.assertIn(
                        ".[audio,dev]" if "audio" in groups else ".[dev]", packages
                    )
                    for package, needed in (
                        ("tensorizer", "llm" in groups),
                        ("FlagEmbedding==1.3.5", "embedding" in groups),
                        ("diffusers==0.35.1", bool({"audio", "image"} & set(groups))),
                        ("funasr==1.2.7", "audio" in groups),
                    ):
                        self.assertEqual(package in packages, needed, package)
                    self.assertEqual(
                        any(tool == "conda" for tool, _, _ in calls), "audio" in groups
                    )
                    self.assertEqual(
                        any(tool == "sudo" for tool, _, _ in calls), "audio" in groups
                    )
                    self.assertEqual(len(installs), 2 if "embedding" in groups else 1)
                    self.assertTrue(
                        all(
                            constraint
                            == str(
                                ROOT / ".github/requirements/model-ci-constraints.txt"
                            )
                            for _, _, constraint in calls
                        )
                    )

    def test_invalid_groups_fail_before_installing_anything(self):
        for groups in ("", "   ", "video", "audio unknown", "llm; echo invalid"):
            with self.subTest(groups=groups):
                result, calls = self.install(groups)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(calls, [])

    def test_audio_model_cache_is_not_restored_for_other_groups(self):
        steps = JOBS["gpu_test_job"]["steps"]
        cache = next(
            step
            for step in steps
            if step.get("name") == "Restore ModelScope model cache"
        )
        self.assertEqual(
            cache["if"], "${{ contains(needs.changes.outputs.groups, 'audio') }}"
        )
        pip_cache = next(
            step for step in steps if step.get("name") == "Restore pip package cache"
        )
        self.assertIn("needs.changes.outputs.groups", pip_cache["with"]["key"])
        self.assertNotIn(".github/workflows/python.yaml", pip_cache["with"]["key"])


if __name__ == "__main__":
    unittest.main()
