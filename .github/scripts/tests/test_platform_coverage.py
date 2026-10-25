"""Exercise the pre-commit coverage check against real Git changes."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class TestPlatformCoverage(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for relative in (
            ".github/scripts/check_test_platform_coverage.py",
            ".github/workflows/python.yaml",
        ):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        self.legacy = "xinference/core/tests/test_legacy.py"
        self.write(self.legacy)
        self.git("init", "-q", "-b", "main")
        self.commit()
        self.base = self.git("rev-parse", "HEAD")
        self.env = {
            key: value
            for key, value in os.environ.items()
            if key
            not in (
                "TEST_BASE_SHA",
                "TEST_HEAD_SHA",
                "PRE_COMMIT_FROM_REF",
                "PRE_COMMIT_TO_REF",
            )
        }

    def git(self, *args):
        return subprocess.check_output(
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
            cwd=self.root,
            text=True,
        ).strip()

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-qm", "test snapshot")

    def write(self, name, content="def test_example(): pass\n"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def check(self, *filenames, **env):
        return subprocess.run(
            [
                sys.executable,
                str(self.root / ".github/scripts/check_test_platform_coverage.py"),
                *filenames,
            ],
            cwd=self.root,
            env={**self.env, **env},
            capture_output=True,
            text=True,
        )

    def test_staged_test_fails_without_ci_shas_and_passes_with_marker(self):
        name = "xinference/core/tests/test_new.py"
        self.write(name)
        self.git("add", name)
        result = self.check(name)
        self.assertEqual(result.returncode, 1)
        self.assertIn(name, result.stderr)
        self.assertIn("# CI: linux", result.stderr)
        self.write(name, "# CI: linux\ndef test_example(): pass\n")
        self.git("add", name)
        self.assertEqual(self.check(name).returncode, 0)

    def test_initial_commit_is_checked_without_head(self):
        self.git("checkout", "--orphan", "initial")
        result = self.check(self.legacy)
        self.assertEqual(result.returncode, 1)
        self.assertIn(self.legacy, result.stderr)

    def test_all_files_checks_changes_without_rejecting_legacy_tests(self):
        self.assertEqual(self.check(self.legacy).returncode, 0)
        name = "xinference/core/tests/test_new.py"
        self.write(name)
        self.git("add", name)
        result = self.check(self.legacy, name)
        self.assertEqual(result.returncode, 1)
        self.assertIn(name, result.stderr)
        self.assertNotIn(self.legacy, result.stderr)

    def test_untracked_explicit_file_is_checked(self):
        name = "xinference/core/tests/test_untracked.py"
        self.write(name)
        result = self.check(name)
        self.assertEqual(result.returncode, 1)
        self.assertIn(name, result.stderr)
        # An unrelated untracked test is not part of a normal commit check.
        self.assertEqual(self.check().returncode, 0)

    def test_workflow_target_and_macos_tests_are_accepted(self):
        for name in (
            "xinference/deploy/test/test_new.py",
            "xinference/model/llm/mlx/tests/test_new.py",
        ):
            self.write(name)
        self.git("add", ".")
        self.assertEqual(self.check().returncode, 0)

    def test_deleted_and_vendored_tests_are_ignored(self):
        self.git("rm", self.legacy)
        self.write("xinference/thirdparty/test_vendor.py")
        self.git("add", ".")
        self.assertEqual(self.check().returncode, 0)

    def test_rename_into_uncovered_directory_is_checked(self):
        name = "xinference/core/tests/test_renamed.py"
        self.git("mv", self.legacy, name)
        result = self.check(name)
        self.assertEqual(result.returncode, 1)
        self.assertIn(name, result.stderr)

    def test_committed_branch_changes_are_checked_against_local_remote_ref(self):
        name = "xinference/core/tests/test_new.py"
        self.write(name)
        self.commit()
        for remote in ("upstream", "origin"):
            with self.subTest(remote=remote):
                ref = f"refs/remotes/{remote}/main"
                self.git("update-ref", ref, self.base)
                result = self.check(self.legacy, name)
                self.assertEqual(result.returncode, 1)
                self.assertIn(name, result.stderr)
                self.assertNotIn(self.legacy, result.stderr)
                self.git("update-ref", "-d", ref)

    def test_pre_commit_ref_range_is_honored(self):
        name = "xinference/core/tests/test_new.py"
        self.write(name)
        self.commit()
        head = self.git("rev-parse", "HEAD")
        result = self.check(PRE_COMMIT_FROM_REF=self.base, PRE_COMMIT_TO_REF=head)
        self.assertEqual(result.returncode, 1)
        self.assertIn(name, result.stderr)

    def test_ci_head_excludes_unrelated_main_changes_in_merge_checkout(self):
        self.git("checkout", "-qb", "topic")
        self.write("xinference/deploy/test/test_branch.py")
        self.commit()
        head = self.git("rev-parse", "HEAD")
        self.git("checkout", "-q", "main")
        unrelated = "xinference/core/tests/test_unrelated.py"
        self.write(unrelated)
        self.commit()
        self.git("merge", "-q", "--no-ff", "topic", "-m", "CI merge checkout")
        result = self.check(TEST_BASE_SHA=self.base, TEST_HEAD_SHA=head)
        self.assertEqual(result.returncode, 0, result.stderr)
        # Using the merge checkout's HEAD would incorrectly include main's test.
        result = self.check(TEST_BASE_SHA=self.base, TEST_HEAD_SHA="HEAD")
        self.assertEqual(result.returncode, 1)
        self.assertIn(unrelated, result.stderr)

    def test_invalid_explicit_ref_fails_instead_of_skipping(self):
        result = self.check(TEST_BASE_SHA="missing", TEST_HEAD_SHA="HEAD")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Cannot validate test platform coverage", result.stderr)


if __name__ == "__main__":
    unittest.main()
