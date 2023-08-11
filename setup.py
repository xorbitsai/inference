# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import platform
import sys
import shutil
import subprocess
import warnings
from sysconfig import get_config_vars

from pkg_resources import parse_version
from setuptools import Command, setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.sdist import sdist

# From https://github.com/pandas-dev/pandas/pull/24274:
# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py
if sys.platform == "darwin":
    if "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
        current_system = platform.mac_ver()[0]
        python_target = get_config_vars().get(
            "MACOSX_DEPLOYMENT_TARGET", current_system
        )
        target_macos_version = "10.9"

        parsed_python_target = parse_version(python_target)
        parsed_current_system = parse_version(current_system)
        parsed_macos_version = parse_version(target_macos_version)
        if parsed_python_target <= parsed_macos_version <= parsed_current_system:
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = target_macos_version


repo_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(repo_root)


class ExtraCommandMixin:
    _extra_pre_commands = []

    def run(self):
        [self.run_command(cmd) for cmd in self._extra_pre_commands]
        super().run()

    @classmethod
    def register_pre_command(cls, cmd):
        cls._extra_pre_commands.append(cmd)


class CustomInstall(ExtraCommandMixin, install):
    pass


class CustomDevelop(ExtraCommandMixin, develop):
    pass


class CustomSDist(ExtraCommandMixin, sdist):
    pass

class BuildWeb(Command):
    """build_web command"""

    user_options = []
    _web_src_path = "xinference/web/ui"
    _web_dest_path = "xinference/web/ui/build/index.html"
    _commands = [
        ["npm", "install"],
        ["npm", "run", "build"],
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    @classmethod
    def run(cls):
        if int(os.environ.get("NO_WEB_UI", "0")):
            return

        npm_path = shutil.which("npm")
        web_src_path = os.path.join(repo_root, *cls._web_src_path.split("/"))
        web_dest_path = os.path.join(repo_root, *cls._web_dest_path.split("/"))

        if npm_path is None:
            warnings.warn("Cannot find NPM, may affect displaying Xinference web UI")
            return
        else:
            replacements = {"npm": npm_path}
            cmd_errored = False
            for cmd in cls._commands:
                cmd = [replacements.get(c, c) for c in cmd]
                proc_result = subprocess.run(cmd, cwd=web_src_path)
                if proc_result.returncode != 0:
                    warnings.warn(f'Failed when running `{" ".join(cmd)}`')
                    cmd_errored = True
                    break
            if not cmd_errored:
                assert os.path.exists(web_dest_path)


CustomInstall.register_pre_command("build_web")
CustomDevelop.register_pre_command("build_web")
CustomSDist.register_pre_command("build_web")

sys.path.append(repo_root)
versioneer = __import__("versioneer")


# build long description
def build_long_description():
    readme_path = os.path.join(os.path.abspath(repo_root), "README.md")

    with open(readme_path, encoding="utf-8") as f:
        return f.read()


setup_options = dict(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(
        {
            "build_web": BuildWeb,
            "install": CustomInstall,
            "develop": CustomDevelop,
            "sdist": CustomSDist,
        }
    ),
    long_description=build_long_description(),
    long_description_content_type="text/markdown",
)
setup(**setup_options)
