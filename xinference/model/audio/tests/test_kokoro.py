# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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

# CI: linux

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement

from xinference._model_catalog import load_model_catalog

from ....core.utils import filter_virtualenv_packages_by_markers
from ..kokoro import KokoroModel


@pytest.mark.parametrize(
    "dependency_source,engine",
    [
        ("audio-extra", None),
        ("cpu-image", None),
        ("Kokoro-82M", "PyTorch"),
        ("Kokoro-82M", "MLX"),
        ("Kokoro-82M-v1.1-zh", "PyTorch"),
    ],
)
def test_kokoro_dependencies_exclude_broken_preshed(dependency_source, engine):
    root = Path(__file__).resolve().parents[4]
    if dependency_source == "audio-extra":
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with (root / "pyproject.toml").open("rb") as source:
            packages = tomllib.load(source)["project"]["optional-dependencies"]["audio"]
    elif dependency_source == "cpu-image":
        requirements_path = (
            root
            / "xinference/deploy/docker/requirements_cpu/requirements_cpu-models.txt"
        )
        packages = [
            line.split("#", 1)[0].strip()
            for line in requirements_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        spec = next(
            spec
            for spec in load_model_catalog(
                Path(__file__).resolve().parents[1] / "models"
            )
            if spec["model_name"] == dependency_source
            and (spec.get("engine") or "PyTorch") == engine
        )
        packages = filter_virtualenv_packages_by_markers(
            spec["virtualenv"]["packages"], model_engine=engine, cuda_version=None
        )
    requirements = [
        Requirement(package) for package in packages if not package.startswith("#")
    ]
    preshed = [req for req in requirements if req.name == "preshed"]
    assert preshed
    assert all("3.0.13" in req.specifier for req in preshed)
    assert all("3.0.14" not in req.specifier for req in preshed)


def test_kokoro_resolves_bundled_voice(tmp_path):
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    voice_path = voices_dir / "af_sky.pt"
    voice_path.write_bytes(b"voice")
    model = KokoroModel(
        "kokoro",
        str(tmp_path),
        SimpleNamespace(model_ability=[]),
    )

    assert model._resolve_voice("af_sky") == str(voice_path)


def test_kokoro_preserves_explicit_and_missing_voices(tmp_path):
    model = KokoroModel(
        "kokoro",
        str(tmp_path),
        SimpleNamespace(model_ability=[]),
    )

    assert model._resolve_voice("/custom/voice.pt") == "/custom/voice.pt"
    assert model._resolve_voice("af_missing") == "af_missing"


def test_kokoro(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="Kokoro-82M",
        model_type="audio",
        compile=False,
        download_hub="huggingface",
    )
    model = client.get_model(model_uid)
    input_string = (
        "chat T T S is a text to speech model designed for dialogue applications."
    )
    response = model.speech(input_string)
    assert type(response) is bytes
    assert len(response) > 0

    response = model.speech(input_string, voice="af_sky")
    assert type(response) is bytes
    assert len(response) > 0

    # Test openai API
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    with client.audio.speech.with_streaming_response.create(
        model=model_uid, input=input_string, voice="am_fenrir"
    ) as response:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
            response.stream_to_file(f.name)
            assert os.stat(f.name).st_size > 0


def test_kokoro_zh(setup):
    endpoint, _ = setup
    from ....client import Client

    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="Kokoro-82M",
        model_type="audio",
        compile=False,
        download_hub="huggingface",
        lang_code="z",
    )
    model = client.get_model(model_uid)
    input_string = "重新启动即可更新"

    response = model.speech(input_string, voice="zf_xiaoyi")
    assert type(response) is bytes
    assert len(response) > 0

    # Test openai API
    import openai

    client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    with client.audio.speech.with_streaming_response.create(
        model=model_uid, input=input_string, voice="zf_xiaoyi"
    ) as response:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
            response.stream_to_file(f.name)
            assert os.stat(f.name).st_size > 0
