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

from ..supervisor import _merge_audio_model_registrations, _merge_worker_engine_params


def test_merge_platform_specific_audio_catalogs():
    linux_registration = {
        "model_name": "whisper-tiny",
        "is_builtin": True,
        "model_specs": [
            {
                "model_engine": "transformers",
                "model_format": "pytorch",
                "cache_name": None,
                "model_hub": "huggingface",
                "model_id": "openai/whisper-tiny",
                "cache_status": True,
            }
        ],
        "download_hubs": ["huggingface"],
    }
    apple_registration = {
        "model_name": "whisper-tiny",
        "is_builtin": True,
        "model_specs": [
            {
                **linux_registration["model_specs"][0],
                "cache_status": False,
            },
            {
                "model_engine": "MLX",
                "model_format": "mlx",
                "cache_name": "whisper-tiny-mlx",
                "model_hub": "huggingface",
                "model_id": "mlx-community/whisper-tiny",
                "cache_status": True,
            },
        ],
        "download_hubs": ["huggingface"],
    }

    merged = _merge_audio_model_registrations(
        [linux_registration, apple_registration], detailed=True
    )

    assert len(merged) == 1
    assert {
        (spec["model_engine"], spec["model_format"], spec["cache_status"])
        for spec in merged[0]["model_specs"]
    } == {("transformers", "pytorch", True), ("MLX", "mlx", True)}


def test_merge_platform_specific_audio_engine_params():
    merged = _merge_worker_engine_params(
        [
            {"transformers": [{"model_format": "pytorch"}]},
            {
                "transformers": [{"model_format": "pytorch"}],
                "MLX": [{"model_format": "mlx"}],
            },
        ]
    )

    assert merged == {
        "transformers": [{"model_format": "pytorch"}],
        "MLX": [{"model_format": "mlx"}],
    }
