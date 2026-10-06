# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import subprocess
import sys


def test_fractional_model_sizes_survive_cached_optional_union_order():
    # Prime typing's cache before importing the specs. In Python <=3.13,
    # Optional[Union[str, int]] can then resolve to int-first in Pydantic v1.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from typing import Optional, Union
Optional[Union[int, str]]
from xinference.model.llm.llm_family import (
    LlamaCppLLMSpecV2, MLXLLMSpecV2, PytorchLLMSpecV2,
)
for cls, model_format in (
    (LlamaCppLLMSpecV2, 'ggufv2'),
    (PytorchLLMSpecV2, 'pytorch'),
    (MLXLLMSpecV2, 'mlx'),
):
    for size, activated, expected_size, expected_activated in (
        ('1_8', '5_1', '1_8', '5_1'),
        ('7', '3', 7, 3),
        (7, 3, 7, 3),
        (7, None, 7, None),
    ):
        spec = cls(
            model_format=model_format,
            model_size_in_billions=size,
            activated_size_in_billions=activated,
            quantization='none',
            model_file_name_template='model.gguf',
        )
        assert spec.model_size_in_billions == expected_size, spec
        assert spec.activated_size_in_billions == expected_activated, spec
        assert spec.dict()['activated_size_in_billions'] == expected_activated
""",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
