import os
import shutil
from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .....constants import XINFERENCE_CACHE_DIR
from ...llm_family import LLMFamilyV1, PytorchLLMSpecV1, cache
from ..tensorizer_utils import (
    get_tensorizer_dir,
    tensorizer_serialize_model,
    tensorizer_serialize_pretrained,
)


# case1: test if tensorizer_serialize_model and .tensor file exists in the correct path
class TestTensorizerSerializeModel:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # Setup: Load the model and tokenizer
        model_full_name = "qwen1.5-chat-pytorch-0_5b"
        self.model_path = f"{XINFERENCE_CACHE_DIR}/{model_full_name}"
        self.tensorizer_dir = get_tensorizer_dir(self.model_path)
        spec = PytorchLLMSpecV1(
            model_format="pytorch",
            model_size_in_billions="0_5",
            quantizations=["4-bit", "8-bit", "none"],
            model_id="Qwen/Qwen1.5-0.5B-Chat",
            model_revision=None,
        )
        family = LLMFamilyV1(
            version=1,
            context_length=32768,
            model_type="LLM",
            model_name="qwen1.5-chat",
            model_lang=["en", "zh"],
            model_ability=["chat", "tools"],
            model_specs=[spec],
            prompt_style=None,
        )

        if not os.path.exists(self.model_path):
            cache(llm_family=family, llm_spec=spec, quantization=None)

        self.model_config = AutoConfig.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
        )
        self.model_prefix = "model"
        self.force = True
        yield

        # Cleanup: Remove the entire directories after the test
        self.cleanup_directory(self.tensorizer_dir)

    def cleanup_directory(self, directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)

    # Test if model.tensor file and model-config.json file generated
    def test_tensor_file_exists(self):
        expected_tensor_file = f"{self.tensorizer_dir}/{self.model_prefix}.tensors"
        expected_model_config_file = (
            f"{self.tensorizer_dir}/{self.model_prefix}-config.json"
        )

        if os.path.exists(expected_tensor_file):
            os.remove(expected_tensor_file)

        if os.path.exists(expected_model_config_file):
            os.remove(expected_model_config_file)

        tensorizer_serialize_model(
            self.model,
            self.model_config,
            self.tensorizer_dir,
            self.model_prefix,
            self.force,
        )

        assert os.path.exists(
            expected_tensor_file
        ), f"{expected_tensor_file} does not exist"

        assert os.path.exists(
            expected_model_config_file
        ), f"{expected_model_config_file} does not exist"

    # Test if tokenizer.zip file generated
    def test_tokenizer_file_exists(self):
        expected_tokenizer_file = f"{self.tensorizer_dir}/tokenizer.zip"
        if os.path.exists(expected_tokenizer_file):
            os.remove(expected_tokenizer_file)

        tensorizer_serialize_pretrained(
            self.tokenizer, self.tensorizer_dir, "tokenizer"
        )

        assert os.path.exists(
            expected_tokenizer_file
        ), f"{expected_tokenizer_file} does not exist"


# case2: test if serialized file exists, load from cache
@pytest.fixture
def model_mock():
    return MagicMock()


@pytest.fixture
def model_config_mock():
    mock = MagicMock()
    mock.to_dict.return_value = {"dummy_key": "dummy_value"}
    return mock


def test_tensorizer_serialize_uses_cache_when_files_exist_and_not_forced(
    tmp_path, model_mock, model_config_mock
):
    tensor_directory = str(tmp_path)
    model_prefix = "test_model"
    expected_config_path = tmp_path / f"{model_prefix}-config.json"
    expected_model_path = tmp_path / f"{model_prefix}.tensors"

    expected_config_path.write_text("dummy config")
    expected_model_path.write_text("dummy model")

    with patch(
        "xinference.model.llm.pytorch.tensorizer_utils.file_is_non_empty",
        return_value=True,
    ) as mock_file_is_non_empty:
        model_path = tensorizer_serialize_model(
            model=model_mock,
            model_config=model_config_mock,
            tensor_directory=tensor_directory,
            model_prefix=model_prefix,
            force=False,
        )

    mock_file_is_non_empty.assert_any_call(str(expected_config_path))
    mock_file_is_non_empty.assert_any_call(str(expected_model_path))

    model_config_mock.to_dict.assert_not_called()

    assert model_path == str(expected_model_path)
