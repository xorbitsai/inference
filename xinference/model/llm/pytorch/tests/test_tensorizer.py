import os
import shutil
from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .....constants import XINFERENCE_CACHE_DIR
from ...llm_family import LLMFamilyV1, PytorchLLMSpecV1, cache
from ..tensorizer_utils import (
    _tensorizer_serialize_model,
    get_tensorizer_dir,
    save_to_tensorizer,
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
        self._cleanup_directory(self.tensorizer_dir)

    def _cleanup_directory(self, directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)

    # Test if model.tensor & tokenizer.zip files generated
    def test_tensor_file_exists(self):
        expected_tensor_file = f"{self.tensorizer_dir}/{self.model_prefix}.tensors"
        expected_tokenizer_file = f"{self.tensorizer_dir}/tokenizer.zip"

        if os.path.exists(expected_tensor_file):
            os.remove(expected_tensor_file)

        if os.path.exists(expected_tokenizer_file):
            os.remove(expected_tokenizer_file)

        save_to_tensorizer(
            self.model_path,
            self.model,
            [("tokenizer", self.tokenizer)],
        )

        assert os.path.exists(
            expected_tensor_file
        ), f"{expected_tensor_file} does not exist"

        assert os.path.exists(
            expected_tokenizer_file
        ), f"{expected_tokenizer_file} does not exist"


@pytest.fixture
def mock_environment(tmp_path):
    model_path = str(tmp_path / "model")
    tensorizer_dir = str(tmp_path / "tensorizer")
    os.makedirs(tensorizer_dir, exist_ok=True)
    tensor_path = f"{tensorizer_dir}/model.tensors"
    # Create a dummy cache file to simulate cache existence
    with open(tensor_path, "w") as f:
        f.write("dummy content")
    return model_path, tensorizer_dir, tensor_path


@patch("xinference.model.llm.pytorch.tensorizer_utils.get_tensorizer_dir")
@patch("xinference.model.llm.pytorch.tensorizer_utils.logger")
def test_tensorizer_serialize_model_cache_exists(
    mock_logger, mock_get_tensorizer_dir, mock_environment
):
    model_path, tensorizer_dir, tensor_path = mock_environment
    mock_get_tensorizer_dir.return_value = tensorizer_dir
    model = MagicMock()

    # Call the function with the mocked environment
    _tensorizer_serialize_model(model_path, model)

    # Check if the logger.info was called with the expected message, indicating early return due to cache existence
    mock_logger.info.assert_called_with(
        f"Cache {tensor_path} exists, skip tensorizer serialize model"
    )
