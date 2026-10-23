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
import base64
import copy
import json
import sys
import tempfile
import types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from packaging import version

from ...tool_parsers.qwen_tool_parser import QwenToolParser


class TestVLLMFP8Quantization:
    @pytest.fixture
    def model(self, monkeypatch, tmp_path):
        from ...llm_family import match_llm
        from .. import core

        monkeypatch.setattr(core, "VLLM_VERSION", version.parse("0.22.0"))
        family = copy.deepcopy(match_llm("glm-4.5", "fp8", 106, "FP8", "huggingface"))
        model = core.VLLMChatModel("test-glm-fp8", family, str(tmp_path), {})
        model._device_count = 1
        return model

    @pytest.mark.parametrize("section", [None, "text_config"])
    @pytest.mark.parametrize(
        "quant_config,expected",
        [
            ({"quant_method": "compressed-tensors"}, None),
            ({"quant_method": "fp8"}, "fp8"),
            ({"quant_method": "unknown"}, None),
            ({"quant_method": 8}, None),
            ({"quant_method": ""}, None),
            ({"quant_method": None}, "fp8"),
            ({}, "fp8"),
            (None, "fp8"),
            ([], None),
        ],
    )
    def test_checkpoint_quantizer(
        self, model, tmp_path, section, quant_config, expected
    ):
        config = {"quantization_config": quant_config}
        if section:
            config = {section: config}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert model._sanitize_model_config({})["quantization"] == expected

    @pytest.mark.parametrize(
        "top,nested,expected",
        [
            ("fp8", "compressed-tensors", "fp8"),
            ("compressed-tensors", "fp8", None),
            (None, "compressed-tensors", None),
        ],
    )
    def test_top_level_precedence(self, model, tmp_path, top, nested, expected):
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "quantization_config": {"quant_method": top},
                    "text_config": {"quantization_config": {"quant_method": nested}},
                }
            )
        )
        assert model._sanitize_model_config({})["quantization"] == expected

    @pytest.mark.parametrize("config", [None, {}])
    @pytest.mark.parametrize("launch_config", [None, {}, {"quantization": "none"}])
    def test_missing_metadata_keeps_fp8(self, model, tmp_path, config, launch_config):
        if config is not None:
            (tmp_path / "config.json").write_text(json.dumps(config))
        assert model._sanitize_model_config(launch_config)["quantization"] == "fp8"

    @pytest.mark.parametrize("contents", ["{", "null", "[]", '{"text_config": []}'])
    def test_malformed_config_defers_to_vllm(self, model, tmp_path, contents):
        (tmp_path / "config.json").write_text(contents)
        assert model._sanitize_model_config({})["quantization"] is None

    def test_unreadable_config_defers_to_vllm(self, model, tmp_path, caplog):
        (tmp_path / "config.json").mkdir()
        assert model._sanitize_model_config({})["quantization"] is None
        assert "Cannot read checkpoint quantization config" in caplog.text

    @pytest.mark.parametrize("method", ["fp8", "compressed-tensors"])
    @pytest.mark.parametrize(
        "launch_config,expected",
        [
            (
                {"model_quantization": "compressed-tensors", "quantization": "fp8"},
                "compressed-tensors",
            ),
            ({"quantization": "compressed-tensors"}, "compressed-tensors"),
            ({"model_quantization": "fp8"}, "fp8"),
        ],
    )
    def test_explicit_quantizer_wins(
        self, model, tmp_path, method, launch_config, expected
    ):
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "quantization_config": {"quant_method": method},
                }
            )
        )
        assert model._sanitize_model_config(launch_config)["quantization"] == expected

    @pytest.mark.parametrize("model_format", ["pytorch", "gptq", "awq", "fp4", "bnb"])
    def test_non_fp8_format_keeps_default(self, model, tmp_path, model_format):
        model.model_spec.model_format = model_format
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "quantization_config": {"quant_method": "fp8"},
                }
            )
        )
        assert model._sanitize_model_config({})["quantization"] is None


def filter_ids_and_created(data):
    if isinstance(data, list):
        return [filter_ids_and_created(item) for item in data]
    elif isinstance(data, dict):
        return {
            k: filter_ids_and_created(v)
            for k, v in data.items()
            if k not in ["id", "created"]
        }
    return data


class TestVLLMMultiModelChatTemplate:
    @pytest.mark.asyncio
    async def test_kimi_k3_uses_tokenizer_native_renderer_without_jinja(self):
        from ..core import VLLMMultiModel

        class NativeTokenizer:
            chat_template = None

            def apply_chat_template(self, messages, **kwargs):
                assert "chat_template" not in kwargs
                assert kwargs["add_generation_prompt"] is True
                assert kwargs["tools"] == [{"type": "function"}]
                return "native-kimi-k3-prompt"

        model = object.__new__(VLLMMultiModel)
        model.model_uid = "kimi-k3-test"
        model.model_family = MagicMock()
        model.model_family.model_name = "Kimi-K3"
        model.model_family.model_family = None
        model.model_family.model_ability = ["chat", "vision", "reasoning", "tools"]
        model.model_family.chat_template = ""
        tokenizer = NativeTokenizer()

        async def get_tokenizer(_):
            return tokenizer

        model._get_tokenizer = get_tokenizer
        chat_template, resolved_tokenizer = (
            await model._get_chat_template_and_tokenizer("Kimi-K3")
        )
        assert chat_template is None
        assert resolved_tokenizer is tokenizer
        assert (
            model.get_full_context(
                [{"role": "user", "content": "hello"}],
                chat_template,
                tokenizer=resolved_tokenizer,
                tools=[{"type": "function"}],
            )
            == "native-kimi-k3-prompt"
        )


class TestVLLMBase64Media:
    class FakeVideo:
        shape = (4, 3, 336, 336)

    @pytest.fixture
    def model(self):
        from ..core import VLLMMultiModel

        return object.__new__(VLLMMultiModel)

    @pytest.mark.parametrize(
        ("media_key", "mime_type", "expected_suffix", "expects_file_uri"),
        [
            ("image_url", "image/png", ".png", False),
            ("image_url", "image/jpeg", ".jpg", False),
            ("video_url", "video/mp4", ".mp4", True),
            ("video_url", "application/x-xinference-test", ".bin", True),
        ],
    )
    def test_handle_base64_media_uses_mime_suffix(
        self, model, media_key, mime_type, expected_suffix, expects_file_uri
    ):
        media_data = f"{media_key}-{mime_type}".encode()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": media_key,
                        media_key: {
                            "url": f"data:{mime_type};base64,"
                            + base64.b64encode(media_data).decode()
                        },
                    }
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            model._handle_base64_media(messages, temp_dir)

            materialized_url = messages[0]["content"][0][media_key]["url"]
            assert materialized_url.startswith("file://") is expects_file_uri
            media_path = Path(materialized_url.removeprefix("file://"))
            assert media_path.is_absolute()
            assert media_path.suffix == expected_suffix
            assert media_path.read_bytes() == media_data

        assert not media_path.exists()

    def test_handle_base64_media_preserves_non_data_urls(self, model):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://example.com/video.mp4"},
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "data:audio/wav;base64,YXVkaW8="},
                    },
                ],
            }
        ]
        expected = [dict(item) for item in messages[0]["content"]]

        with tempfile.TemporaryDirectory() as temp_dir:
            model._handle_base64_media(messages, temp_dir)
            assert list(Path(temp_dir).iterdir()) == []

        assert messages[0]["content"] == expected

    def test_handle_base64_media_accepts_folded_base64(self, model):
        media_data = b"folded-base64-video"
        encoded = base64.b64encode(media_data).decode()
        folded = "\r\n  ".join(
            encoded[index : index + 4] for index in range(0, len(encoded), 4)
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{folded}"},
                    }
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            model._handle_base64_media(messages, temp_dir)

            video_uri = messages[0]["content"][0]["video_url"]["url"]
            video_path = Path(video_uri.removeprefix("file://"))
            assert video_path.read_bytes() == media_data

    @pytest.mark.parametrize("fps", [0, 0.0, -1.0])
    def test_attach_video_metadata_rejects_non_positive_fps(self, model, fps):
        with pytest.raises(ValueError, match="fps must be greater than 0"):
            model._attach_video_metadata([self.FakeVideo()], [fps])

    @pytest.mark.parametrize(
        "data_uri",
        [
            "data:video/mp4;base64,%%%not-base64%%%",
            "data:video/mp4,dm lkZW8=",
            "data:video/mp4;base64,",
        ],
    )
    def test_handle_base64_media_rejects_invalid_data(self, model, data_uri):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": data_uri},
                    }
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="base64 data URI for video_url"):
                model._handle_base64_media(messages, temp_dir)
            assert list(Path(temp_dir).iterdir()) == []

    @pytest.mark.asyncio
    async def test_async_chat_cleans_materialized_video_after_processing(
        self, model, monkeypatch
    ):
        video_data = b"mp4-video-content"
        data_uri = "data:video/mp4;base64," + base64.b64encode(video_data).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": data_uri},
                    }
                ],
            }
        ]
        captured = {}

        qwen_omni_utils = types.ModuleType("qwen_omni_utils")

        def process_vision_info(transformed_messages, *, return_video_kwargs):
            assert return_video_kwargs is True
            video_uri = transformed_messages[0]["content"][0]["video"]
            video_path = Path(video_uri.removeprefix("file://"))
            assert video_uri.startswith("file://")
            assert video_path.read_bytes() == video_data
            captured["video_path"] = video_path
            return None, [self.FakeVideo()], {"fps": [2.0]}

        qwen_omni_utils.process_vision_info = process_vision_info
        qwen_omni_utils.process_audio_info = MagicMock()
        qwen_omni_utils.process_mm_info = MagicMock()
        monkeypatch.setitem(sys.modules, "qwen_omni_utils", qwen_omni_utils)

        model.model_uid = "test-model"
        model.model_family = MagicMock()
        model.model_family.model_family = "qwen2-vl"
        model.model_family.model_name = "qwen2-vl"
        model.model_family.model_ability = ["chat", "vision"]
        model.model_family.chat_template = "chat-template"
        model.reasoning_parser = None
        model._get_chat_template_kwargs_from_generate_config = MagicMock(
            return_value={}
        )

        async def get_chat_template_and_tokenizer(_model_family):
            return "chat-template", None

        model._get_chat_template_and_tokenizer = get_chat_template_and_tokenizer
        model.get_full_context = MagicMock(return_value="prompt")
        model._sanitize_chat_config = MagicMock(return_value={"stream": False})

        async def async_generate(inputs, *_args, **_kwargs):
            assert not captured["video_path"].exists()
            [(video, metadata)] = inputs["multi_modal_data"]["video"]
            assert isinstance(video, self.FakeVideo)
            assert metadata == {
                "total_num_frames": 4,
                "frames_indices": [0, 1, 2, 3],
                "fps": 2.0,
                "duration": 2.0,
                "height": 336,
                "width": 336,
            }
            return {"completion": "ok"}

        model.async_generate = async_generate
        model._to_chat_completion = MagicMock(return_value={"result": "ok"})

        result = await model.async_chat(messages, {})

        assert result == {"result": "ok"}
        assert messages[0]["content"][0]["video_url"]["url"] == data_uri
        assert not captured["video_path"].exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("url", ["/etc/hostname", "file:///etc/hostname"])
    async def test_async_chat_rejects_local_media_url(self, model, monkeypatch, url):
        qwen_omni_utils = types.ModuleType("qwen_omni_utils")
        qwen_omni_utils.process_vision_info = MagicMock()
        qwen_omni_utils.process_audio_info = MagicMock()
        qwen_omni_utils.process_mm_info = MagicMock()
        monkeypatch.setitem(sys.modules, "qwen_omni_utils", qwen_omni_utils)

        model.model_uid = "test-model"
        model.model_family = MagicMock()
        model.model_family.model_family = "qwen2-vl"
        model.model_family.model_ability = ["chat", "vision"]

        messages = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": url}}],
            }
        ]
        with pytest.raises(ValueError, match="Local file"):
            await model.async_chat(messages, {})
        qwen_omni_utils.process_vision_info.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("abilities", "media", "reader_error"),
        [
            ("audio vision", "image", None),
            ("vision", "image", None),
            ("audio", "audio", None),
            ("omni audio vision", "audio image", None),
            ("audio vision", "audio image", None),
            ("audio vision", "video", None),
            ("audio vision", "audio image", "audio"),
            ("audio vision", "audio image", "vision"),
        ],
    )
    async def test_async_chat_runs_media_reader_off_event_loop(
        self, model, monkeypatch, abilities, media, reader_error
    ):
        import asyncio

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": f"{kind}_url",
                        f"{kind}_url": {"url": f"https://8.8.8.8/{kind}"},
                    }
                    for kind in media.split()
                ],
            }
        ]
        original = copy.deepcopy(messages)
        paths = set()
        calls = []
        monkeypatch.setattr(
            "xinference.model.llm.media.fetch_media", lambda *_: b"media"
        )
        audios = ["audio"] if "audio" in media else None
        images = ["image"] if "image" in media else None
        videos = [self.FakeVideo()] if "video" in media else None

        def reader(kind, result, **expected_kwargs):
            def process(transformed_messages, **kwargs):
                with pytest.raises(RuntimeError, match="no running event loop"):
                    asyncio.get_running_loop()
                assert kwargs == expected_kwargs
                calls.append(kind)
                for part in transformed_messages[0]["content"]:
                    path = Path(part[part["type"]].removeprefix("file://"))
                    assert path.read_bytes() == b"media"
                    paths.add(path)
                if reader_error == kind:
                    raise ValueError("reader failed")
                return result

            return process

        qwen_omni_utils = types.ModuleType("qwen_omni_utils")
        qwen_omni_utils.process_vision_info = reader(
            "vision", (images, videos, None), return_video_kwargs=True
        )
        qwen_omni_utils.process_audio_info = reader(
            "audio", audios, use_audio_in_video=False
        )
        qwen_omni_utils.process_mm_info = reader(
            "omni",
            (audios, images, videos, None),
            use_audio_in_video=True,
            return_video_kwargs=True,
        )
        monkeypatch.setitem(sys.modules, "qwen_omni_utils", qwen_omni_utils)

        model.model_uid = "test-model"
        model.model_family = MagicMock()
        model.model_family.model_family = "qwen2-vl"
        model.model_family.model_name = "qwen2-vl"
        model.model_family.model_ability = ["chat", *abilities.split()]
        model.model_family.has_architecture.return_value = False
        model.model_family.chat_template = "chat-template"
        model.reasoning_parser = None
        model._get_chat_template_kwargs_from_generate_config = MagicMock(
            return_value={}
        )

        async def get_chat_template_and_tokenizer(_model_family):
            return "chat-template", None

        model._get_chat_template_and_tokenizer = get_chat_template_and_tokenizer
        model.get_full_context = MagicMock(return_value="prompt")
        model._sanitize_chat_config = MagicMock(return_value={"stream": False})

        async def async_generate(inputs, *_args, **_kwargs):
            expected = {kind: [kind] for kind in media.split()}
            actual = dict(inputs["multi_modal_data"])
            if videos:
                assert actual.pop("video")[0][0] is videos[0]
                expected.pop("video")
            assert actual == expected
            assert inputs.get("mm_processor_kwargs", {}) == (
                {"use_audio_in_video": True} if "omni" in abilities else {}
            )
            return {"completion": "ok"}

        model.async_generate = async_generate
        model._to_chat_completion = MagicMock(return_value={"result": "ok"})

        expectation = (
            pytest.raises(ValueError, match="reader failed")
            if reader_error
            else nullcontext()
        )
        with expectation:
            assert await model.async_chat(messages, {}) == {"result": "ok"}
        expected_calls = ["omni"] if "omni" in abilities else abilities.split()
        if reader_error:
            expected_calls = expected_calls[: expected_calls.index(reader_error) + 1]
        assert calls == expected_calls
        assert messages == original
        assert paths and all(
            not path.exists() and not path.parent.exists() for path in paths
        )


class TestVLLMChatModel:

    @pytest.fixture
    def real_vllm_chat_model(self):
        from ..core import VLLMChatModel

        model = object.__new__(VLLMChatModel)

        model.model_family = MagicMock()
        model.model_family.model_family = "qwen"
        model.model_family.reasoning_start_tag = "<think>"
        model.model_family.reasoning_end_tag = "</think>"
        model.model_uid = "test-model-0"
        model.reasoning_parser = None
        model.tool_parser = QwenToolParser()

        return model

    async def create_mock_chunks(self, chunks_data):
        for chunk in chunks_data:
            yield chunk

    @pytest.fixture
    def gpt_oss_chat_model(self, real_vllm_chat_model):
        model = real_vllm_chat_model
        model.model_family = types.SimpleNamespace(
            model_family=None,
            model_name="gpt-oss",
            has_architecture=lambda _: False,
        )
        return model

    @pytest.mark.asyncio
    async def test_async_to_tool_completion_chunks_without_thinking(
        self, real_vllm_chat_model
    ):
        test_chunks = [
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "<tool_call>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 71,
                    "total_tokens": 230,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "\n", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 72,
                    "total_tokens": 231,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '{"', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 73,
                    "total_tokens": 232,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "name",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 74,
                    "total_tokens": 233,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 75,
                    "total_tokens": 234,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 76,
                    "total_tokens": 235,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "get", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 77,
                    "total_tokens": 236,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "_current",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 78,
                    "total_tokens": 237,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "_weather",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 79,
                    "total_tokens": 238,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '",', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 80,
                    "total_tokens": 239,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 81,
                    "total_tokens": 240,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "arguments",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 82,
                    "total_tokens": 241,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 83,
                    "total_tokens": 242,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' {"', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 84,
                    "total_tokens": 243,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "location",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 85,
                    "total_tokens": 244,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 86,
                    "total_tokens": 245,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 87,
                    "total_tokens": 246,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "上海",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 88,
                    "total_tokens": 247,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": '"}}\n',
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 89,
                    "total_tokens": 248,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "</tool_call>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 90,
                    "total_tokens": 249,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 91,
                    "total_tokens": 250,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 91,
                    "total_tokens": 250,
                },
            },
        ]

        chunks_generator = self.create_mock_chunks(test_chunks)
        result_chunks = []
        expected_chunks = [
            {
                "id": "chatcmpl-7fcac134-7380-4a19-b665-d93ffaacfbca",
                "model": "test-model-0",
                "object": "chat.completion.chunk",
                "created": 1756644905,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_7fcac134-7380-4a19-b665-d93ffaacfbca",
                                    "type": "function",
                                    "function": {
                                        "name": "get_current_weather",
                                        "arguments": '{"location": "上海"}',
                                    },
                                }
                            ],
                        },
                        "logprobs": None,
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": -1,
                    "total_tokens": -1,
                },
            },
            {
                "id": "chatcmpl-06a03091-f455-4dfe-a348-2163cf285811",
                "model": "test-model-0",
                "object": "chat.completion.chunk",
                "created": 1756644905,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "", "tool_calls": []},
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "completion_tokens": 91,
                    "prompt_tokens": 159,
                    "total_tokens": 250,
                },
            },
        ]

        i = 0
        async for chunk in real_vllm_chat_model._async_to_tool_completion_chunks(
            chunks_generator
        ):
            result = filter_ids_and_created(chunk)
            expected_result = filter_ids_and_created(expected_chunks[i])
            assert result == expected_result
            result_chunks.append(chunk)
            i += 1

    @pytest.mark.asyncio
    async def test_async_to_tool_completion_chunks_with_thinking(
        self, real_vllm_chat_model
    ):
        test_chunks = [
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451239,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "<think>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 1,
                    "total_tokens": 160,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451239,
                "model": "qwen3",
                "choices": [
                    {"text": "\n", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 2,
                    "total_tokens": 161,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451239,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "好的",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 3,
                    "total_tokens": 162,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "</think>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 69,
                    "total_tokens": 228,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "\n\n",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 70,
                    "total_tokens": 229,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "<tool_call>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 71,
                    "total_tokens": 230,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "\n", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 72,
                    "total_tokens": 231,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '{"', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 73,
                    "total_tokens": 232,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "name",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 74,
                    "total_tokens": 233,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 75,
                    "total_tokens": 234,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 76,
                    "total_tokens": 235,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "get", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 77,
                    "total_tokens": 236,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "_current",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 78,
                    "total_tokens": 237,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "_weather",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 79,
                    "total_tokens": 238,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '",', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 80,
                    "total_tokens": 239,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 81,
                    "total_tokens": 240,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "arguments",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 82,
                    "total_tokens": 241,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 83,
                    "total_tokens": 242,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' {"', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 84,
                    "total_tokens": 243,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "location",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 85,
                    "total_tokens": 244,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 86,
                    "total_tokens": 245,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 87,
                    "total_tokens": 246,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "上海",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 88,
                    "total_tokens": 247,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": '"}}\n',
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 89,
                    "total_tokens": 248,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "</tool_call>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 90,
                    "total_tokens": 249,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 91,
                    "total_tokens": 250,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 91,
                    "total_tokens": 250,
                },
            },
        ]

        chunks_generator = self.create_mock_chunks(test_chunks)
        result_chunks = []

        gen = real_vllm_chat_model._async_to_tool_completion_chunks(chunks_generator)

        async for chunk in gen:
            result_chunks.append(chunk)

    @pytest.mark.asyncio
    async def test_async_to_tool_completion_chunks_with_parser(
        self, real_vllm_chat_model
    ):
        test_chunks = [
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451239,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "<think>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 1,
                    "total_tokens": 160,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451239,
                "model": "qwen3",
                "choices": [
                    {"text": "\n", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 2,
                    "total_tokens": 161,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451239,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "好的",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 3,
                    "total_tokens": 162,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "</think>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 69,
                    "total_tokens": 228,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "\n\n",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 70,
                    "total_tokens": 229,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "<tool_call>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 71,
                    "total_tokens": 230,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "\n", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 72,
                    "total_tokens": 231,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '{"', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 73,
                    "total_tokens": 232,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "name",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 74,
                    "total_tokens": 233,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 75,
                    "total_tokens": 234,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 76,
                    "total_tokens": 235,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "get", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 77,
                    "total_tokens": 236,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "_current",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 78,
                    "total_tokens": 237,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "_weather",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 79,
                    "total_tokens": 238,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '",', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 80,
                    "total_tokens": 239,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 81,
                    "total_tokens": 240,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "arguments",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 82,
                    "total_tokens": 241,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 83,
                    "total_tokens": 242,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' {"', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 84,
                    "total_tokens": 243,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "location",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 85,
                    "total_tokens": 244,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": '":', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 86,
                    "total_tokens": 245,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": ' "', "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 87,
                    "total_tokens": 246,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "上海",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 88,
                    "total_tokens": 247,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": '"}}\n',
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 89,
                    "total_tokens": 248,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {
                        "text": "</tool_call>",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 90,
                    "total_tokens": 249,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "", "index": 0, "logprobs": None, "finish_reason": None}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 91,
                    "total_tokens": 250,
                },
            },
            {
                "id": "cd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "object": "text_completion",
                "created": 1756451240,
                "model": "qwen3",
                "choices": [
                    {"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 91,
                    "total_tokens": 250,
                },
            },
        ]

        chunks_generator = self.create_mock_chunks(test_chunks)
        result_chunks = []
        expected_chunks = [
            {
                "id": "chatcd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "model": "qwen3",
                "created": 1756451239,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "", "content": None},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "model": "qwen3",
                "created": 1756451239,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "\n", "content": None},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "model": "qwen3",
                "created": 1756451239,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "好的", "content": None},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcd40cd70-84a6-11f0-b7a4-bc2411fe6c28",
                "model": "qwen3",
                "created": 1756451240,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "", "content": None},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": "chatcmpl-e3ec64af-ed8f-4706-9544-4f8d7b42c85b",
                "model": "test-model-0",
                "object": "chat.completion.chunk",
                "created": 1756646208,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "\n\n",
                            "tool_calls": [],
                        },
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": -1,
                    "total_tokens": -1,
                },
            },
            {
                "id": "chatcmpl-490011af-9e50-4dea-969b-f10828d5a5ea",
                "model": "test-model-0",
                "object": "chat.completion.chunk",
                "created": 1756646208,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_490011af-9e50-4dea-969b-f10828d5a5ea",
                                    "type": "function",
                                    "function": {
                                        "name": "get_current_weather",
                                        "arguments": '{"location": "上海"}',
                                    },
                                }
                            ],
                        },
                        "logprobs": None,
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": -1,
                    "total_tokens": -1,
                },
            },
            {
                "id": "chatcmpl-b5a05647-d043-43bb-a7e6-58907e7f4288",
                "model": "test-model-0",
                "object": "chat.completion.chunk",
                "created": 1756646208,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "", "tool_calls": []},
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 159,
                    "completion_tokens": 91,
                    "total_tokens": 250,
                },
            },
        ]
        real_vllm_chat_model.prepare_parse_reasoning_content(True, enable_thinking=True)

        i = 0
        async for chunk in real_vllm_chat_model._async_to_tool_completion_chunks(
            chunks_generator
        ):
            result_chunks.append(chunk)
            result = filter_ids_and_created(chunk)
            expected_result = filter_ids_and_created(expected_chunks[i])
            assert result == expected_result
            i = i + 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("reasoning_content", [False, True])
    async def test_gpt_oss_async_chat_parses_final_channel(
        self, gpt_oss_chat_model, stream, reasoning_content
    ):
        model = gpt_oss_chat_model
        token_groups = [
            [
                "<|channel|>",
                "analysis",
                "<|message|>",
                "The user asks: calculate 17*23.",
            ],
            [
                "<|end|>",
                "<|start|>",
                "assistant",
                "<|channel|>",
                "final",
                "<|message|>",
                "39",
            ],
            ["1", "<|end|>"],
        ]
        generated_text = "".join(token for group in token_groups for token in group)

        def logprobs_for(tokens):
            return {
                "text_offset": [0] * len(tokens),
                "tokens": tokens,
                "token_logprobs": [-0.1] * len(tokens),
                "top_logprobs": [{token: -0.1} for token in tokens],
            }

        request_config = {
            "stream": stream,
            "logprobs": True,
            "chat_template_kwargs": {"reasoning_content": reasoning_content},
        }

        async def prepare_chat_inputs(messages, config):
            chat_template_kwargs = config.pop("chat_template_kwargs", {})
            tools = config.pop("tools", None)
            return "prompt", tools, chat_template_kwargs, config

        model._prepare_chat_inputs = AsyncMock(side_effect=prepare_chat_inputs)

        completion = {
            "id": "gpt-oss-test",
            "object": "text_completion",
            "created": 1,
            "model": "gpt-oss",
            "choices": [
                {
                    "index": 0,
                    "text": generated_text,
                    "finish_reason": "stop",
                    "logprobs": logprobs_for(
                        [token for group in token_groups for token in group]
                    ),
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7},
        }

        async def generated_chunks():
            for tokens in token_groups:
                text = "".join(tokens)
                logprobs = logprobs_for(tokens)
                yield {
                    "id": "gpt-oss-test",
                    "object": "text_completion",
                    "created": 1,
                    "model": "gpt-oss",
                    "choices": [
                        {
                            "index": 0,
                            "text": text,
                            "finish_reason": None,
                            "logprobs": logprobs,
                        }
                    ],
                }
            yield {
                "id": "gpt-oss-test",
                "object": "text_completion",
                "created": 1,
                "model": "gpt-oss",
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
            }
            yield {
                "id": "gpt-oss-test",
                "object": "text_completion",
                "created": 1,
                "model": "gpt-oss",
                "choices": [],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 5,
                    "total_tokens": 7,
                },
            }

        async def async_generate(prompt, config, tools, request_id=None):
            assert prompt == "prompt"
            assert not tools
            return generated_chunks() if stream else completion

        model.async_generate = async_generate
        result = await model.async_chat(
            [{"role": "user", "content": "What is 17*23?"}], request_config
        )

        if stream:
            chunks = [chunk async for chunk in result]
            choices = [chunk["choices"][0] for chunk in chunks if chunk["choices"]]
            content = "".join(
                choice["delta"].get("content") or "" for choice in choices
            )
            reasoning = "".join(
                choice["delta"].get("reasoning_content") or "" for choice in choices
            )
            assert choices[-1]["finish_reason"] == "stop"
            assert chunks[-1]["choices"] == []
            assert chunks[-1]["usage"]["total_tokens"] == 7
            if not reasoning_content:
                assert all("reasoning_content" not in c["delta"] for c in choices)
            entries = [
                entry
                for choice in choices
                for entry in (choice.get("logprobs") or {}).get("content", [])
            ]
        else:
            message = result["choices"][0]["message"]
            content = message["content"]
            reasoning = message.get("reasoning_content") or ""
            assert result["choices"][0]["finish_reason"] == "stop"
            if not reasoning_content:
                assert "reasoning_content" not in message
            entries = result["choices"][0]["logprobs"]["content"]

        assert content == "391"
        assert [entry["token"] for entry in entries] == ["39", "1"]
        assert all(entry["logprob"] == -0.1 for entry in entries)
        assert ("calculate 17*23" in reasoning) is reasoning_content

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        "parts,expected",
        [
            (("analysis", "\n", '{"answer":', "391}"), {"answer": 391}),
            (
                ('{"answer":"', "analysis", ' complete"}'),
                {"answer": "analysis complete"},
            ),
        ],
    )
    async def test_gpt_oss_async_chat_keeps_constrained_json_in_content(
        self, gpt_oss_chat_model, stream, parts, expected
    ):
        model = gpt_oss_chat_model
        generated_text = "".join(parts)
        request_config = {"stream": stream, "guided_json_object": True}

        async def prepare_chat_inputs(messages, config):
            chat_template_kwargs = config.pop("chat_template_kwargs", {})
            tools = config.pop("tools", None)
            return "prompt", tools, chat_template_kwargs, config

        model._prepare_chat_inputs = AsyncMock(side_effect=prepare_chat_inputs)

        completion = {
            "id": "gpt-oss-json-test",
            "object": "text_completion",
            "created": 1,
            "model": "gpt-oss",
            "choices": [
                {
                    "index": 0,
                    "text": generated_text,
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {},
        }

        async def generated_chunks():
            for text in parts:
                yield {
                    "id": "gpt-oss-json-test",
                    "object": "text_completion",
                    "created": 1,
                    "model": "gpt-oss",
                    "choices": [
                        {
                            "index": 0,
                            "text": text,
                            "finish_reason": None,
                            "logprobs": None,
                        }
                    ],
                }
            yield {
                "id": "gpt-oss-json-test",
                "object": "text_completion",
                "created": 1,
                "model": "gpt-oss",
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
            }

        async def async_generate(prompt, config, tools, request_id=None):
            assert prompt == "prompt"
            assert config["guided_json_object"] is True
            assert not tools
            return generated_chunks() if stream else completion

        model.async_generate = async_generate
        result = await model.async_chat(
            [{"role": "user", "content": "Return the answer as JSON."}],
            request_config,
        )

        if stream:
            chunks = [chunk async for chunk in result]
            content = "".join(
                chunk["choices"][0]["delta"].get("content") or ""
                for chunk in chunks
                if chunk["choices"]
            )
            assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        else:
            content = result["choices"][0]["message"]["content"]

        assert json.loads(content) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("model_class", ["VLLMChatModel", "VLLMMultiModel"])
@pytest.mark.parametrize("tool_choice", ["none", "auto", None, "absent"])
@pytest.mark.parametrize("tools_state", ["populated", "empty", "absent"])
@pytest.mark.parametrize("stream", [False, True])
async def test_async_chat_tool_choice_rendering(
    monkeypatch, model_class, tool_choice, tools_state, stream
):
    from .. import core

    model = object.__new__(getattr(core, model_class))
    family = json.loads((Path(__file__).parents[2] / "models/qwen3.json").read_text())[
        0
    ]
    model.model_family = types.SimpleNamespace(
        **family, model_family=None, has_architecture=lambda _: False
    )
    model.model_uid = "test-model"
    model.reasoning_parser = None
    model.tool_parser = QwenToolParser()
    model._get_tokenizer = AsyncMock(return_value=None)
    # Run text-only preparation inline; no background media I/O is needed.
    monkeypatch.setattr(
        core.asyncio,
        "to_thread",
        AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw)),
    )
    # Text-only requests still import the optional multimodal readers.
    monkeypatch.setitem(
        sys.modules,
        "qwen_omni_utils",
        types.SimpleNamespace(
            process_audio_info=None, process_mm_info=None, process_vision_info=None
        ),
    )
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]
    config = {"stream": stream, "stop_token_ids": [151645]}
    if tool_choice != "absent":
        config["tool_choice"] = tool_choice
    if tools_state != "absent":
        config["tools"] = tools if tools_state == "populated" else []
    original = copy.deepcopy(config)
    enabled = tools_state == "populated" and tool_choice != "none"

    async def generate(prompt, generate_config, tools, request_id=None):
        rendered = prompt["prompt"] if isinstance(prompt, dict) else prompt
        assert ("get_weather" in rendered) is enabled
        assert bool(tools) is enabled
        assert "tool_choice" not in generate_config
        completion = {
            "id": "test",
            "model": model.model_uid,
            "created": 1,
            "choices": [{"index": 0, "text": "Hello", "finish_reason": "stop"}],
            "usage": {},
        }

        async def chunks():
            yield completion

        return chunks() if stream else completion

    model.async_generate = generate
    result = await model.async_chat([{"role": "user", "content": "Hi"}], config)
    if stream:
        chunks = [chunk async for chunk in result]
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    else:
        assert result["choices"][0]["message"]["content"] == "Hello"
    assert config == original


class TestVLLMSanitizeGenerateConfig:
    @pytest.fixture(autouse=True)
    def vllm_version(self, monkeypatch):
        from .. import core

        monkeypatch.setattr(core, "VLLM_VERSION", version.parse("0.19.0"))

    """Guided-decoding extraction from response_format=json_schema.

    The OpenAI-compatible request body serializes the JSONSchema model without
    by_alias, so the real key reaching the engine is the field name ``schema_``
    (``schema`` is a reserved Pydantic name, aliased in _compat.JSONSchema).
    _sanitize_generate_config must read ``schema_`` (with a ``schema`` fallback
    for raw passthrough), matching sglang/core.py and llm/utils.py.
    """

    _SCHEMA = {
        "type": "object",
        "properties": {"score": {"type": "integer"}},
        "required": ["score"],
    }

    def _sanitize(self, response_format):
        from ..core import VLLMChatModel

        return VLLMChatModel._sanitize_generate_config(
            {"response_format": response_format}
        )

    def test_json_schema_serialized_key(self):
        # Real serialized shape (field name, not alias).
        sanitized = self._sanitize(
            {
                "type": "json_schema",
                "json_schema": {"name": "judge", "schema_": self._SCHEMA},
            }
        )
        assert sanitized["guided_json"] == self._SCHEMA

    def test_json_schema_alias_fallback(self):
        # Raw passthrough shape using the alias.
        sanitized = self._sanitize(
            {
                "type": "json_schema",
                "json_schema": {"name": "judge", "schema": self._SCHEMA},
            }
        )
        assert sanitized["guided_json"] == self._SCHEMA

    def test_json_object_unaffected(self):
        sanitized = self._sanitize({"type": "json_object"})
        assert sanitized["guided_json"] is None
        assert sanitized["guided_json_object"] is True

    def test_json_schema_empty_schema_preserved(self):
        # An empty dict is a valid schema; `is None` (not truthiness) must keep
        # it instead of falling through to the absent `schema` key. The
        # downstream guard in async_generate likewise uses `is not None`, so a
        # preserved {} actually reaches vLLM guided decoding rather than being
        # dropped. This asserts the intermediate value; the async_generate path
        # depends on vLLM and is not unit-tested here.
        sanitized = self._sanitize(
            {
                "type": "json_schema",
                "json_schema": {"name": "judge", "schema_": {}},
            }
        )
        assert sanitized["guided_json"] == {}

    def test_no_response_format(self):
        from ..core import VLLMChatModel

        sanitized = VLLMChatModel._sanitize_generate_config({})
        assert sanitized["guided_json"] is None


class TestVLLMModelLogprobs:
    """Tests for vLLM logprobs wiring in the completion response path.

    These exercise the function under fix (``_build_logprobs`` and the
    ``_convert_request_output_to_completion`` site that used to hardcode
    ``logprobs=None``) directly, with a mocked vLLM request output. On
    master the response ``logprobs`` is always ``None``; on the branch it is
    populated when the engine produced them.
    """

    @staticmethod
    def _make_logprob(logprob: float, decoded_token: str):
        lp = MagicMock()
        lp.logprob = logprob
        lp.decoded_token = decoded_token
        return lp

    @classmethod
    def _make_request_output(
        cls, *, text, token_ids, logprobs, finish_reason="stop", prompt=""
    ):
        output = MagicMock()
        output.text = text
        output.index = 0
        output.token_ids = list(token_ids)
        output.logprobs = logprobs
        output.finish_reason = finish_reason
        request_output = MagicMock()
        request_output.outputs = [output]
        request_output.prompt = prompt
        request_output.prompt_token_ids = [1, 2, 3]
        return request_output

    def test_build_logprobs_none_when_not_requested(self):
        from ..core import VLLMModel

        request_output = self._make_request_output(
            text=" hello world", token_ids=[10, 11], logprobs=None
        )
        output = request_output.outputs[0]
        assert VLLMModel._build_logprobs(output) is None

    def test_build_logprobs_populated_when_present(self):
        from ..core import VLLMModel

        logprobs = [
            {
                10: self._make_logprob(-0.1, " hello"),
                99: self._make_logprob(-3.2, " hi"),
            },
            {11: self._make_logprob(-0.2, " world")},
        ]
        request_output = self._make_request_output(
            text=" hello world", token_ids=[10, 11], logprobs=logprobs
        )
        result = VLLMModel._build_logprobs(request_output.outputs[0], prompt_offset=4)
        assert result is not None
        assert result["tokens"] == [" hello", " world"]
        assert result["token_logprobs"] == [-0.1, -0.2]
        assert result["top_logprobs"][0][" hello"] == -0.1
        assert result["top_logprobs"][0][" hi"] == -3.2
        assert result["top_logprobs"][1][" world"] == -0.2
        # text_offset is relative to the full prompt + completion text.
        assert result["text_offset"] == [4, 10]

    def test_convert_request_output_to_completion_carries_logprobs(self):
        from ..core import VLLMModel

        logprobs = [{10: self._make_logprob(-0.5, "ab")}]
        request_output = self._make_request_output(
            text="ab",
            token_ids=[10],
            logprobs=logprobs,
            finish_reason="stop",
            prompt="say:",
        )
        completion = VLLMModel._convert_request_output_to_completion(
            request_id="req-1", model="m", request_output=request_output
        )
        choice = completion["choices"][0]
        assert choice["logprobs"] is not None
        assert choice["logprobs"]["tokens"] == ["ab"]
        assert choice["logprobs"]["token_logprobs"] == [-0.5]
        assert choice["logprobs"]["text_offset"] == [4]

    def test_convert_request_output_to_completion_logprobs_none_when_absent(self):
        from ..core import VLLMModel

        request_output = self._make_request_output(
            text="ab", token_ids=[10], logprobs=None
        )
        completion = VLLMModel._convert_request_output_to_completion(
            request_id="req-1", model="m", request_output=request_output
        )
        assert completion["choices"][0]["logprobs"] is None

    def test_convert_request_output_to_completion_chunk_carries_logprobs(self):
        from ..core import VLLMModel

        logprobs = [{10: self._make_logprob(-0.5, "ab")}]
        request_output = self._make_request_output(
            text="ab", token_ids=[10], logprobs=logprobs, finish_reason=None
        )
        chunk, finish_reason = VLLMModel._convert_request_output_to_completion_chunk(
            request_id="req-1", model="m", request_output=request_output
        )
        choice = chunk["choices"][0]
        assert choice["logprobs"] is not None
        assert choice["logprobs"]["tokens"] == ["ab"]
        assert choice["logprobs"]["token_logprobs"] == [-0.5]

    def test_convert_request_output_to_completion_chunk_logprobs_none_when_absent(self):
        from ..core import VLLMModel

        request_output = self._make_request_output(
            text="ab", token_ids=[10], logprobs=None, finish_reason=None
        )
        chunk, _ = VLLMModel._convert_request_output_to_completion_chunk(
            request_id="req-1", model="m", request_output=request_output
        )
        assert chunk["choices"][0]["logprobs"] is None

    def test_build_logprobs_skips_empty_decoded_token(self):
        """A malformed Logprob with no decoded token must not emit a ``str(tid)``
        key -- OpenAI ``top_logprobs`` keys are decoded token strings."""
        from ..core import VLLMModel

        lp_with_text = self._make_logprob(-0.1, "ab")
        lp_no_text = MagicMock()
        lp_no_text.logprob = -2.0
        lp_no_text.decoded_token = None
        logprobs = [{10: lp_with_text, 99: lp_no_text}]
        request_output = self._make_request_output(
            text="ab", token_ids=[10], logprobs=logprobs
        )
        result = VLLMModel._build_logprobs(request_output.outputs[0])
        assert result is not None
        # Malformed alternatives without decoded text are omitted.
        assert "99" not in result["top_logprobs"][0]
        assert "" not in result["top_logprobs"][0]
        assert "ab" in result["top_logprobs"][0]

    def test_build_logprobs_clamps_negative_infinity(self):
        from ..core import VLLMModel

        logprobs = [
            {
                10: self._make_logprob(float("-inf"), "ab"),
                99: self._make_logprob(float("-inf"), "cd"),
            }
        ]
        request_output = self._make_request_output(
            text="ab", token_ids=[10], logprobs=logprobs
        )
        result = VLLMModel._build_logprobs(request_output.outputs[0])
        assert result is not None
        assert result["token_logprobs"] == [-9999.0]
        assert result["top_logprobs"] == [{"ab": -9999.0, "cd": -9999.0}]

    def test_slice_logprobs_returns_only_new_stream_tokens(self):
        from ..core import VLLMModel

        cumulative = VLLMModel._build_logprobs(
            self._make_request_output(
                text="ab",
                token_ids=[10, 11],
                logprobs=[
                    {10: self._make_logprob(-0.1, "a")},
                    {11: self._make_logprob(-0.2, "b")},
                ],
            ).outputs[0],
            prompt_offset=4,
        )
        assert cumulative is not None
        delta = VLLMModel._slice_logprobs(cumulative, 1)
        assert delta == {
            "text_offset": [5],
            "tokens": ["b"],
            "token_logprobs": [-0.2],
            "top_logprobs": [{"b": -0.2}],
        }

    def test_sanitize_translates_logprobs_request_to_vllm_int(self, monkeypatch):
        from .. import core

        monkeypatch.setattr(core, "VLLM_VERSION", version.parse("0.21.0"))

        # Legacy completions pass their integer logprobs value through.
        sanitized = core.VLLMModel._sanitize_generate_config({"logprobs": 5})
        assert sanitized["logprobs"] == 5
        sanitized = core.VLLMModel._sanitize_generate_config({"logprobs": 0})
        assert sanitized["logprobs"] == 0

        # Chat requests use a boolean plus top_logprobs.
        sanitized = core.VLLMModel._sanitize_generate_config(
            {"logprobs": True, "top_logprobs": 5}
        )
        assert sanitized["logprobs"] == 5
        sanitized = core.VLLMModel._sanitize_generate_config({"logprobs": True})
        assert sanitized["logprobs"] == 0

        # vLLM uses None, rather than 0, to disable logprobs.
        sanitized = core.VLLMModel._sanitize_generate_config({"logprobs": False})
        assert sanitized["logprobs"] is None
        sanitized = core.VLLMModel._sanitize_generate_config({})
        assert sanitized["logprobs"] is None

        # prompt_logprobs passes through
        sanitized = core.VLLMModel._sanitize_generate_config({"prompt_logprobs": 3})
        assert sanitized["prompt_logprobs"] == 3
