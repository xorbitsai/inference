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

import re

from ..harmony import HarmonyStreamParser, async_stream_harmony_chat_completion


def test_streaming_parser_multiple_texts():
    parser = HarmonyStreamParser()

    texts = [
        'analysisThe user says "你好" (Hello). We need to respond in Chinese presumably. '
        "Should greet back and ask how can help.\n"
        "We should keep it short but friendly.assistantfinal你好！很高兴和你聊聊天。"
        "请问有什么我可以帮忙的？祝你今天愉快！",
        'analysisThe user wrote "你好" (Hello in Chinese). We need to respond appropriately, '
        "presumably in Chinese. The system instructions say we should produce natural, friendly tone, "
        "follow style guidelines.\n\n"
        "We can greet back and ask how we can help. "  # noqa: codespell
        "Possibly ask if they want assistance with something specific. Use Chinese. "
        "Also note use of markdown optionally.\n\n"
        "Will respond with a friendly greeting and ask what they need."
        "assistantfinal你好！有什么我可以帮忙的吗？😊如果有任何问题或需要讨论的话题，尽管告诉我。",
    ]

    all_segments = []
    for text in texts:
        for seg in parser.feed(text):
            all_segments.append(seg)

    # We expect 4 segments total: 2 per text
    assert len(all_segments) == 4

    # Check channels
    assert all_segments[0]["channel"] == "analysis"
    assert all_segments[1]["channel"] == "final"
    assert all_segments[2]["channel"] == "analysis"
    assert all_segments[3]["channel"] == "final"

    # Optionally, check some content keywords
    assert "The user says" in all_segments[0]["content"]
    assert "你好！很高兴" in all_segments[1]["content"]
    assert "The user wrote" in all_segments[2]["content"]
    assert "你好！有什么我可以帮忙" in all_segments[3]["content"]


async def test_harmony_streaming_and_nonstreaming():
    texts = [
        'analysisThe user says "你好" (Hello). We need to respond in Chinese presumably. '
        "Should greet back and ask how can help.\n"
        "We should keep it short but friendly.assistantfinal你好！很高兴和你聊聊天。"
        "请问有什么我可以帮忙的？祝你今天愉快！",
        'analysisThe user wrote "你好" (Hello in Chinese). We need to respond appropriately, '
        "presumably in Chinese. The system instructions say we should produce natural, friendly tone, "
        "follow style guidelines.\n\n"
        "We can greet back and ask how we can help. "
        "Possibly ask if they want assistance with something specific. Use Chinese. "
        "Also note use of markdown optionally.\n\nWill respond with a friendly greeting and ask what they need."
        "assistantfinal你好！有什么我可以帮忙的吗？😊如果有任何问题或需要讨论的话题，尽管告诉我。",
    ]

    ### --- Non-streaming test ---
    full_chat = {
        "id": "full_test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"content": t, "reasoning_content": ""}}
            for t in texts
        ],
        "usage": {},
    }

    parsed_full = []
    async for chunk in async_stream_harmony_chat_completion(full_chat):
        parsed_full.append(chunk)

    assert len(parsed_full) == 1
    for i, choice in enumerate(parsed_full[0]["choices"]):
        msg = choice["message"]
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")
        assert "你好" in content, "Final content missing"
        assert "The user" in reasoning, "Reasoning content missing"
        assert not content.startswith(
            "analysis"
        ), "Final content incorrectly contains analysis prefix"

    ### --- Streaming test (realistic incremental chunks) ---
    import asyncio

    async def async_chunk_generator():
        chunk_size = 15  # simulate token-level streaming

        for t in texts:
            # Split by key prefixes, keep prefixes with content
            segments = re.split(r"(analysis|assistantfinal)", t)
            segs_with_prefix = []

            i = 0
            while i < len(segments):
                if segments[i] in ("analysis", "assistantfinal"):
                    segs_with_prefix.append(segments[i] + segments[i + 1])
                    i += 2
                else:
                    if segments[i]:
                        segs_with_prefix.append(segments[i])
                    i += 1

            # Now cut each segment into smaller chunks
            for seg in segs_with_prefix:
                for j in range(0, len(seg), chunk_size):
                    piece = seg[j : j + chunk_size]
                    yield {
                        "id": "stream_test",
                        "model": "test-model",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None,
                            }
                        ],
                    }
                    await asyncio.sleep(
                        0
                    )  # Small delay to simulate real async behavior

    incremental_results = []
    async for chunk in async_stream_harmony_chat_completion(async_chunk_generator()):
        incremental_results.append(chunk)

    # Streaming test: accumulate manually
    accum_content = ""
    accum_reasoning = ""

    for chunk in incremental_results:
        delta = chunk["choices"][0]["delta"]
        accum_content += delta.get("content", "") or ""
        reasoning_content = delta.get("reasoning_content", "")
        if reasoning_content is not None:
            accum_reasoning += reasoning_content

    # Verify accumulated result
    assert "你好" in accum_content, "Streaming final content missing"
    assert "The user" in accum_reasoning, "Streaming reasoning content missing"
    assert not accum_content.startswith(
        "analysis"
    ), "Streaming content incorrectly contains analysis prefix"


stream_chunks = [
    {
        "id": "chat1",
        "model": "gpt-oss",
        "created": 1755090509,
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chat2",
        "model": "gpt-oss",
        "created": 1755090509,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": "analysis"}, "finish_reason": None}
        ],
    },
    {
        "id": "chat3",
        "model": "gpt-oss",
        "created": 1755090509,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "The"}, "finish_reason": None}],
    },
    {
        "id": "chat4",
        "model": "gpt-oss",
        "created": 1755090509,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": " user says"}, "finish_reason": None}
        ],
    },
    {
        "id": "chat5",
        "model": "gpt-oss",
        "created": 1755090510,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": ' "你好'}, "finish_reason": None}
        ],
    },
    {
        "id": "chat6",
        "model": "gpt-oss",
        "created": 1755090510,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": '" ('}, "finish_reason": None}],
    },
    {
        "id": "chat7",
        "model": "gpt-oss",
        "created": 1755090510,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": "Hello)."}, "finish_reason": None}
        ],
    },
    {
        "id": "chat8",
        "model": "gpt-oss",
        "created": 1755090510,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": " We should"}, "finish_reason": None}
        ],
    },
    {
        "id": "chat9",
        "model": "gpt-oss",
        "created": 1755090510,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": " respond in"}, "finish_reason": None}
        ],
    },
    {
        "id": "chat10",
        "model": "gpt-oss",
        "created": 1755090510,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": " Chinese,"}, "finish_reason": None}
        ],
    },
    {
        "id": "chat11",
        "model": "gpt-oss",
        "created": 1755090510,
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {"content": " friendly greeting"},
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chat12",
        "model": "gpt-oss",
        "created": 1755090511,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": ".\n\n We"}, "finish_reason": None}
        ],
    },
    {
        "id": "chat13",
        "model": "gpt-oss",
        "created": 1755090511,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": " can say"}, "finish_reason": None}
        ],
    },
    {
        "id": "chat14",
        "model": "gpt-oss",
        "created": 1755090511,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": " something like"}, "finish_reason": None}
        ],
    },
    {
        "id": "chat15",
        "model": "gpt-oss",
        "created": 1755090511,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": ' "你好'}, "finish_reason": None}
        ],
    },
    {
        "id": "chat16",
        "model": "gpt-oss",
        "created": 1755090511,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": "！有什么"}, "finish_reason": None}
        ],
    },
    {
        "id": "chat17",
        "model": "gpt-oss",
        "created": 1755090511,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": "可以帮"}, "finish_reason": None}
        ],
    },
    {
        "id": "chat18",
        "model": "gpt-oss",
        "created": 1755090511,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": '您?"'}, "finish_reason": None}],
    },
    {
        "id": "chat19",
        "model": "gpt-oss",
        "created": 1755090511,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": " Keep brief"}, "finish_reason": None}
        ],
    },
    {
        "id": "chat20",
        "model": "gpt-oss",
        "created": 1755090512,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "."}, "finish_reason": None}],
    },
    {
        "id": "chat21",
        "model": "gpt-oss",
        "created": 1755090512,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": "assistant"}, "finish_reason": None}
        ],
    },
    {
        "id": "chat22",
        "model": "gpt-oss",
        "created": 1755090512,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "final"}, "finish_reason": None}],
    },
    {
        "id": "chat23",
        "model": "gpt-oss",
        "created": 1755090512,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "你好"}, "finish_reason": None}],
    },
    {
        "id": "chat24",
        "model": "gpt-oss",
        "created": 1755090512,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "！很"}, "finish_reason": None}],
    },
    {
        "id": "chat25",
        "model": "gpt-oss",
        "created": 1755090512,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "高兴"}, "finish_reason": None}],
    },
    {
        "id": "chat26",
        "model": "gpt-oss",
        "created": 1755090512,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "见到"}, "finish_reason": None}],
    },
    {
        "id": "chat27",
        "model": "gpt-oss",
        "created": 1755090513,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "你。"}, "finish_reason": None}],
    },
    {
        "id": "chat28",
        "model": "gpt-oss",
        "created": 1755090513,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "请问"}, "finish_reason": None}],
    },
    {
        "id": "chat29",
        "model": "gpt-oss",
        "created": 1755090513,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": "有什么我"}, "finish_reason": None}
        ],
    },
    {
        "id": "chat30",
        "model": "gpt-oss",
        "created": 1755090513,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": "可以帮助"}, "finish_reason": None}
        ],
    },
    {
        "id": "chat31",
        "model": "gpt-oss",
        "created": 1755090513,
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"content": "你的？"}, "finish_reason": None}
        ],
    },
    {
        "id": "chat32",
        "model": "gpt-oss",
        "created": 1755090513,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
    },
]


async def test_async_stream_chunks():
    """Test with async generator instead of sync list"""
    import asyncio

    async def async_chunk_generator():
        for chunk in stream_chunks:
            yield chunk
            await asyncio.sleep(0)  # Small delay to simulate real async behavior

    content_accum = ""
    reasoning_accum = ""
    all_chunks = []

    # Test that reasoning_content is None when content has value but reasoning_content is empty
    chunk_with_content_count = 0
    async for chunk in async_stream_harmony_chat_completion(async_chunk_generator()):
        all_chunks.append(chunk)
        delta = chunk["choices"][0].get("delta", {})
        if delta.get("content"):
            chunk_with_content_count += 1
            # 当 content 不为空时，reasoning_content 应该为 None
            assert (
                delta.get("reasoning_content") is None
            ), f"Chunk {chunk_with_content_count}: reasoning_content should be None when content has value"

        if delta.get("reasoning_content"):
            reasoning_accum += delta.get("reasoning_content", "")
        else:
            content_accum += delta.get("content", "")

    # Check that the last chunk has finish_reason: "stop"
    last_chunk = all_chunks[-1]
    last_choice = last_chunk["choices"][0]
    assert (
        last_choice.get("finish_reason") == "stop"
    ), f"Last chunk should have finish_reason 'stop', got {last_choice.get('finish_reason')}"

    expected_content = "你好！很高兴见到你。请问有什么我可以帮助你的？"
    expected_reasoning = (
        'The user says "你好" (Hello). We should respond in Chinese, friendly greeting.\n\n'
        ' We can say something like "你好！有什么可以帮您?" Keep brief.'
    )

    assert content_accum == expected_content
    assert reasoning_accum == expected_reasoning
    assert chunk_with_content_count > 0, "Should have chunks with content"
