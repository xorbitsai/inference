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
import logging
import re
import time
import uuid
from typing import Iterator, Optional, Sequence, Tuple

from ....types import CompletionChoice, CompletionChunk, CompletionUsage

logger = logging.getLogger(__name__)


def generate_stream(
    model,
    model_ref,
    prompt: str,
    *,
    max_new_tokens: Optional[int] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    last_n_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
    stream: Optional[bool] = False,
    threads: Optional[int] = None,
    stop: Optional[Sequence[str]] = None,
    reset: Optional[bool] = None,
    **kwargs,
) -> Iterator[Tuple[CompletionChunk, CompletionUsage]]:
    stop = stop or []
    if isinstance(stop, str):
        stop = [stop]

    tokens = model_ref.tokenize(prompt)

    stop_regex = re.compile("|".join(map(re.escape, stop)))
    count = 0
    text = ""
    total_text = ""
    incomplete = b""

    # parameters needed for Xinference.
    finish_reason = None

    try:
        from ctransformers.utils import utf8_split_incomplete
    except ImportError:
        error_message = (
            "Failed to import module 'ctransformers - utf8_split_incomplete'"
        )

        installation_guide = [
            "Please make sure 'ctransformers' is installed. You can install it by checking out the repository: "
            "https://github.com/marella/ctransformers",
        ]

        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    for token in model_ref.generate(
        tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        last_n_tokens=last_n_tokens,
        seed=seed,
        batch_size=batch_size,
        threads=threads,
        reset=reset,
    ):
        # Handle incomplete UTF-8 multi-byte characters.
        incomplete += model_ref.detokenize([token], decode=False)
        complete, incomplete = utf8_split_incomplete(incomplete)
        output = complete.decode(errors="ignore")
        text += output
        total_text += output

        # https://github.com/abetlen/llama-cpp-python/blob/1a13d76c487df1c8560132d10bda62d6e2f4fa93/llama_cpp/llama.py#L686-L706
        # Check if one of the stop sequences is part of the text.
        # Note that the stop sequence may not always be at the end of text.
        if stop:
            match = stop_regex.search(text)
            if match:
                text = text[: match.start()]
                finish_reason = "stop"
                break

        # Avoid sending the longest suffix of text which is also a prefix
        # of a stop sequence, as it can form a stop sequence with the text
        # generated later.
        longest = 0
        for s in stop:
            for i in range(len(s), 0, -1):
                if text.endswith(s[:i]):
                    longest = max(i, longest)
                    break

        end = len(text) - longest
        if end > 0:
            output = text[:end]
            completion_choice = CompletionChoice(
                text=output, index=0, logprobs=None, finish_reason=None
            )
            completion_chunk = CompletionChunk(
                id=str(uuid.uuid1()),
                object="text_completion",
                created=int(time.time()),
                model=model,
                choices=[completion_choice],
            )
            completion_usage = CompletionUsage(
                prompt_tokens=len(tokens),
                completion_tokens=count + 1,
                total_tokens=count + 1 + len(tokens),
            )

            yield completion_chunk, completion_usage
            text = text[end:]

        count += 1
        if max_new_tokens is not None and count >= max_new_tokens:
            finish_reason = "length"
            break

    if stream is False:
        completion_choice = CompletionChoice(
            text=total_text, index=0, logprobs=None, finish_reason=finish_reason
        )
    else:
        completion_choice = CompletionChoice(
            text=text, index=0, logprobs=None, finish_reason=finish_reason
        )

    completion_chunk = CompletionChunk(
        id=str(uuid.uuid1()),
        object="text_completion",
        created=int(time.time()),
        model=model,
        choices=[completion_choice],
    )
    completion_usage = CompletionUsage(
        prompt_tokens=len(tokens),
        completion_tokens=count,
        total_tokens=count + len(tokens),
    )

    yield completion_chunk, completion_usage
