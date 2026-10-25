# Copyright 2022-2026 XProbe Inc.
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

import math
import re
from typing import List, Tuple

import torch
from xoscar.utils import lazy_import

diffusers = lazy_import("diffusers")

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text):
    r"""
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


class PromptChunk:
    """
    This object contains token ids, weight (multipliers:1.4) and textual inversion embedding info for a chunk of prompt.
    If a prompt is short, it is represented by one PromptChunk, otherwise, multiple are necessary.
    Each PromptChunk contains an exact amount of tokens - 77, which includes one for start and end token,
    so just 75 tokens from prompt.
    """

    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []


class TextConditionalModel(torch.nn.Module):
    def __init__(
        self, model: "diffusers.DiffusionPipeline", return_pooled: bool = False  # type: ignore
    ):
        super().__init__()

        self.model = model

        self.chunk_length = model.tokenizer.model_max_length - 2

        self.is_trainable = False
        self.input_key = "txt"
        self.return_pooled = return_pooled

        self.comma_token = None
        self.id_start = None
        self.id_end = None
        self.id_pad = None

    def empty_chunk(self) -> PromptChunk:
        """creates an empty PromptChunk and returns it"""

        chunk = PromptChunk()
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        chunk.multipliers = [1.0] * (self.chunk_length + 2)
        return chunk

    def get_target_prompt_token_count(self, token_count: int) -> int:
        """returns the maximum number of tokens a prompt of a known length can have before it requires one more PromptChunk to be represented"""

        return math.ceil(max(token_count, 1) / self.chunk_length) * self.chunk_length

    def tokenize(self, texts: List[str]):
        """Converts a batch of texts into a batch of token ids"""

        raise NotImplementedError

    def encode_with_transformers(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        converts a batch of token ids (in python lists) into a single tensor with numeric representation of those tokens;
        All python lists with tokens are assumed to have same length, usually 77.
        if input is a list with B elements and each element has T tokens, expected output shape is (B, T, C), where C depends on
        model - can be 768 and 1024.
        """

        output = self.model.text_encoder(input_ids=tokens, output_hidden_states=True)
        clip_skip = getattr(self.model, "clip_skip", 1)
        if getattr(self.model, "is_sdxl", False):
            hidden = output.hidden_states[-(clip_skip + 1)]
        elif clip_skip > 1:
            hidden = self.model.text_encoder.text_model.final_layer_norm(
                output.hidden_states[-clip_skip]
            )
        else:
            hidden = output.last_hidden_state
        if self.return_pooled:
            hidden.pooled = output.text_embeds
        return hidden

    def tokenize_line(self, line: str) -> Tuple[List[str], int]:
        """
        this transforms a single prompt into a list of PromptChunk objects - as many as needed to
        represent the prompt.
        Returns the list and the total number of tokens in the prompt.
        """

        parsed = parse_prompt_attention(line)

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            """puts current chunk into the list of results and produces the next one - empty;
            if is_last is true, tokens <end-of-text> tokens at the end won't add to token_count
            """
            nonlocal token_count
            nonlocal last_comma
            nonlocal chunk

            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += self.chunk_length

            to_add = self.chunk_length - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [self.id_end] * to_add
                chunk.multipliers += [1.0] * to_add

            chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

            last_comma = -1
            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == "BREAK" and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]

                if token == self.comma_token:
                    last_comma = len(chunk.tokens)

                # this is when we are at the end of allotted 75 tokens for the current chunk, and the current token is not a comma. opts.comma_padding_backtrack
                # is a setting that specifies that if there is a comma nearby, the text after the comma should be moved out of this chunk and into the next.
                # opts.comma_padding_backtrack is 20 by default
                elif (
                    len(chunk.tokens) == self.chunk_length
                    and last_comma != -1
                    and len(chunk.tokens) - last_comma <= 20
                ):
                    break_location = last_comma + 1

                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]

                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]

                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults

                if len(chunk.tokens) == self.chunk_length:
                    next_chunk()

                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1
                continue

        if chunk.tokens or not chunks:
            next_chunk(is_last=True)

        return chunks, token_count

    def process_texts(self, texts: List[str]) -> Tuple[List[List[str]], int]:
        """
        Accepts a list of texts and calls tokenize_line() on each, with cache. Returns the list of results and maximum
        length, in tokens, of all texts.
        """

        token_count = 0

        cache = {}  # type: ignore
        batch_chunks = []
        for line in texts:
            if line in cache:
                chunks = cache[line]
            else:
                chunks, current_token_count = self.tokenize_line(line)
                token_count = max(current_token_count, token_count)

                cache[line] = chunks

            batch_chunks.append(chunks)

        return batch_chunks, token_count

    def forward(self, texts: List[str]):
        """
        Accepts an array of texts; Passes texts through transformers network to create a tensor with numerical representation of those texts.
        Returns a tensor with shape of (B, T, C), where B is length of the array; T is length, in tokens, of texts (including padding) - T will
        be a multiple of 77; and C is dimensionality of each token - for SD1 it's 768, for SD2 it's 1024, and for SDXL it's 1280.
        An example shape returned by this function can be: (2, 77, 768).
        For SDXL, instead of returning one tensor avobe, it returns a tuple with two: the other one with shape (B, 1280) with pooled values.
        Webui usually sends just one text at a time through this function - the only time when texts is an array with more than one element
        is when you do prompt editing: "a picture of a [cat:dog:0.4] eating ice cream"
        """

        batch_chunks, token_count = self.process_texts(texts)

        chunk_count = max([len(x) for x in batch_chunks])

        zs = []
        for i in range(chunk_count):
            batch_chunk = [
                chunks[i] if i < len(chunks) else self.empty_chunk()
                for chunks in batch_chunks
            ]

            tokens = [x.tokens for x in batch_chunk]  # type: ignore
            multipliers = [x.multipliers for x in batch_chunk]  # type: ignore
            z = self.process_tokens(tokens, multipliers)
            zs.append(z)

        if self.return_pooled:
            return torch.hstack(zs), zs[0].pooled
        else:
            return torch.hstack(zs)

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        """
        sends one single prompt chunk to be encoded by transformers neural network.
        remade_batch_tokens is a batch of tokens - a list, where every element is a list of tokens; usually
        there are exactly 77 tokens in the list. batch_multipliers is the same but for multipliers instead of tokens.
        Multipliers are used to give more or less weight to the outputs of transformers network. Each multiplier
        corresponds to one token.
        """
        tokens = torch.asarray(remade_batch_tokens).to(self.model.text_encoder.device)

        # this is for SD2: SD1 uses the same token for padding and end of text, while SD2 uses different ones.
        if self.id_end != self.id_pad:
            for batch_pos in range(len(remade_batch_tokens)):
                index = remade_batch_tokens[batch_pos].index(self.id_end)
                tokens[batch_pos, index + 1 : tokens.shape[1]] = self.id_pad

        z = self.encode_with_transformers(tokens)

        pooled = getattr(z, "pooled", None)

        multipliers = torch.asarray(batch_multipliers, device=z.device, dtype=z.dtype)
        original_mean = z.mean()
        z = z * multipliers.reshape(multipliers.shape + (1,)).expand(z.shape)
        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        new_mean = z.mean()
        z = z * torch.where(
            new_mean.abs() > 1e-6, original_mean / new_mean, torch.ones_like(new_mean)
        )

        if pooled is not None:
            z.pooled = pooled

        return z


class FrozenCLIPEmbedderWithCustomWords(TextConditionalModel):
    def __init__(
        self, model: "diffusers.DiffusionPipeline", return_pooled: bool = False  # type: ignore
    ):
        super().__init__(model, return_pooled=return_pooled)
        self.model = model
        self.tokenizer = model.tokenizer

        vocab = self.tokenizer.get_vocab()

        self.comma_token = vocab.get(",</w>", None)

        self.token_mults = {}
        tokens_with_parens = [
            (k, v)
            for k, v in vocab.items()
            if "(" in k or ")" in k or "[" in k or "]" in k
        ]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == "[":
                    mult /= 1.1
                if c == "]":
                    mult *= 1.1
                if c == "(":
                    mult *= 1.1
                if c == ")":
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

        self.id_start = model.tokenizer.bos_token_id
        self.id_end = model.tokenizer.eos_token_id
        self.id_pad = self.id_end

    def tokenize(self, texts: List[str]):
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=False)[
            "input_ids"
        ]

        return tokenized


def prompt_to_prompt_embeds(texts, model_base, model, clip_skip=1):
    from types import SimpleNamespace

    if model_base not in ("SD 1.5", "SD 2.0", "SD 2.1", "SDXL"):
        raise ValueError(f"Weighted prompts are unsupported for {model_base}")
    is_sdxl = model_base == "SDXL"
    pairs = [(model.tokenizer, model.text_encoder)]
    if is_sdxl:
        pairs.append((model.tokenizer_2, model.text_encoder_2))
    embeddings, empty_embeddings, pooled = [], [], None
    for index, (tokenizer, text_encoder) in enumerate(pairs):
        proxy = SimpleNamespace(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            clip_skip=clip_skip,
            is_sdxl=is_sdxl,
        )
        encoder = FrozenCLIPEmbedderWithCustomWords(
            proxy, return_pooled=is_sdxl and index == 1
        )
        encoded = encoder(texts)
        empty = encoder([""] * len(texts))
        if isinstance(encoded, tuple):
            encoded, pooled = encoded
            empty = empty[0]
        embeddings.append(encoded)
        empty_embeddings.append(empty)
    maximum = max(value.shape[1] for value in embeddings)
    for index, value in enumerate(embeddings):
        while value.shape[1] < maximum:
            value = torch.cat((value, empty_embeddings[index]), dim=1)
        embeddings[index] = value
    combined = torch.cat(embeddings, dim=-1)
    return (combined, pooled) if is_sdxl else combined


@torch.no_grad()
def gen_prompt_embeds(kwargs: dict, model_base: str, model):
    prompt = kwargs.pop("prompt", "")
    negative_prompt = kwargs.pop("negative_prompt", "")
    clip_skip = kwargs.pop("clip_skip", 1) or 1
    encoded = prompt_to_prompt_embeds(
        [prompt, negative_prompt], model_base, model, clip_skip
    )
    if isinstance(encoded, tuple):
        encoded, pooled = encoded
        kwargs["pooled_prompt_embeds"] = pooled[:1]
        kwargs["negative_pooled_prompt_embeds"] = pooled[1:]
    kwargs["prompt_embeds"] = encoded[:1]
    kwargs["negative_prompt_embeds"] = encoded[1:]
