# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

import re

import torch
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother

DEFAULT_SPEECH_TOKEN = "<speech>"
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class LlmTokenizerWrapper:
    @classmethod
    def build_llm_tokenizer(cls, llm_path, use_flash_attn=False):
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
        if use_flash_attn:
            tokenizer.padding_side = "left"
        else:
            tokenizer.padding_side = "right"
        special_tokens_dict = {"additional_special_tokens": [DEFAULT_SPEECH_TOKEN]}
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer

    @classmethod
    def clean_text(cls, origin_text):
        """remove punc, remove space between Chinese and keep space between English"""
        # remove punc
        text = re.sub("[，。？！,\.!?《》（）\·“”、\\/]", "", origin_text)
        # merge space
        text = re.sub("\s+", " ", text)

        # remove space between Chinese and keep space between English
        pattern = re.compile(r'([\u3400-\u4dbf\u4e00-\u9fff])')  # Chinese
        parts = pattern.split(text.strip())
        parts = [p for p in parts if len(p.strip()) > 0]
        text = "".join(parts)
        text = text.strip()

        text = text.lower()
        return text

    @classmethod
    def preprocess_texts(cls, origin_texts, tokenizer, max_len, decode=False):
        messages = []
        clean_texts = []
        for i, origin_text in enumerate(origin_texts):
            text = cls.clean_text(origin_text)
            clean_texts.append(text)
            text = text if not decode else ""
            message = [
                {"role": "user", "content": f"{DEFAULT_SPEECH_TOKEN}请转写音频为文字"},
                {"role": "assistant", "content": text},
            ]
            messages.append(message)

        texts = []
        if not decode:
            TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
        else:
            TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{''}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
        for i, msg in enumerate(messages):
            texts.append(
                tokenizer.apply_chat_template(
                    msg,
                    tokenize=True,
                    chat_template=TEMPLATE,
                    add_generation_prompt=False,
                    max_length=max_len,
                    truncation=True,
                    return_dict=False
                )
            )

        # Padding texts
        max_len_texts = max([len(text) for text in texts])
        if tokenizer.padding_side == "right":
            texts = [
                list(text) + [tokenizer.pad_token_id] * (max_len_texts - len(text))
                for text in texts
            ]
        else:
            texts = [
                [tokenizer.pad_token_id] * (max_len_texts - len(text)) + text
                for text in texts
            ]
        input_ids = torch.tensor(texts, dtype=torch.int)

        target_ids = input_ids.clone()
        target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID

        # first get the indices of the tokens
        mask_prompt = True
        if mask_prompt:
            mask_indices = torch.where(
                input_ids == tokenizer.convert_tokens_to_ids("assistant")
                )
            for i in range(mask_indices[0].size(0)):
                row = mask_indices[0][i]
                col = mask_indices[1][i]
                target_ids[row, : col + 2] = IGNORE_TOKEN_ID

        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        target_ids = target_ids.type(torch.LongTensor)
        input_ids = input_ids.type(torch.LongTensor)
        return input_ids, attention_mask, target_ids, clean_texts
