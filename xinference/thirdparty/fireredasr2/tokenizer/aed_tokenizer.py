# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

import logging
import re

import sentencepiece as spm

from ..data.token_dict import TokenDict


class ChineseCharEnglishSpmTokenizer:
    """
    - One Chinese char is a token.
    - Split English word into SPM and one piece is a token.
    - Ignore ' ' between Chinese char
    - Replace ' ' between English word with "▁" by spm_model
    - Need to put SPM piece into dict file
    - If not set spm_model, will use English char and <space>
    """
    SPM_SPACE = "▁"

    def __init__(self, dict_path, spm_model, unk="<unk>", space="<space>"):
        self.dict = TokenDict(dict_path, unk=unk)
        self.space = space
        if spm_model:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(spm_model)
        else:
            self.sp = None
            print("[WRAN] Not set spm_model, will use English char")
            print("[WARN] Please check how to deal with ' '(space)")
            if self.space not in self.dict:
                print("Please add <space> to your dict, or it will be <unk>")

    def tokenize(self, text, replace_punc=True):
        #if text == "":
        #    logging.info(f"empty text")
        text = text.upper()
        tokens = []
        if replace_punc:
            text = re.sub("[，。？！,\.?!]", " ", text)
        pattern = re.compile(r'([\u3400-\u4dbf\u4e00-\u9fff])')
        parts = pattern.split(text.strip())
        parts = [p for p in parts if len(p.strip()) > 0]
        for part in parts:
            if pattern.fullmatch(part) is not None:
                tokens.append(part)
            else:
                if self.sp:
                    for piece in self.sp.EncodeAsPieces(part.strip()):
                        tokens.append(piece)
                else:
                    for char in part.strip():
                        tokens.append(char if char != " " else self.space)
        tokens_id = []
        for token in tokens:
            tokens_id.append(self.dict.get(token, self.dict.unk))
        return tokens, tokens_id

    def detokenize(self, inputs, join_symbol="", replace_spm_space=True):
        """inputs is ids or tokens, do not need self.sp"""
        if len(inputs) > 0 and type(inputs[0]) == int:
            tokens = [self.dict[id] for id in inputs]
        else:
            tokens = inputs
        s = f"{join_symbol}".join(tokens)
        if replace_spm_space:
            s = s.replace(self.SPM_SPACE, ' ').strip()
        return s

    def merge_spm_timestamp(self, timestamp):
        merged_timestamp = []
        i = 0
        while i < len(timestamp):
            token, start, end = timestamp[i]
            if token.startswith(self.SPM_SPACE):
                token = token.replace(self.SPM_SPACE, "")
                current_end = end
                next_i = i + 1
                while next_i < len(timestamp):
                    next_token, next_start, next_end = timestamp[next_i]
                    if re.match("^[a-zA-Z']+$", next_token):
                        token += next_token
                        current_end = next_end
                        next_i += 1
                    else:
                        break
                end = current_end
                i = next_i
            else:
                i += 1
            merged_timestamp.append((token, start, end))
        return merged_timestamp
