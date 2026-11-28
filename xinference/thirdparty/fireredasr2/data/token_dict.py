# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

import logging

logger = logging.getLogger(__name__)


class TokenDict:
    def __init__(self, dict_path, unk=""):
        assert dict_path != ""
        self.id2word, self.word2id = self.read_dict(dict_path)
        self.unk = unk
        assert unk == "" or unk in self.word2id
        self.unkid = self.word2id[unk] if unk else -1

    def get(self, key, default):
        if type(default) == str:
            default = self.word2id[default]
        return self.word2id.get(key, default)

    def __getitem__(self, key):
        if type(key) == str:
            if self.unk:
                return self.word2id.get(key, self.word2id[self.unk])
            else:
                return self.word2id[key]
        elif type(key) == int:
            return self.id2word[key]
        else:
            raise TypeError("Key should be str or int")

    def __len__(self):
        return len(self.id2word)

    def __contains__(self, query):
        if type(query) == str:
            return query in self.word2id
        elif type(query) == int:
            return query in self.id2word
        else:
            raise TypeError("query should be str or int")

    def read_dict(self, dict_path):
        id2word, word2id = [], {}
        with open(dict_path, encoding='utf8') as f:
            for i, line in enumerate(f):
                tokens = line.strip().split()
                if len(tokens) >= 2:
                    word, index = tokens[0], int(tokens[1])
                elif len(tokens) == 1:
                    word, index = tokens[0], i
                else:  # empty line or space
                    logger.info(f"Find empty line or space '{line.strip()}' in {dict_path}:L{i}, set to ' '")
                    word, index = " ", i
                assert len(id2word) == index
                assert len(word2id) == index
                if word == "<space>":
                    logger.info(f"NOTE: Find <space> in {dict_path}:L{i} and convert it to ' '")
                    word = " "
                word2id[word] = index
                id2word.append(word)
        assert len(id2word) == len(word2id)
        return id2word, word2id
