from abc import ABC, abstractmethod
from typing import List, Dict

from ovis.util.constants import IMAGE_TOKEN_ID, IGNORE_ID, IMAGE_TOKEN


class ConversationFormatter(ABC):
    support_tokenizer_types = None

    def __init__(self, tokenizer):
        tokenizer_type = type(tokenizer).__name__
        assert tokenizer_type in self.support_tokenizer_types, \
            f'Invalid tokenizer type, expected one from `{self.support_tokenizer_types}`, but got `{tokenizer_type}`'
        self.tokenizer = tokenizer
        self.image_token = IMAGE_TOKEN
        self.image_token_id = IMAGE_TOKEN_ID
        self.ignore_id = IGNORE_ID
        self.im_end = None

    def _tokenize_with_image_symbol(self, text):
        text_chunks = [self.tokenizer(chunk, add_special_tokens=False).input_ids for chunk in
                       text.split(self.image_token)]
        token_ids = []
        num_chuck = len(text_chunks)
        for i, chunk in enumerate(text_chunks):
            token_ids.extend(chunk)
            if i < num_chuck - 1:
                token_ids.append(self.image_token_id)
        return token_ids

    @abstractmethod
    def format(self, conversations: List[Dict], generation_preface=None):
        pass

    @abstractmethod
    def format_query(self, query, generation_preface=""):
        pass


class QwenConversationFormatter(ConversationFormatter):
    support_tokenizer_types = ['QWenTokenizer', 'Qwen2TokenizerFast']

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.from2role = {
            "system": "<|im_start|>system\n",
            "human": "<|im_start|>user\n",
            "gpt": "<|im_start|>assistant\n",
        }
        self.gpt_token_num = None
        self.im_end = "<|im_end|>\n"
        self.default_system_prompt = "You are a helpful assistant."

    def format(self, conversations: List[Dict], generation_preface=None):
        if self.gpt_token_num is None:
            self.gpt_token_num = len(self.tokenizer(self.from2role["gpt"], add_special_tokens=False).input_ids)

        if conversations[0]["from"] != "system":
            conversations.insert(0, {
                "from": "system",
                "value": self.default_system_prompt
            })

        if generation_preface is not None:
            conversations.append({
                "from": "gpt",
                "value": generation_preface
            })

        prompt = ""
        input_ids = []
        labels = []
        num_conversation = len(conversations)
        for i, conversation in enumerate(conversations):
            frm = conversation["from"]
            role = self.from2role[frm]
            message = conversation["value"]
            text = role + message
            if i < num_conversation - 1 or generation_preface is None:
                text += self.im_end
            prompt += text
            token_ids = self._tokenize_with_image_symbol(text)
            input_ids.extend(token_ids)
            label_ids = [self.ignore_id] * len(token_ids)
            if frm == "gpt" and generation_preface is None:
                # learning `\n` following `im_end` is meaningless, so the last `\n` token is ignored in label
                label_ids[self.gpt_token_num:-1] = token_ids[self.gpt_token_num:-1]
            labels.extend(label_ids)

        assert self._tokenize_with_image_symbol(prompt) == input_ids
        assert len(input_ids) == len(labels)

        return prompt, input_ids, labels

    def format_query(self, query, generation_preface=""):
        prompt, input_ids, _ = self.format([{
            "from": "human",
            "value": query
        }], generation_preface=generation_preface)

        return prompt, input_ids


class Llama3ConversationFormatter(ConversationFormatter):
    support_tokenizer_types = ['PreTrainedTokenizerFast']

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.from2role = {
            "system": "<|start_header_id|>system<|end_header_id|>\n\n",
            "human": "<|start_header_id|>user<|end_header_id|>\n\n",
            "gpt": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        }
        self.gpt_token_num = None
        self.im_end = "<|eot_id|>"
        self.default_system_prompt = "You are a helpful and honest multimodal assistant."
        self.bos_token = "<|begin_of_text|>"
        self.bos_token_ids = None

    def format(self, conversations: List[Dict], generation_preface=None):
        if self.gpt_token_num is None:
            self.gpt_token_num = len(self.tokenizer(self.from2role["gpt"], add_special_tokens=False).input_ids)

        if self.bos_token_ids is None:
            self.bos_token_ids = self.tokenizer(self.bos_token, add_special_tokens=False).input_ids

        if conversations[0]["from"] != "system":
            conversations.insert(0, {
                "from": "system",
                "value": self.default_system_prompt
            })

        if generation_preface is not None:
            conversations.append({
                "from": "gpt",
                "value": generation_preface
            })

        prompt = "" + self.bos_token
        input_ids = [] + self.bos_token_ids
        labels = [] + [IGNORE_ID] * len(input_ids)
        num_conversation = len(conversations)
        for i, conversation in enumerate(conversations):
            frm = conversation["from"]
            role = self.from2role[frm]
            message = conversation["value"].strip()
            text = role + message
            if i < num_conversation - 1 or generation_preface is None:
                text += self.im_end
            prompt += text
            token_ids = self._tokenize_with_image_symbol(text)
            input_ids.extend(token_ids)
            label_ids = [self.ignore_id] * len(token_ids)
            if frm == "gpt":
                label_ids[self.gpt_token_num:] = token_ids[self.gpt_token_num:]
            labels.extend(label_ids)

        assert self._tokenize_with_image_symbol(prompt) == input_ids
        assert len(input_ids) == len(labels)

        return prompt, input_ids, labels

    def format_query(self, query, generation_preface=""):
        prompt, input_ids, _ = self.format([{
            "from": "human",
            "value": query
        }], generation_preface=generation_preface)

        return prompt, input_ids


class GemmaConversationFormatter(ConversationFormatter):
    support_tokenizer_types = ['GemmaTokenizer', 'GemmaTokenizerFast']

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        # Gemma does not support system prompt
        self.from2role = {
            "human": "<start_of_turn>user\n",
            "gpt": "<start_of_turn>model\n",
        }
        self.gpt_token_num = None
        self.im_end = "<end_of_turn>\n"
        self.bos_token = "<bos>"
        self.bos_token_ids = None

    def format(self, conversations: List[Dict], generation_preface=None):
        if self.gpt_token_num is None:
            self.gpt_token_num = len(self.tokenizer(self.from2role["gpt"], add_special_tokens=False).input_ids)

        if self.bos_token_ids is None:
            self.bos_token_ids = self.tokenizer(self.bos_token, add_special_tokens=False).input_ids

        if conversations[0]["from"] == "system":
            raise ValueError("Gemma does not support system prompt")

        if generation_preface is not None:
            conversations.append({
                "from": "gpt",
                "value": generation_preface
            })

        prompt = "" + self.bos_token
        input_ids = [] + self.bos_token_ids
        labels = [] + [IGNORE_ID] * len(input_ids)
        num_conversation = len(conversations)
        for i, conversation in enumerate(conversations):
            frm = conversation["from"]
            role = self.from2role[frm]
            message = conversation["value"].strip()
            text = role + message
            if i < num_conversation - 1 or generation_preface is None:
                text += self.im_end
            prompt += text
            token_ids = self._tokenize_with_image_symbol(text)
            input_ids.extend(token_ids)
            label_ids = [self.ignore_id] * len(token_ids)
            if frm == "gpt":
                # learning `\n` following `im_end` is meaningless, so the last `\n` token is ignored in label
                label_ids[self.gpt_token_num:-1] = token_ids[self.gpt_token_num:-1]
            labels.extend(label_ids)

        assert self._tokenize_with_image_symbol(prompt) == input_ids
        assert len(input_ids) == len(labels)

        return prompt, input_ids, labels

    def format_query(self, query, generation_preface=""):
        prompt, input_ids, _ = self.format([{
            "from": "human",
            "value": query
        }], generation_preface=generation_preface)

        return prompt, input_ids
