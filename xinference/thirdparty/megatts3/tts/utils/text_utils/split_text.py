# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

def chunk_text_chinese(text, limit=60):
    # 中文字符匹配
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    # 标点符号匹配
    punctuation = "，。！？；：,\.!?;"
    
    result = []  # 存储断句结果
    current_chunk = []  # 当前片段
    chinese_count = 0  # 中文字符计数

    i = 0
    while i < len(text):
        char = text[i]
        current_chunk.append(char)
        if chinese_pattern.match(char):
            chinese_count += 1
        
        if chinese_count >= limit:  # 达到限制字符数
            # 从当前位置往前找最近的标点符号
            for j in range(len(current_chunk) - 1, -1, -1):
                if current_chunk[j] in punctuation:
                    result.append(''.join(current_chunk[:j + 1]))
                    current_chunk = current_chunk[j + 1:]
                    chinese_count = sum(1 for c in current_chunk if chinese_pattern.match(c))
                    break
            else:
                # 如果前面没有标点符号，则继续找后面的标点符号
                for k in range(i + 1, len(text)):
                    if text[k] in punctuation:
                        result.append(''.join(current_chunk)+text[i+1:k+1])
                        current_chunk = []
                        chinese_count = 0
                        i = k
                        break
        i+=1

    # 添加最后剩余的部分
    if current_chunk:
        result.append(''.join(current_chunk))

    return result

def chunk_text_english(text, max_chars=130):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

if __name__ == '__main__':
    print(chunk_text_chinese("哇塞！家人们，你们太好运了。我居然发现了一个宝藏零食大礼包，简直适合所有人的口味！有香辣的，让你舌尖跳舞；有盐焗的，咸香可口；还有五香的，香气四溢。就连怀孕的姐妹都吃得津津有味！整整三十包啊！什么手撕蟹柳、辣子鸡、嫩豆干、手撕素肉、鹌鹑蛋、小肉枣肠、猪肉腐、魔芋、魔芋丝等等，应有尽有。香辣土豆爽辣过瘾，各种素肉嚼劲十足，鹌鹑蛋营养美味，真的太多太多啦，...家人们，现在价格太划算了，赶紧下单。"))
    print(chunk_text_english("Washington CNN When President Donald Trump declared in the House Chamber this week that executives at the nation’s top automakers were “so excited” about their prospects amid his new tariff regime, it did not entirely reflect the conversation he’d held with them earlier that day."))