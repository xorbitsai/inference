import re
from typing import Iterable

TOKENIZE_PATTERN = r"(?:[a-zA-Z]\.)+|[a-zA-Z]+(?:['\-][a-zA-Z]+)*|\d+(?:\.\d+)?|[\u4e00-\u9fff]|\s+|\S"

# Inline paralinguistic tags understood by Ming-3.0-tts. They are literal
# model input and must reach the tokenizer verbatim, so every text-cleaning
# step has to route around them (see protect_paralinguistic_tags).
PARALINGUISTIC_TAGS = (
    "[Breathing]",
    "[Cough]",
    "[Sigh]",
    "[Laughter]",
    "[Surprise-ah]",
    "[Surprise-oh]",
    "[Dissatisfaction-hnn]",
    "[Uhm]",
    "[Confirmation-en]",
    "[Question-ei]",
    "[Question-ah]",
    "[Question-en]",
    "[Question-oh]",
    "[Question-huh]",
)
PARALINGUISTIC_TAG_RE = re.compile(
    "|".join(re.escape(tag) for tag in PARALINGUISTIC_TAGS)
)


def protect_paralinguistic_tags(text, normalize):
    """Apply ``normalize`` to the text around paralinguistic tags only.

    The normalization pipeline replaces every character outside its
    whitelist, which turns ``[Laughter]`` into ``，Laughter，``. Splitting on
    the tags and normalizing the surrounding segments keeps the tags intact
    without placeholder round-trips that TN could itself rewrite.
    """
    if not text:
        return text
    tags = PARALINGUISTIC_TAG_RE.findall(text)
    if not tags:
        return normalize(text)

    segments = PARALINGUISTIC_TAG_RE.split(text)
    pieces = []
    for index, segment in enumerate(segments):
        if segment.strip():
            # Reinstate the boundary whitespace: without it an English word
            # would be glued onto the neighbouring tag. The normalized text is
            # stripped first so this stays correct whether or not ``normalize``
            # trims its own output.
            lead = " " if segment[:1].isspace() else ""
            trail = " " if segment[-1:].isspace() else ""
            pieces.append(lead + normalize(segment).strip() + trail)
        else:
            pieces.append(segment)
        if index < len(tags):
            pieces.append(tags[index])
    return "".join(pieces)


def tokenize_mixed_text(text: str):
    """
    将混合文本字符串处理为标记列表，并保留原始的空格。
    """
    return re.findall(TOKENIZE_PATTERN, text)

def tokenize_mixed_text_iterator(text_iterator: Iterable[str]):
    """
    Args:
        text_iterator (Iterable[str]): 一个产出字符串的迭代器。
            例如，一个打开的文件对象，或一个生成器表达式。
    Yields:
        Iterator[str]: 逐一产出文本中的标记（单词、中文、标点、空格等）。
    """
    # 函数本身就是一个生成器，将从 re.finditer 返回的迭代器中逐一产出结果
    for chunk in text_iterator:
        # re.finditer 返回一个迭代器，而不是列表
        for match in re.finditer(TOKENIZE_PATTERN, chunk):
            yield match.group(0)


def split_uppercase_acronyms(text):
    """
    将连续大写字母（缩写词）逐字母用空格隔开。
    单个大写字母或首字母大写的单词不受影响。
    """
    return re.sub(r'[A-Z]{2,}', lambda m: ' '.join(m.group(0)), text)


def replace_parens_with_sep(text, target_sep='，'):
    """
    将中英文括号替换为target_sep。
    """
    for ch in '()（）':
        text = text.replace(ch, target_sep)
    return text


def clean_text_formatting(text):
    if not text:
        return text

    # 定义开头需要被替换为空格的“杂质标点”
    # 保留了：() {} [] （） 【】 "" '' “” ‘’ 《》 等成对符号
    # 替换掉：， 。 ！ ？ ； ： 、 …… 等容易出现在开头且没意义的标点
    leading_punct_pattern = r'^[，。！？；：、……\.\!\?\-\—]+'

    # 替换开头标点
    text = re.sub(leading_punct_pattern, ' ', text)
    # # 聚合连续空格
    # text = re.sub(r'\s+', ' ', text)
    # 去除首尾多余空格
    return text.strip()


def chinese_punc_en2cn(text):
    """
    含中文文本清理：包含中文即处理，多个堆叠标点最终只保留一个。
    1. 增加 URL和小数点 保护，防止链接内部符号被误转。
    2. 增强标点优先级逻辑，确保顿号（、）正确去重。
    3. 修复异类标点堆叠的匹配规则。
    4、修复开头符号
    """
    # # 允许的字符范围：汉字、中文标点、全角符号、空格、以及常见的英文标点
    # # 明确排除：a-zA-Z（英文字母）、0-9（数字）
    # # gen_tag = 'tn'
    # if re.search(r'[a-zA-Z0-9]', text):
    #     return text

    # 仅检查是否包含中文，无中文则直接返回原文本
    # gen_tag = 'tn_v1'
    if not text or not re.search(r'[\u4e00-\u9fa5]', text):
        return clean_text_formatting(text)

    # URL 占位保护
    urls = re.findall(r'https?://[^\s，。！？]+', text)
    for i, url in enumerate(urls):
        text = text.replace(url, f"URLHOLDER{i}")

    # 保护数字间的小数点，防止被 punc_map 误转为 。
    text = re.sub(r'(?<=\d)\.(?=\d)', 'DOTPLACEHOLDER', text)

    # 基础符号转换
    text = text.replace('...', '……')
    punc_map = {
        ',': '，', '.': '。', ';': '；', ':': '：', '!': '！', '?': '？',
        '(': '（', ')': '）', '[': '【', ']': '】', '<': '《', '>': '》', '_': '——'
    }
    map_pattern = re.compile('|'.join(re.escape(k) for k in punc_map.keys()))
    text = map_pattern.sub(lambda m: punc_map[m.group(0)], text)

    # 还原小数点
    text = text.replace('DOTPLACEHOLDER', '.')

    # 修复：引号计数与转换 (改用闭包计数，避免 := 运算符在 re.sub 中的潜在崩溃)
    dq_count = [0]
    sq_count = [0]

    def dq_sub(_):
        dq_count[0] += 1
        return '“' if dq_count[0] % 2 != 0 else '”'

    def sq_sub(_):
        sq_count[0] += 1
        return '‘' if sq_count[0] % 2 != 0 else '’'

    text = re.sub(r'"', dq_sub, text)
    text = re.sub(r"'", sq_sub, text)

    # 彻底消除标点周边的空格
    ch_puncs = r'，。！？；：“”‘’（）【】《》——……、'
    text = re.sub(fr'[ \t\u00A0\u3000]*([{ch_puncs}])[ \t\u00A0\u3000]*', r'\1', text)

    # 标点递归清理
    while True:
        old_text = text

        # 压缩完全相同的重复标点
        text = re.sub(r'([{ch_puncs}])\1+', r'\1', text)

        # 弱标点(，；、) 遇到 强标点(。！？) 时删除弱标点
        text = re.sub(r'[，；、]+([。！？])', r'\1', text)
        text = re.sub(r'([。！？])[，；、]+', r'\1', text)

        # 处理 逗号/分号 与 顿号 的堆叠
        text = re.sub(r'[，；]+[、]|[、][，；]+', '、', text)

        # 强标点间唯一化
        text = re.sub(r'([。！？])[。！？]+', r'\1', text)

        # 剩余异类弱标点堆叠唯一化
        text = re.sub(r'([，；、])[，；、]+', r'\1', text)

        if old_text == text:
            break

    # 压缩汉字/单词间连续空格
    text = re.sub(r'[ \t\u00A0\u3000]+', ' ', text)

    # 还原 URL
    for i, url in enumerate(urls):
        text = text.replace(f"URLHOLDER{i}", url)

    return clean_text_formatting(text)


# 弱标点结尾视为未收束，升格为句末标点（替换而非追加：
# 纯英文不经 chinese_punc_en2cn 的标点去重，追加会留下 ",." 漏网）
WEAK_TAIL_PUNCT = '，,、;；'


def add_period(text):
    """尾部无标点则添加句号；汉字结尾加'。'，否则加'.'。弱标点(，,、;；)结尾升格为句末标点"""
    stripped = text.rstrip()
    if not stripped:          # 空串或全空白
        return "."

    last_char = stripped[-1]

    # 弱标点结尾（逗号/顿号/分号）：升格为句末标点
    if last_char in WEAK_TAIL_PUNCT:
        has_chinese = bool(re.search(r'[一-鿿]', stripped))   # U+4E00-U+9FFF 汉字区
        return stripped[:-1] + ('。' if has_chinese else '.')

    # 如果最后一个字符不是字母或数字，则认为它已是标点符号，直接返回
    if not last_char.isalnum():
        return stripped

    # 判断是否为汉字（基本汉字区和扩展A区）
    is_chinese = ('\u4e00' <= last_char <= '\u9fff' or
                  '\u3400' <= last_char <= '\u4dbf')
    suffix = '。' if is_chinese else '.'
    return stripped + suffix


def fallback_replace_non_alnum(text, replacement='，'):
    """
    兜底方案：将非汉字、英文、数字及基础TTS标点的字符替换为指定符号。
    并合并连续的替换符号。
    保留中文基础标点：，。！？；：、
    保留英文基础标点：,.!?;:
    保留小数点：.和空格等
    """
    # 白名单新增: () （） " “ ” (半角/全角括号 + 半角/中文双引号) 不再被替换成逗号
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9，。！？；：、,.!?;:\. \u2019\u2018'\"()（）\u201C\u201D]", replacement, text)
    text = re.sub(re.escape(replacement) + '+', replacement, text).strip()
    return text


def normalize_text_kit(text):
    """
    完整文本规范化流水线：
    连续大写字母拆开（NBA → N B A）
    (已禁用) 中英括号替换为逗号
    兜底：非汉字/英文/数字/基础标点 → 逗号 (已把 () （） " “ ” 加入白名单)
    英文标点→中文标点 + 标点去重 + 空格清理
    """
    #print(f'[normalize_text_kit] 输入                   : {text!r}')
    # ---- 已禁用: 连续大写字母拆分（NBA → N B A）----
    # text = split_uppercase_acronyms(text)
    #print(f'[normalize_text_kit] step1 大写拆分后       : {text!r}')
    # ---- 已禁用: 不再把括号替换成逗号 ----
    # text = replace_parens_with_sep(text)
    # print(f'[normalize_text_kit] step2 括号替换后       : {text!r}')
    text = fallback_replace_non_alnum(text)
    #print(f'[normalize_text_kit] step3 兜底非白名单→，  : {text!r}')
    text = chinese_punc_en2cn(text)
    #print(f'[normalize_text_kit] step4 英文标点→中文    : {text!r}')
    text = re.sub(r'\s+', ' ', text).strip()
    #print(f'[normalize_text_kit] step5 空格清理后       : {text!r}')
    return text


def strip_markdown_code(text):
    """
    清除 Markdown 行内代码标记（反引号）。
    支持单反引号 `code` 和双反引号 ``code`` 两种格式。
    """
    # 双反引号优先匹配（避免单反引号先匹配到内层）
    text = re.sub(r'``(.+?)``', r'\1', text)
    # 单反引号
    text = re.sub(r'`(.+?)`', r'\1', text)

    # 清除残留的未配对反引号
    text = text.replace('`', '')
    return text


def preprosess_text(text):
    """
    文本预处理：
    清除 Markdown 行内代码标记
    """
    text = strip_markdown_code(text)
    return text
