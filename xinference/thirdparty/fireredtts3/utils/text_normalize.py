"""
TTS 文本前端（text front-end）工具。

some functions are adapted from https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/utils/frontend_utils.py
some functions are adapted from https://github.com/OpenBMB/VoxCPM/blob/main/src/voxcpm/utils/text_normalize.py
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

try:
    import regex
except ImportError:  # pragma: no cover
    regex = None

# --------------------------------------------------------------------------- #
# 1) 文本初步清洗
# --------------------------------------------------------------------------- #
def preprocess_text(sentence: str) -> str:
    """对文本做最基础的 utf-8 级清洗（参考 FireRedTTS.preprocess_text）。"""
    if not sentence:
        return ""

    # utf-8 编码/解码，忽略无法解码的字节
    sentence = bytes(sentence, "utf-8").decode("utf-8", "ignore")

    if regex is not None:
        # 去除零宽字符（保留零宽连接符 \u200d）
        sentence = regex.sub(r"[\p{Cf}--[\u200d]]", "", sentence, flags=regex.V1)
        # 去除私有区字符
        sentence = regex.sub(r"\p{Co}", "", sentence)
    else:
        sentence = re.sub(r"[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff]", "", sentence)
        sentence = re.sub(r"[\ue000-\uf8ff]", "", sentence)

    sentence = sentence.replace("\u00a0", " ")  # 不换行空格 -> 普通空格
    sentence = sentence.replace("\ufffd", "")   # 替换字符 -> 空
    # 行分隔符 / 段分隔符 -> 换行（U+2028 LINE SEPARATOR, U+2029 PARAGRAPH SEPARATOR）
    sentence = sentence.replace("\u2028", "\n")
    sentence = sentence.replace("\u2029", "\n")

    return sentence


# --------------------------------------------------------------------------- #
# 2) Markdown 清洗 + 去除表情符号（参考 VoxCPM.utils.text_normalize）
# --------------------------------------------------------------------------- #
def clean_markdown(md_text: str) -> str:
    """去除 Markdown 语法，保留纯文本内容。"""
    if not md_text:
        return md_text
    # 去除图片语法 ![alt](url)
    md_text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", md_text)
    # 去除链接但保留文本 [text](url) -> text
    md_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md_text)
    # 替换无序列表符号
    md_text = re.sub(r"^(\s*)-\s+", r"\1", md_text, flags=re.MULTILINE)
    # 去除标题符号（#）
    md_text = re.sub(r"^#{1,6}\s*", "", md_text, flags=re.MULTILINE)
    # 只处理句首的加粗/斜体/删除线等强调标记（**text**、*text*、__text__、~~text~~），
    # 行首的 * / ~ 几乎只可能是 markdown 标记；正文中的素 * / ~ （乘法、波浪号）不受影响。
    md_text = re.sub(r"^\s*\*\*([^*]+)\*\*", r"\1", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"^\s*__([^_]+)__", r"\1", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"^\s*\*([^*]+)\*", r"\1", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"^\s*~~([^~]+)~~", r"\1", md_text, flags=re.MULTILINE)
    # 行首多余的离散 * / ~（例如独立的 * item）
    md_text = re.sub(r"^\s*[*~]\s+", "", md_text, flags=re.MULTILINE)
    # 去除多余空行
    md_text = re.sub(r"\n\s*\n", "\n", md_text)
    md_text = md_text.strip()
    return md_text


def remove_emoji(text: str) -> str:
    """去除表情符号（Emoji_Presentation 与带变体选择符的 Emoji）。"""
    if regex is None:
        return text
    return regex.compile(
        r"\p{Emoji_Presentation}|\p{Emoji}\uFE0F", flags=regex.UNICODE
    ).sub("", text)


_CJK = r"\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
_RE_CJK_SPACE_CJK = re.compile(r"([{}])\s+([{}])".format(_CJK, _CJK))
_RE_CJK_SPACE_LATIN = re.compile(r"([{0}])\s+([a-zA-Z0-9])".format(_CJK))
_RE_LATIN_SPACE_CJK = re.compile(r"([a-zA-Z0-9])\s+([{0}])".format(_CJK))


def clean_tn_spaces(text: str) -> str:
    """清理 TN 后在汉字之间、汉字与英文/数字之间多余的空格。

    TN（尤其 LLM-based）经常在汉字/数字/符号间插入不必要的空格，如：
    "今天 天气 很好"  → "今天天气很好"
    "苹果 AI 助手"     → "苹果AI助手"
    """
    if not text:
        return text
    text = _RE_CJK_SPACE_CJK.sub(r"\1\2", text)
    text = _RE_CJK_SPACE_LATIN.sub(r"\1\2", text)
    text = _RE_LATIN_SPACE_CJK.sub(r"\1\2", text)
    return text


# 符号化简（参考 FireRedTTS utils.py 的 symbol_reduction）。
# 只保留"输出会被后续 _SYMBOL_TO_SPACE / _SYMBOL_TO_COMMA 处理"的映射：
#   - 映射到 ~（→ 逗号）、·（→ 空格）、...（→ 逗号）的条目。
# 其余映射的目的地（如 " ( ) . : ; ! ? - % + = 等）本身就在 _WETEXT_KEEP
# 允许集中，原字符也会被保留，映射纯属多余，故全部去掉。
_SYMBOL_REDUCTION = {
    # 波浪/破折线 → ASCII ~（随后被 _SYMBOL_TO_COMMA 化简为逗号）
    "〜": "~", "～": "~",
    # 间隔点/圆点 → ·（随后被 _SYMBOL_TO_SPACE 化简为空格）
    "・": "·", "•": "·", "‧": "·",
    # 省略号变体 → "..."（避免被 _SYMBOL_TO_COMMA 吞成一个逗号）
    "…": "...", "⋯": "...", "〰": "...", "﹏": "...",
}
_SYMBOL_TO_SPACE = re.compile(r"[·•‧│|¦/\\]")
_SYMBOL_TO_COMMA = re.compile(r"[…~&*%$#^:;/\\|]+")
_WETEXT_KEEP = re.compile(
    r"[^"
    r"\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"   # CJK 汉字
    r"0-9A-Za-z"
    r"，。、？！：；…—～·"
    r".,:;!?()\[\]'\"\-_"
    r"\s"
    r"]"
)


def _apply_symbol_reduction(text: str) -> str:
    """把全角/异体符号化简为对应的半角标准符号（参考 symbol_reduction）。"""
    return "".join(_SYMBOL_REDUCTION.get(ch, ch) for ch in text)


def clean_wetext_output(text: str) -> str:
    # 清洗 wetext 归一化后的中/英文输出，把无法读出的符号化简为逗号或空格。
    if not text:
        return text
    text = _apply_symbol_reduction(text)
    text = _SYMBOL_TO_SPACE.sub(" ", text)
    text = _SYMBOL_TO_COMMA.sub(",", text)
    text = _WETEXT_KEEP.sub("", text)
    # 规整空白，连续逗号合并
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r"[,，]{2,}", ",", text)
    text = re.sub(r"\s+[,，]", ",", text)
    text = re.sub(r"[,，]+\s*$", "", text)
    return text


def clean_text(text: str, llm_normalizer: Optional[Callable[[str], str]] = None) -> str:
    """文本初步清洗：utf-8 清洗 + markdown 清洗 + 去除表情符号 + 规整空白。

    Args:
        text: 输入文本。
        llm_normalizer: 可选，一个可调用的文本归一化器（通常为
            ``llm_tn`` 的 ``TextNormalizer.normalize``）。若提供，则在基础
            清洗之后调用它以支持多语种 TN（例如日语、韩语等 wetext
            不支持的语种）。该调用方需自行保证模板齐全。
    """
    if not text:
        return text
    text = preprocess_text(text)
    text = clean_markdown(text)
    text = remove_emoji(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    if llm_normalizer is not None:
        try:
            text = llm_normalizer(text)
        except Exception as e:  # 失败时回退为清洗后的原文
            print(f"[WARN] llm_tn normalization failed, fallback to raw text: {e}", flush=True)
        text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# 3) 分句分段（参考 VoxCPM.utils.text_normalize.split_paragraph）
# --------------------------------------------------------------------------- #
def _is_decimal_dot(text: str, i: int) -> bool:
    """判断位置 *i* 上的 ``.`` / ``:`` 是否为数字内部标点（而非句号/冒号）。

    例如 ``€1.2M`` 中的小数点 (前后是数字)、``19:30`` 中的冒号 (前后是数字)。
    用于分句时不切断这些数字内部的标点。
    """
    return (i > 0 and i + 1 < len(text)
            and text[i - 1].isdigit()
            and text[i + 1].isdigit())


def split_paragraph(
    text: str,
    tokenize: Optional[Callable[[str], List[str]]] = None,
    lang: str = "zh",
    token_max_n: int = 80,
    token_min_n: int = 60,
    merge_len: int = 20
) -> List[str]:
    """将段落按句号/标点拆分为若干句，并合并过短的句子。

    分句逻辑：
    1. 每个句子最大长度 ``token_max_n``、最小长度 ``token_min_n``；
       若末尾句子长度小于 ``merge_len`` 则并入前一句。
    2. 按语种计算句子长度（zh 按字符数，其余按 token 数）。
    3. 按标点切分句子。

    注意：紧邻两边都是数字的 ``.``（如 ``€1.2M``、``0.85``、``v1.8.3``）
    及 ``:``（如 ``19:30``）被视为数字内部标点，不做切分。
    """
    def _measure(_text: str) -> int:
        if lang == "zh":
            return len(_text)
        if tokenize is not None:
            n = tokenize(_text)
            if isinstance(n, int):
                return n
            return len(n)
        return len(_text)

    def calc_utt_length(_text: str) -> int:
        return _measure(_text)

    def should_merge(_text: str) -> bool:
        return _measure(_text) < merge_len

    if lang == "zh":
        pounc = ["。", "？", "！", "；", "、", ".", "?", "!", ";"]
    else:
        pounc = [".", "?", "!", ";"]

    # 按标点切分（跳过数字内部的小数点 ``.`` ）
    st = 0
    utts: List[str] = []
    for i, c in enumerate(text):
        if c in pounc:
            # 对 . 和 : 检查是否在数字内部（如 €1.2M）
            if c == "." and _is_decimal_dot(text, i):
                continue
            if len(text[st:i]) > 0:
                utts.append(text[st:i] + c)
            if i + 1 < len(text) and text[i + 1] in ['"', "”"]:
                tmp = utts.pop(-1)
                utts.append(tmp + text[i + 1])
                st = i + 2
            else:
                st = i + 1
    trailing = text[st:] if st < len(text) else ""
    if trailing:
        utts.append(trailing)
    elif len(utts) == 0:
        utts.append(text + ("。" if lang == "zh" else ""))
    # 合并过短 / 超长处理
    final_utts: List[str] = []
    cur_utt = ""
    for utt in utts:
        if calc_utt_length(cur_utt + utt) > token_max_n and calc_utt_length(cur_utt) > token_min_n:
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    if len(cur_utt) > 0:
        if should_merge(cur_utt) and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            final_utts.append(cur_utt)

    return final_utts


# --------------------------------------------------------------------------- #
# 4) 语种自动判定（复用 llm_tn/text_normalizer.py 的 fasttext）
# --------------------------------------------------------------------------- #
# 将 llm_tn 的 locale（如 "zh-CN"、"en-US"）映射到 fireredtts3 的 lang tag。
_LOCALE_TO_LANG_TAG = {
    "zh-CN": "Chinese",
    "en-US": "English",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "es-MX": "Spanish",
    "fr-FR": "French",
    "ru-RU": "Russian",
    "ar-SA": "Arabic",
    "tr-TR": "Turkish",
    "id-ID": "Indonesian",
    "pt-BR": "Portuguese",
    "it-IT": "Italian",
    "nl-NL": "Dutch",
    "vi-VN": "Vietnamese",
    "de-DE": "German",
    "uk-UA": "Ukrainian",
    "th-TH": "Thai",
    "pl-PL": "Polish",
    "ro-RO": "Romanian",
    "el-GR": "Greek",
    "cs-CZ": "Czech",
    "fi-FI": "Finnish",
    "hi-IN": "Hindi",
}

# lang tag -> llm_tn locale（反向映射）。当上层已指定语种时，用它把已知语种
# 传给 llm_tn，从而覆盖 llm_tn 内部的自动检测，避免误判（如俄语被当成中文）。
_LANG_TAG_TO_LOCALE = {v: k for k, v in _LOCALE_TO_LANG_TAG.items()}
# 中文方言统一走 zh-CN 的 llm_tn 模板（方言无独立模板）。
_LANG_DIALECT_TO_LOCALE = {
    "Cantonese": "zh-CN",
    "ZH_Anhui": "zh-CN", "ZH_Fujian": "zh-CN", "ZH_Gansu": "zh-CN",
    "ZH_Guizhou": "zh-CN", "ZH_Hebei": "zh-CN", "ZH_Henan": "zh-CN",
    "ZH_Hubei": "zh-CN", "ZH_Hunan": "zh-CN", "ZH_Jiangxi": "zh-CN",
    "ZH_Liaoning": "zh-CN", "ZH_Minnan": "zh-CN", "ZH_Ningxia": "zh-CN",
    "ZH_Shaanxi": "zh-CN", "ZH_Shandong": "zh-CN", "ZH_Shanghai": "zh-CN",
    "ZH_Shanxi": "zh-CN", "ZH_Sichuan": "zh-CN", "ZH_Tianjin": "zh-CN",
    "ZH_Wenzhou": "zh-CN", "ZH_Wu": "zh-CN", "ZH_Yunnan": "zh-CN",
}


def lang_tag_to_locale(lang_tag: str) -> Optional[str]:
    """把 lang tag 映射为 llm_tn 的 locale。

    支持标准语言 tag（如 ``"Russian"``）与中文方言 tag（如 ``"ZH_Sichuan"``、
    ``"Cantonese"``）。方言无独立 llm_tn 模板，统一映射到 ``zh-CN``。
    无法识别时返回 ``None``。
    """
    if not lang_tag:
        return None
    if lang_tag in _LANG_TAG_TO_LOCALE:
        return _LANG_TAG_TO_LOCALE[lang_tag]
    if lang_tag in _LANG_DIALECT_TO_LOCALE:
        return _LANG_DIALECT_TO_LOCALE[lang_tag]
    return None


def detect_language(
    text: str,
    fasttext_detector: Optional[Callable[[str], Optional[str]]] = None,
    default_locale: str = "zh-CN",
) -> str:
    """自动判定文本语种
    内部使用 Meta FastText ``lid.176.ftz`` 模型。
    """
    if fasttext_detector is not None:
        try:
            locale = fasttext_detector(text)
            if locale:
                tag = _LOCALE_TO_LANG_TAG.get(locale)
                if tag:
                    return tag
        except Exception:
            pass

    # 回退：含汉字 -> 中文；含日文假名 -> 日文；否则英文
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Chinese"
    if re.search(r"[\u3040-\u30ff\u31f0-\u31ff]", text):  # 平假名/片假名
        return "Japanese"
    return "English"


# --------------------------------------------------------------------------- #
# 5) 归一化器工厂
# --------------------------------------------------------------------------- #
def build_wetext_normalizer() -> Optional[Callable[[str], str]]:
    """构建 wetext 归一化器，仅处理中文和英文。

    实现参考 VoxCPM 的
    ``voxcpm/utils/text_normalize.py``：使用 ``wetext.Normalizer``（zh / en）
    做文本归一化，并用 ``contains_chinese`` 自动判定中/英。仅支持中文和英文，
    其他语种需使用 ``llm_tn``。

    Returns:
        一个以 ``text`` 为输入、返回归一化文本的可调用对象；若 wetext 不可用
        返回 ``None``。
    """
    try:
        from wetext import Normalizer

        zh_tn_model = Normalizer(lang="zh", operator="tn", remove_erhua=True)
        en_tn_model = Normalizer(lang="en", operator="tn")

        chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]+")

        def _contains_chinese(text: str) -> bool:
            return bool(chinese_char_pattern.search(text))

        def _normalize(text: str) -> str:
            if not text:
                return text
            is_zh = _contains_chinese(text)
            if is_zh:
                out = zh_tn_model.normalize(text)
            else:
                out = en_tn_model.normalize(text)
            # 对 wetext TN 输出做强力清洗，去掉所有不应保留的特殊符号
            return clean_wetext_output(out)

        return _normalize
    except Exception as e:
        print(f"[WARN] Failed to build wetext normalizer: {e}", flush=True)
        return None


def build_llm_normalizer(
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> Optional[Callable[[str], str]]:
    """根据用户提供的 API 配置构建一个 llm_tn 归一化器（可调用对象）。

    llm_tn（``llm_tn/text_normalizer.py``）通过 LLM 实现多语种 TN，只要
    templates 齐全即可处理 wetext（仅中/英）不支持的语种（日、韩、俄等）。

    Args:
        api_url: LLM API 地址。为 ``None`` 时从环境变量 / ``.env`` 读取
            （``LLM_TN_API_URL``，llm_tn 不内置默认地址）。
        api_key: LLM API 密钥。为 ``None`` 时从环境变量 / ``.env`` 读取
            （``LLM_TN_API_KEY``，llm_tn 不内置默认密钥）。
        model: 使用的模型名。为 ``None`` 时从环境变量 / ``.env`` 读取
            （``LLM_TN_MODEL``，llm_tn 不内置默认模型）。
        **kwargs: 透传给 ``llm_tn.TextNormalizer`` 的其他参数。

    Returns:
        一个以 ``text`` 为输入、返回归一化文本的可调用对象；若初始化失败
        返回 ``None``。
    """
    try:
        from fireredtts3.utils.llm_tn.text_normalizer import (
            TextNormalizer as LlmTextNormalizer,
        )

        init_kwargs = dict(kwargs)
        if api_url is not None:
            init_kwargs["api_url"] = api_url
        if api_key is not None:
            init_kwargs["api_key"] = api_key
        if model is not None:
            init_kwargs["model"] = model
        tn = LlmTextNormalizer(**init_kwargs)

        def _normalize(text: str, locale: Optional[str] = None) -> str:
            """调用 llm_tn.normalize；当提供了 locale 时覆盖其内部自动检测。"""
            if locale:
                return tn.normalize(text, locale=locale, auto_detect=False)
            return tn.normalize(text)

        return _normalize
    except Exception as e:
        print(f"[WARN] Failed to build llm_tn normalizer: {e}", flush=True)
        return None


# --------------------------------------------------------------------------- #
# 统一入口
# --------------------------------------------------------------------------- #
class TextNormalizer:
    """封装 初步清洗 -> 分句 -> 逐句按需 TN 的文本处理管线。

    处理顺序：
    1. ``clean``    -- 仅做初步清洗（utf-8 / markdown / emoji / 空白规整），
                        不做任何 TN。
    2. ``split``     -- 清洗后按语种分句。
    3. 逐句 TN      -- 对每个句子做文本归一化：

        * llm_tn（``llm_normalizer``）优先，可处理全部语种；其内部通过
          ``needs_normalization`` 判断，仅当句子确实需要 TN 时才发起 LLM 调用。
        * wetext（``wetext_normalizer``）作为中/英回退；耗时极短，不做
          "是否需要 TN" 的判断，直接归一化。
    """

    def __init__(
        self,
        fasttext_detector: Optional[Callable[[str], Optional[str]]] = None,
        llm_normalizer: Optional[Callable[[str], str]] = None,
        wetext_normalizer: Optional[Callable[[str], str]] = None,
    ):
        """初始化文本处理管线。

        Args:
            fasttext_detector: 语种判定器，接收文本返回 ``llm_tn`` 的 locale。
            llm_normalizer: 多语种文本归一化器（``llm_tn`` 的 ``normalize``）。
                内部自带 ``needs_normalization`` 判断，按需调用 LLM。
            wetext_normalizer: 中/英文本归一化器（``wetext`` 的 ``Normalizer``）。
                当 llm_normalizer 不可用时，对中/英文本直接归一化。
        """
        self.fasttext_detector = fasttext_detector
        self.llm_normalizer = llm_normalizer
        self.wetext_normalizer = wetext_normalizer

    def clean(self, text: str) -> str:
        """仅做初步清洗，不做任何 TN。"""
        return clean_text(text)

    def _tn(self, text: str, lang_tag: Optional[str] = None) -> str:
        """对单个句子做 TN。

        TN 回退链：llm_tn（全部语种） → wetext（仅中/英）→ 原文。

        当提供了 ``lang_tag``（如 ``"Russian"``），wetext 分支只会对中/英语种
        调用 wetext；非中/英时直接返回原文（只做空格清理），从而避免无
        llm_tn 时俄语、日语等被错误地送进英文 TN。

        TN 之后统一清理汉字之间、汉字与英文/数字之间的多余空格。
        """
        if self.llm_normalizer is not None:
            locale = lang_tag_to_locale(lang_tag) if lang_tag else None
            try:
                return clean_tn_spaces(self.llm_normalizer(text, locale=locale))
            except Exception as e:
                print(f"[WARN] llm_tn normalization failed, fallback to raw text: {e}", flush=True)
                return clean_tn_spaces(text)
        # 没有 llm_tn 时，只有中/英/方言/粤语才走 wetext
        if self.wetext_normalizer is not None:
            if lang_tag is None:
                # 未提供语种 -> 按文本内容自动判定
                lang_tag = detect_language(text, self.fasttext_detector)
            _can_wetext = (
                lang_tag in ("Chinese", "English")
                or lang_tag == "Cantonese"
                or lang_tag.startswith("ZH_")
            )
            if _can_wetext:
                try:
                    return clean_tn_spaces(self.wetext_normalizer(text))
                except Exception as e:
                    print(f"[WARN] wetext normalization failed, fallback to raw text: {e}", flush=True)
                    return clean_tn_spaces(text)
        return clean_tn_spaces(text)

    def split(
        self,
        text: str,
        lang: str = "zh",
        tokenize: Optional[Callable[[str], List[str]]] = None,
        do_tn: bool = True,
        lang_tag: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """清洗后分句，并对每个句子逐句做 TN（默认开启）。

        Args:
            text: 输入文本。
            lang: 分句所用的语种（zh 按字符，其他按 token）。
            tokenize: 分句长度计算用的 tokenizer（非 zh 时使用）。
            do_tn: 是否对每个句子执行 TN。为 ``False`` 时仅分句不做归一化。
            lang_tag: 若已知语种 tag（如 ``"Russian"``），传给 llm_tn 覆盖自动检测。
            **kwargs: 透传给 ``split_paragraph`` 的其他参数。
        """
        text = self.clean(text)
        if not text:
            return []
        utts = split_paragraph(text, tokenize=tokenize, lang=lang, **kwargs)
        if not do_tn:
            return utts
        return [self._tn(u, lang_tag=lang_tag) for u in utts]

    def detect_lang(self, text: str) -> str:
        """判定文本语种（返回 lang tag）。"""
        return detect_language(text, self.fasttext_detector)


if __name__ == "__main__":
    wetext_normalizer = build_wetext_normalizer()
    llm_normalizer = None # build_llm_normalizer()
    tn = TextNormalizer(
        wetext_normalizer=wetext_normalizer,
        llm_normalizer=llm_normalizer,
    )
    samples = [
        "**你好，世界！** 今天天气很好。我们一起去公园散步吧。",
        "バスケがしたいいです",
        # ---- Qwen-Audio-3.0-TTS Long-text Generation 样例 ----
        "As she passed through the crowd of squires and yeomen who already filled the lower end of the vast apartment a scrap of paper was thrust into her hand which she received almost unconsciously and continued to hold without examining its contents. The assurance that she possessed some friend in this awful assembly gave her courage to look around and to mark into whose presence she had been conducted. She gazed accordingly upon a scene which might well have struck terror into a bolder heart than hers. On an elevated seat at the upper end of the great hall directly before the accused sat the grand master of the temple in full and ample robes of flowing white holding in his hand the mystic staff which bore the symbol of the order. At his feet was placed a table occupied by two scribes whose duty it was to record the proceedings of the day. Their chairs were black and formed a marked contrast to the warlike appearance of the knights who attended the solemn gathering. The preceptors of whom there were four present occupied seats behind their superiors and behind them stood the esquires of the order robed in white",
        "天色渐渐暗下来的时候，我把最后一件旧毛衣叠好，轻轻放进那只褪色的帆布包里。门锁轻轻一扣，楼道的灯光跟着亮了又灭，像极了这些年里头起起落落的日常。我站在台阶上深吸了口气，冷空气顺着鼻腔往里钻，带着点泥土与枯叶的腥气。其实心里头早就有了准备，只是真到了要转身的那一刻，脚步还是忍不住慢了半拍。风掠过树梢发出沙沙的响动，像是在劝我别回头。我知道这一走，有些人就再也碰不着了，可生活原本就是这样，聚散从来不讲道理，只能顺着岁月的河水流向该去的地方。 往前走的时候，脑海里不断闪过些零碎画面，那间住了好多年的小屋，窗台上总是晒着半干的衣裳，厨房的排烟罩一到做饭时分就会嗡嗡作响。那时候总觉得日子长得很，怎么熬也熬不到头，如今真要离开了，反倒觉得光阴像手里的细沙，攥得越紧漏得越快。我伸手摸了摸口袋里那张没带走的旧车票，纸边已经起了毛。其实送行的人并不多，大家都默契地没有多问什么，只说了句保重。我也学着他们的样子笑了笑，把那些没说出口的牵挂都咽进了肚子里。人嘛，总要学会一个人把情绪消化掉，等到夜深人静的时候再慢慢拿出来晾干。 拐过街角，远处的路灯渐渐亮起来，映在潮湿的地面上泛起一层柔和的光晕。",
        # ---- Qwen-Audio-3.0-TTS Text Normalization 样例 ----
        # EN
        "Sales promo: Buy 3 items, get Chapter XII free!",
        "The shadow puppet show begins promptly at 19:30 and ends at 21:15.",
        "Ventilation rate must stay between 0.85 and 1.25 air changes per hour.",
        "Grant awarded: €1.2M for cryo-EM infrastructure",
        "The grant covers $12K for lab materials and $3.5M for equipment.",
        "Stir gently until pH = 7.0 ± 0.2, then cap.",
        # ZH
        "今日体测数据：静息心率62.4次/分，体脂率18.7%，深蹲最大负重95.3公斤，跑步机耐力测试成绩12.8分钟。",
        "置信区间估计中，μ的95%CI为12.3～15.7，σ²的90%CI是8.1—11.4，而p值范围设为0.01→0.05。",
        "本次支付成功率对比：微信 97:3，支付宝 96:4，云闪付 95:5，三者比例差异需重点关注。",
        "固件升级包已发布，兼容版本号v1.8.3与v2.1.0，需确认设备当前运行build 7892。",
        "特种设备检验员需持TSG Z8002证书上岗，现场须佩戴N95口罩及PPE防护装备。",
        "本次实验中，样品A的纯度为99.95%，样品B含杂质1.2‰，两者差值达98.75%。",
    ]
    for s in samples:
        print("=" * 60)
        print("INPUT    :", s)
        print("LANG     :", tn.detect_lang(s))
        cleaned = tn.clean(s)
        print("CLEAN    :", cleaned)
        # 分句（含逐句 TN）
        parts_tn = tn.split(s, do_tn=True)
        print("SPLIT+TN:")
        for i, u in enumerate(parts_tn):
            print(f"  [{i}] {u}")
