import re
import inflect

'''
number normalization for English
'''

_inflect = inflect.engine()

# 移除逗号：1,000 → 1000
_comma_number_re = re.compile(r"([0-9][0-9,]+[0-9])")

# 百分比：支持负数（-25.5%）
_percent_number_re = re.compile(r"(-?[0-9.,]*[0-9]+)%")

# 英镑：£-150 → minus one hundred fifty pounds
_pounds_re = re.compile(r"£(-?[0-9,]*[0-9]+(?:\.[0-9]+)?)")

# 美元：$-12.99
_dollars_re = re.compile(r"\$(-?[0-9.,]*[0-9]+(?:\.[0-9]+)?)")

# 分数：3/4
_fraction_re = re.compile(r"([0-9]+)\/([0-9]+)")

# 序数：1st, 2nd
_ordinal_re = re.compile(r"\b[0-9]+(st|nd|rd|th)\b")

# 普通数字（最后处理）
_number_re = re.compile(r"\b-?[0-9]+(?:\.[0-9]+)?\b")

# 单位：10ms, 5.5 kg 等
_unit_re = re.compile(
    r'\b(-?\d+(?:\.\d+)?)\s*(ms|s|Hz|kHz|MHz|GHz|kb|mb|gb|tb|KB|MB|GB|TB|bps|kbps|Mbps|Gbps|cm|km|kg|V|A|W|°C|°F)\b',
    re.IGNORECASE
)

# 正则：支持 v2, v2.1, gpt-4, gpt-4.5
_version_re = re.compile(r'\b([a-zA-Z]+)([-]?)([0-9]+(?:\.[0-9]+)?)\b')

# 空白字符
_whitespace_re = re.compile(r"\s+")

# 标点符号（可选）
_colon_re = re.compile(r":")


_unit_mapping = {
    'ms': 'milliseconds',
    's': 'seconds',
    'hz': 'hertz',
    'khz': 'kilohertz',
    'mhz': 'megahertz',
    'ghz': 'gigahertz',
    'kb': 'kilobytes',
    'mb': 'megabytes',
    'gb': 'gigabytes',
    'tb': 'terabytes',
    'kbps': 'kilobits per second',
    'mbps': 'megabits per second',
    'gbps': 'gigabits per second',
    'bps': 'bits per second',
    'cm': 'centimeters',
    'km': 'kilometers',
    'kg': 'kilograms',
    'v': 'volts',
    'a': 'amperes',
    'w': 'watts',
    '°c': 'degrees celsius',
    '°f': 'degrees fahrenheit',
}


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_colon(m):
    return " colon "


def _expand_unit(m):
    num_str = m.group(1)
    unit = m.group(2).lower()
    unit_word = _unit_mapping.get(unit, unit)

    try:
        is_negative = num_str.startswith('-')
        clean_num = num_str.lstrip('-')
        if not clean_num:
            clean_num = '0'

        if '.' in clean_num:
            parts = clean_num.split('.', 1)
            integer_part = parts[0] or '0'
            decimal_part = parts[1]

            # 检查是否全为数字
            if not integer_part.isdigit() or not decimal_part.isdigit():
                return f" {num_str} {unit} "

            integer_word = _inflect.number_to_words(int(integer_part), andword="") if integer_part != '0' else 'zero'
            decimal_words = " ".join(
                _inflect.number_to_words(int(d), andword="") for d in decimal_part if d.isdigit()
            )
            num_word = f"{integer_word} point {decimal_words}"
        else:
            if not clean_num.isdigit():
                return f" {num_str} {unit} "
            num_word = _inflect.number_to_words(int(clean_num), andword="")

        if is_negative:
            num_word = f"minus {num_word}"

        return f" {num_word} {unit_word} "
    except Exception:
        return f" {num_str} {unit} "


def _expand_percent(m):
    num_str = m.group(1)
    try:
        is_negative = num_str.startswith('-')
        clean_num = num_str.lstrip('-')
        if not clean_num:
            clean_num = '0'

        if '.' in clean_num:
            parts = clean_num.split('.', 1)
            integer_part = parts[0] or '0'
            decimal_part = parts[1]

            if not integer_part.isdigit() or not decimal_part.isdigit():
                return f" {num_str} percent "

            integer_word = _inflect.number_to_words(int(integer_part), andword="") if integer_part != '0' else 'zero'
            decimal_words = " ".join(
                _inflect.number_to_words(int(d), andword="") for d in decimal_part if d.isdigit()
            )
            num_word = f"{integer_word} point {decimal_words}"
        else:
            if not clean_num.isdigit():
                return f" {num_str} percent "
            num_word = _inflect.number_to_words(int(clean_num), andword="")

        if is_negative:
            num_word = f"minus {num_word}"

        return f" {num_word} percent "
    except Exception:
        return f" {num_str} percent "


def _expand_dollars(m):
    match = m.group(1)
    is_negative = match.startswith('-')
    clean_match = match.lstrip('-')
    if not clean_match:
        clean_match = '0'

    try:
        if '.' in clean_match:
            parts = clean_match.split('.', 1)
            integer_part = parts[0] or '0'
            decimal_part = parts[1]

            if not integer_part.isdigit() or not decimal_part.isdigit():
                return f" {clean_match} dollars "

            integer_word = _inflect.number_to_words(int(integer_part), andword="") if integer_part != '0' else 'zero'
            decimal_words = " ".join(
                _inflect.number_to_words(int(d), andword="") for d in decimal_part if d.isdigit()
            )
            num_word = f"{integer_word} point {decimal_words}"
        else:
            if not clean_match.isdigit():
                return f" {clean_match} dollars "
            num_word = _inflect.number_to_words(int(clean_match), andword="")

        if is_negative:
            num_word = f"minus {num_word}"

        value = float(clean_match)
        unit = "dollar" if abs(value) == 1.0 else "dollars"
        return f" {num_word} {unit} "
    except Exception:
        return f" {clean_match} dollars "


def _expand_pounds(m):
    num_str = m.group(1)
    is_negative = num_str.startswith('-')
    clean_num = num_str.lstrip('-')
    if not clean_num:
        clean_num = '0'

    try:
        if '.' in clean_num:
            parts = clean_num.split('.', 1)
            integer_part = parts[0] or '0'
            decimal_part = parts[1]

            if not integer_part.isdigit() or not decimal_part.isdigit():
                return f" {clean_num} pounds "

            integer_word = _inflect.number_to_words(int(integer_part), andword="") if integer_part != '0' else 'zero'
            decimal_words = " ".join(
                _inflect.number_to_words(int(d), andword="") for d in decimal_part if d.isdigit()
            )
            num_word = f"{integer_word} point {decimal_words}"
        else:
            if not clean_num.isdigit():
                return f" {clean_num} pounds "
            num_word = _inflect.number_to_words(int(clean_num), andword="")

        if is_negative:
            num_word = f"minus {num_word}"

        value = float(clean_num)
        unit = "pound" if abs(value) == 1.0 else "pounds"
        return f" {num_word} {unit} "
    except Exception:
        return f" {clean_num} pounds "


def fraction_to_words(numerator, denominator):
    try:
        if numerator == 1 and denominator == 2:
            return " one half "
        if numerator == 1 and denominator == 4:
            return " one quarter "
        if denominator == 2:
            plural = " half" if numerator == 1 else " halves"
            return f" {_inflect.number_to_words(numerator)}{plural} "
        if denominator == 4:
            plural = " quarter" if numerator == 1 else " quarters"
            return f" {_inflect.number_to_words(numerator)}{plural} "
        ordinal = _inflect.ordinal(_inflect.number_to_words(denominator))
        return f" {_inflect.number_to_words(numerator)} {ordinal} "
    except Exception:
        return f" {numerator} over {denominator} "


def _expand_fraction(m):
    try:
        numerator = int(m.group(1))
        denominator = int(m.group(2))
        return fraction_to_words(numerator, denominator)
    except Exception:
        return m.group(0)


def _expand_ordinal(m):
    try:
        # 提取纯数字部分
        num = int(re.sub(r"(st|nd|rd|th)", "", m.group(0)))
        word = _inflect.number_to_words(num)
        return f" {word} "
    except Exception:
        return m.group(0)


def _expand_number(m):
    num_str = m.group(0)
    try:
        is_negative = num_str.startswith('-')
        clean_num = num_str.lstrip('-')
        if not clean_num:
            clean_num = '0'

        if '.' in clean_num:
            parts = clean_num.split('.', 1)
            integer_part = parts[0]
            decimal_part = parts[1]

            if not integer_part:
                integer_part = '0'
            if not integer_part.isdigit() or not decimal_part.isdigit():
                return f" {num_str} "

            integer_word = _inflect.number_to_words(int(integer_part), andword="") if integer_part != '0' else 'zero'
            decimal_words = " ".join(
                _inflect.number_to_words(int(d), andword="") for d in decimal_part if d.isdigit()
            )
            num_word = f"{integer_word} point {decimal_words}"
        else:
            if not clean_num.isdigit():
                return f" {num_str} "
            num_word = _inflect.number_to_words(int(clean_num), andword="")

        if is_negative:
            num_word = f"minus {num_word}"

        return f" {num_word} "
    except Exception:
        return f" {num_str} "


def _expand_version(m):
    prefix = m.group(1)
    sep = m.group(2)
    num_str = m.group(3)

    try:
        if '.' in num_str:
            parts = num_str.split('.', 1)
            integer_part = parts[0]
            decimal_part = parts[1]

            if not integer_part.isdigit() or not decimal_part.isdigit():
                return m.group(0)

            integer_word = _inflect.number_to_words(int(integer_part))
            decimal_words = " ".join(
                _inflect.number_to_words(int(d)) for d in decimal_part if d.isdigit()
            )
            word = f"{integer_word} point {decimal_words}"
        else:
            if not num_str.isdigit():
                return m.group(0)
            word = _inflect.number_to_words(int(num_str))
    except Exception:
        return m.group(0)

    # 连字符替换为空格，如 gpt-4 → gpt four
    if sep == '-':
        return f"{prefix} {word}"
    # 直接拼接则加空格读出：v2 → v two
    return f"{prefix} {word}"


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_unit_re, _expand_unit, text)             # 带单位的量（含负数）
    text = re.sub(_pounds_re, _expand_pounds, text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_fraction_re, _expand_fraction, text)
    text = re.sub(_percent_number_re, _expand_percent, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_version_re, _expand_version, text)       # 新增：处理 v2, gpt-4
    text = re.sub(_number_re, _expand_number, text)         # 剩余孤立数字
    # text = re.sub(_colon_re, _expand_colon, text)         # 冒号（可选）
    text = re.sub(_whitespace_re, " ", text)
    return text.strip()
