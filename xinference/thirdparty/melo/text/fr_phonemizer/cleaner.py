"""Set of default text cleaners"""
# TODO: pick the cleaner for languages dynamically

import re
from .french_abbreviations import abbreviations_fr

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")


rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": ".",
    "…": ".",
    "$": ".",
    "“": "",
    "”": "",
    "‘": "",
    "’": "",
    "（": "",
    "）": "",
    "(": "",
    ")": "",
    "《": "",
    "》": "",
    "【": "",
    "】": "",
    "[": "",
    "]": "",
    "—": "",
    "～": "-",
    "~": "-",
    "「": "",
    "」": "",
    "¿" : "",
    "¡" : ""
}


def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    return replaced_text

def expand_abbreviations(text, lang="fr"):
    if lang == "fr":
        _abbreviations = abbreviations_fr
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text).strip()

def remove_punctuation_at_begin(text):
    return re.sub(r'^[,.!?]+', '', text)

def remove_aux_symbols(text):
    text = re.sub(r"[\<\>\(\)\[\]\"\«\»]+", "", text)
    return text


def replace_symbols(text, lang="en"):
    """Replace symbols based on the lenguage tag.

    Args:
      text:
       Input text.
      lang:
        Lenguage identifier. ex: "en", "fr", "pt", "ca".

    Returns:
      The modified text
      example:
        input args:
            text: "si l'avi cau, diguem-ho"
            lang: "ca"
        Output:
            text: "si lavi cau, diguemho"
    """
    text = text.replace(";", ",")
    text = text.replace("-", " ") if lang != "ca" else text.replace("-", "")
    text = text.replace(":", ",")
    if lang == "en":
        text = text.replace("&", " and ")
    elif lang == "fr":
        text = text.replace("&", " et ")
    elif lang == "pt":
        text = text.replace("&", " e ")
    elif lang == "ca":
        text = text.replace("&", " i ")
        text = text.replace("'", "")
    elif lang== "es":
        text=text.replace("&","y")
        text = text.replace("'", "")
    return text

def french_cleaners(text):
    """Pipeline for French text. There is no need to expand numbers, phonemizer already does that"""
    text = expand_abbreviations(text, lang="fr")
    # text = lowercase(text) # as we use the cased bert
    text = replace_punctuation(text)
    text = replace_symbols(text, lang="fr")
    text = remove_aux_symbols(text)
    text = remove_punctuation_at_begin(text)
    text = collapse_whitespace(text)
    text = re.sub(r'([^\.,!\?\-…])$', r'\1.', text)
    return text

