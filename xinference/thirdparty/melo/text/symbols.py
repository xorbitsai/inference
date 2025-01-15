# punctuation = ["!", "?", "…", ",", ".", "'", "-"]
punctuation = ["!", "?", "…", ",", ".", "'", "-", "¿", "¡"]
pu_symbols = punctuation + ["SP", "UNK"]
pad = "_"

# chinese
zh_symbols = [
    "E",
    "En",
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "f",
    "g",
    "h",
    "i",
    "i0",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "ir",
    "iu",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ong",
    "ou",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "ui",
    "un",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "w",
    "x",
    "y",
    "z",
    "zh",
    "AA",
    "EE",
    "OO",
]
num_zh_tones = 6

# japanese
ja_symbols = [
    "N",
    "a",
    "a:",
    "b",
    "by",
    "ch",
    "d",
    "dy",
    "e",
    "e:",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "i:",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "o:",
    "p",
    "py",
    "q",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "u:",
    "w",
    "y",
    "z",
    "zy",
]
num_ja_tones = 1

# English
en_symbols = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "V",
    "w",
    "y",
    "z",
    "zh",
]
num_en_tones = 4

# Korean
kr_symbols = ['ᄌ', 'ᅥ', 'ᆫ', 'ᅦ', 'ᄋ', 'ᅵ', 'ᄅ', 'ᅴ', 'ᄀ', 'ᅡ', 'ᄎ', 'ᅪ', 'ᄑ', 'ᅩ', 'ᄐ', 'ᄃ', 'ᅢ', 'ᅮ', 'ᆼ', 'ᅳ', 'ᄒ', 'ᄆ', 'ᆯ', 'ᆷ', 'ᄂ', 'ᄇ', 'ᄉ', 'ᆮ', 'ᄁ', 'ᅬ', 'ᅣ', 'ᄄ', 'ᆨ', 'ᄍ', 'ᅧ', 'ᄏ', 'ᆸ', 'ᅭ', '(', 'ᄊ', ')', 'ᅲ', 'ᅨ', 'ᄈ', 'ᅱ', 'ᅯ', 'ᅫ', 'ᅰ', 'ᅤ', '~', '\\', '[', ']', '/', '^', ':', 'ㄸ', '*']
num_kr_tones = 1

# Spanish
es_symbols = [
        "N",
        "Q",
        "a",
        "b",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "ɑ",
        "æ",
        "ʃ",
        "ʑ",
        "ç",
        "ɯ",
        "ɪ",
        "ɔ",
        "ɛ",
        "ɹ",
        "ð",
        "ə",
        "ɫ",
        "ɥ",
        "ɸ",
        "ʊ",
        "ɾ",
        "ʒ",
        "θ",
        "β",
        "ŋ",
        "ɦ",
        "ɡ",
        "r",
        "ɲ",
        "ʝ",
        "ɣ",
        "ʎ",
        "ˈ",
        "ˌ",
        "ː"
    ]
num_es_tones = 1

# French 
fr_symbols = [
    "\u0303",
    "œ",
    "ø",
    "ʁ",
    "ɒ",
    "ʌ",
    "ɜ",
    "ɐ"
]
num_fr_tones = 1

# German 
de_symbols = [
    "ʏ",
    "̩"
  ]
num_de_tones = 1

# Russian 
ru_symbols = [
    "ɭ",
    "ʲ",
    "ɕ",
    "\"",
    "ɵ",
    "^",
    "ɬ"
]
num_ru_tones = 1

# combine all symbols
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols + kr_symbols + es_symbols + fr_symbols + de_symbols + ru_symbols))
symbols = [pad] + normal_symbols + pu_symbols
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]

# combine all tones
num_tones = num_zh_tones + num_ja_tones + num_en_tones + num_kr_tones + num_es_tones + num_fr_tones + num_de_tones + num_ru_tones

# language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2, "ZH_MIX_EN": 3, 'KR': 4, 'ES': 5, 'SP': 5 ,'FR': 6}
num_languages = len(language_id_map.keys())

language_tone_start_map = {
    "ZH": 0,
    "ZH_MIX_EN": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
    'KR': num_zh_tones + num_ja_tones + num_en_tones,
    "ES": num_zh_tones + num_ja_tones + num_en_tones + num_kr_tones,
    "SP": num_zh_tones + num_ja_tones + num_en_tones + num_kr_tones,
    "FR": num_zh_tones + num_ja_tones + num_en_tones + num_kr_tones + num_es_tones,
}

if __name__ == "__main__":
    a = set(zh_symbols)
    b = set(en_symbols)
    print(sorted(a & b))
