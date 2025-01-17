import re

# List of (regular expression, replacement) pairs for abbreviations in english:
abbreviations_en = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]

def expand_abbreviations(text, lang="en"):
    if lang == "en":
        _abbreviations = abbreviations_en
    else:
        raise NotImplementedError()
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text