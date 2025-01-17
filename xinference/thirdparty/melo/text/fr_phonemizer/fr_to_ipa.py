from .cleaner import french_cleaners
from .gruut_wrapper import Gruut


def remove_consecutive_t(input_str):
    result = []
    count = 0

    for char in input_str:
        if char == 't':
            count += 1
        else:
            if count < 3:  
                result.extend(['t'] * count)
            count = 0
            result.append(char)

    if count < 3:
        result.extend(['t'] * count)

    return ''.join(result)

def fr2ipa(text):
    e = Gruut(language="fr-fr", keep_puncs=True, keep_stress=True, use_espeak_phonemes=True)
    # text = french_cleaners(text)
    phonemes = e.phonemize(text, separator="")
    # print(phonemes)
    phonemes = remove_consecutive_t(phonemes)
    # print(phonemes)
    return phonemes