from .cleaner import spanish_cleaners
from .gruut_wrapper import Gruut

def es2ipa(text):
    e = Gruut(language="es-es", keep_puncs=True, keep_stress=True, use_espeak_phonemes=True)
    # text = spanish_cleaners(text)
    phonemes = e.phonemize(text, separator="")
    return phonemes


if __name__ == '__main__':
  print(es2ipa('¿Y a quién echaría de menos, en el mundo si no fuese a vos?'))