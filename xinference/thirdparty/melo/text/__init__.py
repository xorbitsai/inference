from .symbols import *


_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language, symbol_to_id=None):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    symbol_to_id_map = symbol_to_id if symbol_to_id else _symbol_to_id
    phones = [symbol_to_id_map[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids


def get_bert(norm_text, word2ph, language, device):
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert
    from .chinese_mix import get_bert_feature as zh_mix_en_bert
    from .spanish_bert import get_bert_feature as sp_bert
    from .french_bert import get_bert_feature as fr_bert
    from .korean import get_bert_feature as kr_bert

    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert, 'ZH_MIX_EN': zh_mix_en_bert, 
                          'FR': fr_bert, 'SP': sp_bert, 'ES': sp_bert, "KR": kr_bert}
    bert = lang_bert_func_map[language](norm_text, word2ph, device)
    return bert
