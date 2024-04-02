from ..utils import get_language_from_model_id


def test_get_language_from_model_id():
    model_id = "BAAI/bge-large-zh-v1.5"
    assert get_language_from_model_id(model_id) == "zh"

    model_id = "BAAI/bge-large-base-v1.5"
    assert get_language_from_model_id(model_id) == "en"

    model_id = "google-bert/bert-base-multilingual-cased"
    assert get_language_from_model_id(model_id) == "zh"

    model_id = "jinaai/jina-embeddings-v2-base-en"
    assert get_language_from_model_id(model_id) == "en"

    model_id = "jinaai/jina-embeddings-v2-base-es"
    # now only support zh and en, if it is not chinese, then en, even the language is specified as es
    assert get_language_from_model_id(model_id) == "en"

    model_id = "bge-large-zh-v1.5"
    # wrong model id will cause the en is returned
    assert get_language_from_model_id(model_id) == "en"

    model_id = "BAAI/newtype/bge-large-zh-v1.5"
    # wrong model id format, return en
    assert get_language_from_model_id(model_id) == "en"

    model_id = ""
    assert get_language_from_model_id(model_id) == "en"
