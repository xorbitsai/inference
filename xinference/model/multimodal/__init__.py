import codecs
import json
import os

from .core import MultimodalModelSpec

_model_spec_json = os.path.join(os.path.dirname(__file__), "model_spec.json")
BUILTIN_MULTIMODAL_MODELS = dict(
    (spec["model_name"], MultimodalModelSpec(**spec))
    for spec in json.load(codecs.open(_model_spec_json, "r", encoding="utf-8"))
)
del _model_spec_json
