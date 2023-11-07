# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Iterable, List, Union

from pydantic import validate_arguments
from pydantic.fields import ModelField


def get_pydantic_model_from_method(
    meth,
    exclude_fields: Iterable[str] = None,
    include_fields: Dict[str, ModelField] = None,
):
    f = validate_arguments(meth, config={"arbitrary_types_allowed": True})
    model = f.model
    model.__fields__.pop("self", None)
    model.__fields__.pop("args", None)
    model.__fields__.pop("kwargs", None)
    pydantic_private_keys = [
        key for key in model.__fields__.keys() if key.startswith("v__")
    ]
    for key in pydantic_private_keys:
        model.__fields__.pop(key)
    if exclude_fields is not None:
        for key in exclude_fields:
            model.__fields__.pop(key)
    if include_fields is not None:
        assert all(isinstance(field, ModelField) for field in include_fields.values())
        model.__fields__.update(include_fields)
    return model



def generate_pydantic_model(model, name=None):
    if model is None:
        return []
    code = [f"class {name or model.__name__}(BaseModel):"]
    for key, field in model.__fields__.items():
        code.append(f"    {key} = {repr(field)}".replace("type=", "type_="))
    code.append("\n\n")
    return code


def generate_dynamic_types():
    code = [
        "from typing import List, Optional, Union",
        "",
        "from llama_cpp import LlamaGrammar, LogitsProcessorList, StoppingCriteriaList",
        "from pydantic import BaseModel",
        "from pydantic.fields import ModelField",
        "from pydantic.typing import NoneType",
        "\n"
    ]
    try:
        from llama_cpp import Llama

        CreateCompletionLlamaCpp = get_pydantic_model_from_method(
            Llama.create_completion
        )
    except ImportError:
        CreateCompletionLlamaCpp = None
    code.extend(generate_pydantic_model(CreateCompletionLlamaCpp))
    return code


if __name__ == "__main__":
    import os
    dynamic_types_file = os.path.join(os.path.dirname(__file__), "dynamic_types.py")
    with open(dynamic_types_file, "w") as f:
        code = "\n".join(generate_dynamic_types())
        f.write(code)