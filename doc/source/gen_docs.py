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

import json
import os
from jinja2 import Environment, FileSystemLoader


def main():
    template_dir = '../templates' 
    env = Environment(loader=FileSystemLoader(template_dir))

    with open('../../xinference/model/llm/llm_family.json', 'r') as model_file:
        models = json.load(model_file)

        model_scope_file = open('../../xinference/model/llm/llm_family_modelscope.json')

        model_scope_models = json.load(model_scope_file)
        print(model_scope_models)
        models.update(model_scope_models)

        sorted_models = sorted(models, key=lambda x: x['model_name'].lower())
        output_dir = './models/builtin/llm'
        os.makedirs(output_dir, exist_ok=True)

        for model in sorted_models:
            rendered = env.get_template('llm.rst.jinja').render(model)
            output_file_path = os.path.join(output_dir, f"{model['model_name'].lower()}.rst")
            with open(output_file_path, 'w') as output_file:
                output_file.write(rendered)

        index_file_path = os.path.join(output_dir, "index.rst")
        with open(index_file_path, "w") as file:
            
            rendered_index = env.get_template('llm_index.rst.jinja').render(models=sorted_models)
            file.write(rendered_index)


    with open('../../xinference/model/embedding/model_spec.json', 'r') as file:
        models = json.load(file)

        sorted_models = sorted(models, key=lambda x: x['model_name'].lower())
        output_dir = './models/builtin/embedding'
        os.makedirs(output_dir, exist_ok=True)

        for model in sorted_models:
            rendered = env.get_template('embedding.rst.jinja').render(model)
            output_file_path = os.path.join(output_dir, f"{model['model_name'].lower()}.rst")
            with open(output_file_path, 'w') as output_file:
                output_file.write(rendered)

        index_file_path = os.path.join(output_dir, "index.rst")
        with open(index_file_path, "w") as file:
            
            rendered_index = env.get_template('embedding_index.rst.jinja').render(models=sorted_models)
            file.write(rendered_index)

    with open('../../xinference/model/rerank/model_spec.json', 'r') as file:
        models = json.load(file)

        sorted_models = sorted(models, key=lambda x: x['model_name'].lower())
        output_dir = './models/builtin/rerank'
        os.makedirs(output_dir, exist_ok=True)

        for model in sorted_models:
            rendered = env.get_template('rerank.rst.jinja').render(model)
            output_file_path = os.path.join(output_dir, f"{model['model_name'].lower()}.rst")
            with open(output_file_path, 'w') as output_file:
                output_file.write(rendered)

        index_file_path = os.path.join(output_dir, "index.rst")
        with open(index_file_path, "w") as file:
            
            rendered_index = env.get_template('rerank_index.rst.jinja').render(models=sorted_models)
            file.write(rendered_index)

if __name__ == "__main__":
    main()
