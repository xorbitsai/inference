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

MODEL_HUB_HUGGING_FACE = "Hugging Face"
MODEL_HUB_MODELSCOPE = "ModelScope"

def main():
    template_dir = '../templates' 
    env = Environment(loader=FileSystemLoader(template_dir))

    with open('../../xinference/model/llm/llm_family.json', 'r') as model_file:
        models = json.load(model_file)

        model_by_names = { m['model_name']: m for m in models}
        model_scope_file = open('../../xinference/model/llm/llm_family_modelscope.json')
        models_modelscope = json.load(model_scope_file)
        model_by_names_modelscope = { m['model_name']: m for m in models_modelscope}

        sorted_models = []
        output_dir = './models/builtin/llm'
        os.makedirs(output_dir, exist_ok=True)

        for model_name in sorted(model_by_names, key=str.lower):

            model = model_by_names[model_name]
            sorted_models.append(model)

            for model_spec in model['model_specs']:
                model_spec['model_hubs'] = [{                                                                  
                    'name': MODEL_HUB_HUGGING_FACE, 
                    'url': f"https://huggingface.co/{model_spec['model_id']}"
                }]

            # manual merge
            if model_name in model_by_names_modelscope.keys():

                def get_unique_id(spec):
                    return spec['model_format'] + '-' + str(spec['model_size_in_billions'])

                model_by_ids_modelscope = {get_unique_id(s) : s for s in model_by_names_modelscope[model_name]['model_specs']}

                for model_spec in model['model_specs']:
                    spec_id = get_unique_id(model_spec)
                    if spec_id in model_by_ids_modelscope.keys():
                        model_spec['model_hubs'].append({
                            'name': MODEL_HUB_MODELSCOPE,
                            'url': f"https://modelscope.cn/models/{model_by_ids_modelscope[spec_id]['model_id']}"
                        })

            rendered = env.get_template('llm.rst.jinja').render(model)
            output_file_path = os.path.join(output_dir, f"{model['model_name'].lower()}.rst")
            with open(output_file_path, 'w') as output_file:
                output_file.write(rendered)
                print(output_file_path)

        index_file_path = os.path.join(output_dir, "index.rst")
        with open(index_file_path, "w") as file:
            rendered_index = env.get_template('llm_index.rst.jinja').render(models=sorted_models)
            file.write(rendered_index)


    with open('../../xinference/model/embedding/model_spec.json', 'r') as file:
        models = json.load(file)

        model_by_names = { m['model_name']: m for m in models}
        model_scope_file = open('../../xinference/model/embedding/model_spec_modelscope.json')
        models_modelscope = json.load(model_scope_file)

        model_by_names_modelscope = { s['model_name']: s for s in models_modelscope}


        sorted_models = []
        output_dir = './models/builtin/embedding'
        os.makedirs(output_dir, exist_ok=True)

        for model_name in sorted(model_by_names, key=str.lower):
            model = model_by_names[model_name]

            sorted_models.append(model)

            model['model_hubs'] = [
                {
                    'name': MODEL_HUB_HUGGING_FACE, 
                    'url': f"https://huggingface.co/{model['model_id']}"
                }
            ]

            # manual merge
            if model['model_name'] in model_by_names_modelscope.keys():
                model_id_modelscope = model_by_names_modelscope[model['model_name']]['model_id']
                model['model_hubs'].append({
                    'name': MODEL_HUB_MODELSCOPE,
                    'url': f"https://modelscope.cn/models/{model_id_modelscope}"
                })

            rendered = env.get_template('embedding.rst.jinja').render(model)
            output_file_path = os.path.join(output_dir, f"{model['model_name'].lower()}.rst")
            with open(output_file_path, 'w') as output_file:
                output_file.write(rendered)
                print(output_file_path)

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

    with open('../../xinference/model/image/model_spec.json', 'r') as file:
        models = json.load(file)

        sorted_models = sorted(models, key=lambda x: x['model_name'].lower())
        output_dir = './models/builtin/image'
        os.makedirs(output_dir, exist_ok=True)

        for model in sorted_models:
            available_controlnet = [cn["model_name"] for cn in model.get("controlnet", [])]
            if not available_controlnet:
                available_controlnet = None
            model["available_controlnet"] = available_controlnet
            rendered = env.get_template('image.rst.jinja').render(model)
            output_file_path = os.path.join(output_dir, f"{model['model_name'].lower()}.rst")
            with open(output_file_path, 'w') as output_file:
                output_file.write(rendered)

        index_file_path = os.path.join(output_dir, "index.rst")
        with open(index_file_path, "w") as file:
            rendered_index = env.get_template('image_index.rst.jinja').render(models=sorted_models)
            file.write(rendered_index)

    with open('../../xinference/model/audio/model_spec.json', 'r') as file:
        models = json.load(file)

        sorted_models = sorted(models, key=lambda x: x['model_name'].lower())
        output_dir = './models/builtin/audio'
        os.makedirs(output_dir, exist_ok=True)

        for model in sorted_models:
            rendered = env.get_template('audio.rst.jinja').render(model)
            output_file_path = os.path.join(output_dir, f"{model['model_name'].lower()}.rst")
            with open(output_file_path, 'w') as output_file:
                output_file.write(rendered)

        index_file_path = os.path.join(output_dir, "index.rst")
        with open(index_file_path, "w") as file:
            rendered_index = env.get_template('audio_index.rst.jinja').render(models=sorted_models)
            file.write(rendered_index)


if __name__ == "__main__":
    main()
