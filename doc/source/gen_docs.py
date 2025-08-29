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
import sys
from collections import defaultdict

from jinja2 import Environment, FileSystemLoader

# Mock engine libraries before importing xinference modules
def mock_engine_libraries():
    """Mock engine libraries to make them appear installed for documentation generation"""
    from types import ModuleType
    from importlib.machinery import ModuleSpec
    
    # Create mock vllm module
    vllm_mock = ModuleType('vllm')
    vllm_mock.__version__ = "1.0.0"  # Latest version for full feature support
    vllm_mock.__spec__ = ModuleSpec('vllm', None)
    vllm_mock.__file__ = "mock_vllm.py"
    
    # Create mock mlx module with core submodule
    
    mlx_mock = ModuleType('mlx')
    mlx_mock.__version__ = "0.1.0"
    mlx_mock.__spec__ = ModuleSpec('mlx', None)
    mlx_mock.__file__ = "mock_mlx.py"
    
    mlx_core_mock = ModuleType('mlx.core')
    mlx_core_mock.__spec__ = ModuleSpec('mlx.core', None)
    mlx_core_mock.__file__ = "mock_mlx_core.py"
    # Add required attributes for xoscar serialization
    mlx_core_mock.array = type('MockArray', (), {})
    mlx_mock.core = mlx_core_mock
    
    # Create mock lmdeploy module  
    lmdeploy_mock = ModuleType('lmdeploy')
    lmdeploy_mock.__version__ = "0.6.0"
    lmdeploy_mock.__spec__ = ModuleSpec('lmdeploy', None)
    lmdeploy_mock.__file__ = "mock_lmdeploy.py"
    
    # Create mock sglang module
    sglang_mock = ModuleType('sglang')
    sglang_mock.__version__ = "0.3.0"
    sglang_mock.__spec__ = ModuleSpec('sglang', None)
    sglang_mock.__file__ = "mock_sglang.py"
    
    # Mock these modules in sys.modules
    sys.modules['vllm'] = vllm_mock
    sys.modules['mlx'] = mlx_mock
    sys.modules['mlx.core'] = mlx_core_mock
    sys.modules['lmdeploy'] = lmdeploy_mock
    sys.modules['sglang'] = sglang_mock

# Apply mocking before importing xinference modules
mock_engine_libraries()

from xinference.model.llm.llm_family import SUPPORTED_ENGINES, check_engine_by_spec_parameters
from xinference.model.llm.vllm.core import VLLM_INSTALLED, VLLM_SUPPORTED_MODELS, VLLM_SUPPORTED_CHAT_MODELS

# Additional mocking for platform/hardware checks in documentation generation
def mock_platform_checks():
    """Mock platform and hardware checks for documentation generation"""
    from unittest.mock import patch
    
    # Mock vLLM platform checks
    import xinference.model.llm.vllm.core as vllm_core
    vllm_core.VLLMModel._is_linux = lambda: True
    vllm_core.VLLMModel._has_cuda_device = lambda: True
    vllm_core.VLLMChatModel._is_linux = lambda: True
    vllm_core.VLLMChatModel._has_cuda_device = lambda: True
    vllm_core.VLLMVisionModel._is_linux = lambda: True
    vllm_core.VLLMVisionModel._has_cuda_device = lambda: True
    
    # Mock SGLang platform checks if available
    try:
        import xinference.model.llm.sglang.core as sglang_core
        sglang_core.SGLANGModel._is_linux = lambda: True
        sglang_core.SGLANGModel._has_cuda_device = lambda: True
        sglang_core.SGLANGChatModel._is_linux = lambda: True
        sglang_core.SGLANGChatModel._has_cuda_device = lambda: True
        sglang_core.SGLANGVisionModel._is_linux = lambda: True
        sglang_core.SGLANGVisionModel._has_cuda_device = lambda: True
    except ImportError:
        pass
    
    # Mock LMDEPLOY platform checks if available
    try:
        import xinference.model.llm.lmdeploy.core as lmdeploy_core
        lmdeploy_core.LMDeployModel._is_linux = lambda: True
        lmdeploy_core.LMDeployModel._has_cuda_device = lambda: True
        lmdeploy_core.LMDeployChatModel._is_linux = lambda: True
        lmdeploy_core.LMDeployChatModel._has_cuda_device = lambda: True
    except ImportError:
        pass

mock_platform_checks()

# Re-register engines with mocked platform checks
from xinference.model.llm import generate_engine_config_by_model_family
from xinference.model.llm.llm_family import BUILTIN_LLM_FAMILIES, LLM_ENGINES

# Clear existing engine configurations
LLM_ENGINES.clear()

# Re-register all model families with mocked platform checks
for family in BUILTIN_LLM_FAMILIES:
    generate_engine_config_by_model_family(family)

MODEL_HUB_HUGGING_FACE = "Hugging Face"
MODEL_HUB_MODELSCOPE = "ModelScope"


def gen_vllm_models():
    prefix_to_models = defaultdict(list)
    for model in VLLM_SUPPORTED_MODELS + VLLM_SUPPORTED_CHAT_MODELS:
        prefix = model.split('-', 1)[0]
        prefix_to_models[prefix].append(model)
    return [list(v) for _, v in prefix_to_models.items()]


def get_metrics_from_url(metrics_url):
    from prometheus_client.parser import text_string_to_metric_families
    import requests

    metrics = requests.get(metrics_url).content
    result = []
    for family in text_string_to_metric_families(metrics.decode("utf-8")):
        result.append({
            "name": family.name,
            "type": family.type,
            "help": family.documentation,
        })
    return result

def main():
    template_dir = '../templates' 
    env = Environment(loader=FileSystemLoader(template_dir))

    with open('../../xinference/model/llm/llm_family.json', 'r') as model_file:
        models = json.load(model_file)

        model_by_names = { m['model_name']: m for m in models}

        sorted_models = []
        output_dir = './models/builtin/llm'
        os.makedirs(output_dir, exist_ok=True)
        current_files = {f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))}

        for model_name in sorted(model_by_names, key=str.lower):

            model = model_by_names[model_name]
            sorted_models.append(model)

            for model_spec in model['model_specs']:
                model_spec['model_hubs'] = []
                
                # Process different model sources
                if 'model_src' in model_spec:
                    # Handle new model_src structure
                    if 'huggingface' in model_spec['model_src']:
                        hf_src = model_spec['model_src']['huggingface']
                        model_spec['model_hubs'].append({
                            'name': MODEL_HUB_HUGGING_FACE,
                            'url': f"https://huggingface.co/{hf_src['model_id']}"
                        })
                        # Set model_id and quantizations for template compatibility
                        model_spec['model_id'] = hf_src['model_id']
                        model_spec['quantizations'] = hf_src['quantizations']
                        quantizations = hf_src['quantizations']
                    
                    if 'modelscope' in model_spec['model_src']:
                        ms_src = model_spec['model_src']['modelscope']
                        model_spec['model_hubs'].append({
                            'name': MODEL_HUB_MODELSCOPE,
                            'url': f"https://modelscope.cn/models/{ms_src['model_id']}"
                        })
                        
                    # If only modelscope exists and no huggingface, use modelscope data
                    if 'modelscope' in model_spec['model_src'] and 'huggingface' not in model_spec['model_src']:
                        ms_src = model_spec['model_src']['modelscope']
                        model_spec['model_id'] = ms_src['model_id']
                        model_spec['quantizations'] = ms_src['quantizations']
                        quantizations = ms_src['quantizations']
                else:
                    # Fallback for old format if still exists
                    model_spec['model_hubs'].append({
                        'name': MODEL_HUB_HUGGING_FACE,
                        'url': f"https://huggingface.co/{model_spec['model_id']}"
                    })
                    quantizations = model_spec.get('quantizations', [])

                # model engines
                engines = []
                for engine in SUPPORTED_ENGINES:
                    for quantization in quantizations:
                        size = model_spec['model_size_in_billions']
                        if isinstance(size, str) and '_' not in size:
                            size = int(size)
                        try:
                            check_engine_by_spec_parameters(engine, model_name, model_spec['model_format'],
                                                            size, quantization)
                        except ValueError:
                            continue
                        else:
                            engines.append(engine)
                model_spec['engines'] = sorted(list(set(engines)), reverse=True)

            rendered = env.get_template('llm.rst.jinja').render(model)
            output_file_name = f"{model['model_name'].lower()}.rst"
            if output_file_name in current_files:
                current_files.remove(output_file_name)
            output_file_path = os.path.join(output_dir, output_file_name)
            with open(output_file_path, 'w') as output_file:
                output_file.write(rendered)
                print(output_file_path)

        if current_files:
            for f in current_files:
                print(f"remove {f}")
                os.remove(os.path.join(output_dir, f))

        index_file_path = os.path.join(output_dir, "index.rst")
        with open(index_file_path, "w") as file:
            rendered_index = env.get_template('llm_index.rst.jinja').render(models=sorted_models)
            file.write(rendered_index)


    with open('../../xinference/model/embedding/model_spec.json', 'r') as file:
        models = json.load(file)

        model_by_names = { m['model_name']: m for m in models}

        sorted_models = []
        output_dir = './models/builtin/embedding'
        os.makedirs(output_dir, exist_ok=True)

        for model_name in sorted(model_by_names, key=str.lower):
            model = model_by_names[model_name]

            sorted_models.append(model)

            model['model_hubs'] = []
            
            # Process model specs for new model_src structure
            if 'model_specs' in model and model['model_specs']:
                model_spec = model['model_specs'][0]  # Use first spec for model hubs
                if 'model_src' in model_spec:
                    if 'huggingface' in model_spec['model_src']:
                        hf_src = model_spec['model_src']['huggingface']
                        model['model_hubs'].append({
                            'name': MODEL_HUB_HUGGING_FACE,
                            'url': f"https://huggingface.co/{hf_src['model_id']}"
                        })
                        # Set model_id for template compatibility (prefer huggingface)
                        model['model_id'] = hf_src['model_id']
                    
                    if 'modelscope' in model_spec['model_src']:
                        ms_src = model_spec['model_src']['modelscope']
                        model['model_hubs'].append({
                            'name': MODEL_HUB_MODELSCOPE,
                            'url': f"https://modelscope.cn/models/{ms_src['model_id']}"
                        })
                        # Only set modelscope model_id if no huggingface exists
                        if 'huggingface' not in model_spec['model_src']:
                            model['model_id'] = ms_src['model_id']
                else:
                    # Fallback for old format
                    model_id = model_spec.get('model_id', model.get('model_id', ''))
                    model['model_id'] = model_id
                    model['model_hubs'].append({
                        'name': MODEL_HUB_HUGGING_FACE,
                        'url': f"https://huggingface.co/{model_id}"
                    })
            else:
                # Fallback for very old format
                if 'model_id' in model:
                    model['model_hubs'].append({
                        'name': MODEL_HUB_HUGGING_FACE,
                        'url': f"https://huggingface.co/{model['model_id']}"
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
            # Initialize model_hubs list
            model['model_hubs'] = []
            
            # Process model specs for new model_src structure
            model_spec = model['model_specs'][0]  # Use first spec for model hubs
            if 'model_src' in model_spec:
                if 'huggingface' in model_spec['model_src']:
                    hf_src = model_spec['model_src']['huggingface']
                    model['model_hubs'].append({
                        'name': MODEL_HUB_HUGGING_FACE,
                        'url': f"https://huggingface.co/{hf_src['model_id']}"
                    })
                    # Set model_id for template compatibility (prefer huggingface)
                    model['model_id'] = hf_src['model_id']
                
                if 'modelscope' in model_spec['model_src']:
                    ms_src = model_spec['model_src']['modelscope']
                    model['model_hubs'].append({
                        'name': MODEL_HUB_MODELSCOPE,
                        'url': f"https://modelscope.cn/models/{ms_src['model_id']}"
                    })
                    # Only set modelscope model_id if no huggingface exists
                    if 'huggingface' not in model_spec['model_src']:
                        model['model_id'] = ms_src['model_id']
            
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
            # Process model_src for template compatibility
            if 'model_src' in model:
                if 'huggingface' in model['model_src']:
                    hf_src = model['model_src']['huggingface']
                    model['model_id'] = hf_src['model_id']
                    # Handle GGUF related fields
                    if 'gguf_model_id' in hf_src:
                        model['gguf_model_id'] = hf_src['gguf_model_id']
                    if 'gguf_quantizations' in hf_src:
                        model['gguf_quantizations'] = ", ".join(hf_src['gguf_quantizations'])
                    # Handle Lightning related fields
                    if 'lightning_model_id' in hf_src:
                        model['lightning_model_id'] = hf_src['lightning_model_id']
                    if 'lightning_versions' in hf_src:
                        model['lightning_versions'] = ", ".join(hf_src['lightning_versions'])
                elif 'modelscope' in model['model_src']:
                    model['model_id'] = model['model_src']['modelscope']['model_id']
            
            available_controlnet = [cn["model_name"] for cn in model.get("controlnet", [])]
            if not available_controlnet:
                available_controlnet = None
            model["available_controlnet"] = available_controlnet
            model["model_ability"] = ', '.join(model.get("model_ability"))
            
            # Ensure gguf_quantizations is properly formatted (fallback for old format)
            if "gguf_quantizations" not in model:
                model["gguf_quantizations"] = ", ".join(model.get("gguf_quantizations", []))
            
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
            # Process model_src for template compatibility
            if 'model_src' in model:
                if 'huggingface' in model['model_src']:
                    model['model_id'] = model['model_src']['huggingface']['model_id']
                elif 'modelscope' in model['model_src']:
                    model['model_id'] = model['model_src']['modelscope']['model_id']
            
            rendered = env.get_template('audio.rst.jinja').render(model)
            output_file_path = os.path.join(output_dir, f"{model['model_name'].lower()}.rst")
            with open(output_file_path, 'w') as output_file:
                output_file.write(rendered)

        index_file_path = os.path.join(output_dir, "index.rst")
        with open(index_file_path, "w") as file:
            rendered_index = env.get_template('audio_index.rst.jinja').render(models=sorted_models)
            file.write(rendered_index)

    with open('../../xinference/model/video/model_spec.json', 'r') as file:
        models = json.load(file)

        sorted_models = sorted(models, key=lambda x: x['model_name'].lower())
        output_dir = './models/builtin/video'
        os.makedirs(output_dir, exist_ok=True)

        for model in sorted_models:
            # Process model_src for template compatibility
            if 'model_src' in model:
                if 'huggingface' in model['model_src']:
                    model['model_id'] = model['model_src']['huggingface']['model_id']
                elif 'modelscope' in model['model_src']:
                    model['model_id'] = model['model_src']['modelscope']['model_id']
            
            model["model_ability"] = ', '.join(model.get("model_ability"))
            rendered = env.get_template('video.rst.jinja').render(model)
            output_file_path = os.path.join(output_dir, f"{model['model_name'].lower()}.rst")
            with open(output_file_path, 'w') as output_file:
                output_file.write(rendered)

        index_file_path = os.path.join(output_dir, "index.rst")
        with open(index_file_path, "w") as file:
            rendered_index = env.get_template('video_index.rst.jinja').render(models=sorted_models)
            file.write(rendered_index)

    if VLLM_INSTALLED:
        vllm_models = gen_vllm_models()
        groups = [', '.join("``%s``" % m for m in group) for group in vllm_models]
        vllm_model_str = '\n'.join('- %s' % group for group in groups)
        for fn in ['getting_started/installation.rst', 'user_guide/backends.rst']:
            with open(fn) as f:
                content = f.read()
            start_label = '.. vllm_start'
            end_label = '.. vllm_end'
            start = content.find(start_label) + len(start_label)
            end = content.find(end_label)
            new_content = content[:start] + '\n\n' + vllm_model_str + '\n' + content[end:]
            with open(fn, 'w') as f:
                f.write(new_content)

    try:
        output_dir = './user_guide'
        os.makedirs(output_dir, exist_ok=True)

        supervisor_metrics = get_metrics_from_url("http://127.0.0.1:9997/metrics")
        worker_metrics = get_metrics_from_url("http://127.0.0.1:9977/metrics")
        all_metrics = {"supervisor_metrics": supervisor_metrics, "worker_metrics": worker_metrics}
        rendered = env.get_template('metrics.jinja').render(all_metrics)
        output_file_path = os.path.join(output_dir, "metrics.rst")
        with open(output_file_path, 'w') as output_file:
            output_file.write(rendered)
    except Exception:
        print("Skip generate metrics doc, please start a local xinference server by: `xinference-local -mp 9977`.")


if __name__ == "__main__":
    main()
