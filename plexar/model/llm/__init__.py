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


def install():
    from plexar.model.llm.core import LlamaCppModel

    from .. import MODEL_SPECS, ModelSpec
    from .vicuna import VicunaUncensoredGgml
    from .wizardlm import WizardlmGgml

    MODEL_SPECS.append(
        ModelSpec(
            name="baichuan",
            n_parameters_in_billions=7,
            fmt="ggml",
            quantization="q4_0",
            url="https://huggingface.co/TheBloke/baichuan-llama-7B-GGML/resolve/main/baichuan-llama-7b.ggmlv3.q4_0.bin",
            cls=LlamaCppModel,
        )
    )

    MODEL_SPECS.append(
        ModelSpec(
            name="wizardlm",
            n_parameters_in_billions=7,
            fmt="ggml",
            quantization="q4_0",
            url="https://huggingface.co/TheBloke/WizardLM-7B-V1.0-Uncensored-GGML/resolve/main/wizardlm-7b-v1.0-uncensored.ggmlv3.q4_0.bin",
            cls=WizardlmGgml,
        ),
    )

    MODEL_SPECS.append(
        ModelSpec(
            name="vicuna-uncensored",
            n_parameters_in_billions=7,
            fmt="ggml",
            quantization="q4_0",
            url="https://huggingface.co/vicuna/ggml-vicuna-7b-1.1/blob/main/ggml-vic7b-uncensored-q4_0.bin",
            cls=VicunaUncensoredGgml,
        ),
    )
