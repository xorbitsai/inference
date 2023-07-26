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
    from .. import MODEL_FAMILIES, ModelFamily
    from .chatglm import ChatglmCppChatModel
    from .core import LlamaCppModel
    from .llama2 import Llama2ChatGgml
    from .orca import OrcaMiniGgml
    from .pytorch.baichuan import BaichuanPytorch, BaichuanPytorchChat
    from .pytorch.vicuna import VicunaCensoredPytorch
    from .vicuna import VicunaCensoredGgml
    from .wizardlm import WizardlmGgml

    baichuan_url_generator = lambda model_size, quantization: (
        f"https://huggingface.co/TheBloke/baichuan-llama-{model_size}B-GGML/resolve/main/"
        f"baichuan-llama-{model_size}b.ggmlv3.{quantization}.bin"
    )

    baichuan_url_raw_generator = lambda model_size, quantization: (
        f"https://huggingface.co/TheBloke/baichuan-llama-{model_size}B-GGML/raw/main/"
        f"baichuan-llama-{model_size}b.ggmlv3.{quantization}.bin"
    )

    MODEL_FAMILIES.append(
        ModelFamily(
            model_name="baichuan",
            model_format="ggmlv3",
            model_sizes_in_billions=[7],
            quantizations=[
                "q2_K",
                "q3_K_L",
                "q3_K_M",
                "q3_K_S",
                "q4_0",
                "q4_1",
                "q4_K_M",
                "q4_K_S",
                "q5_0",
                "q5_1",
                "q5_K_M",
                "q5_K_S",
                "q6_K",
                "q8_0",
            ],
            url_generator=baichuan_url_generator,
            rp_url_generator=baichuan_url_raw_generator,
            cls=LlamaCppModel,
        )
    )

    baichuan_chat_url_generator = lambda model_size, quantization: (
        f"https://huggingface.co/TheBloke/baichuan-vicuna-{model_size}B-GGML/resolve/main/"
        f"baichuan-vicuna-{model_size}b.ggmlv3.{quantization}.bin"
    )

    baichuan_chat_url_raw_generator = lambda model_size, quantization: (
        f"https://huggingface.co/TheBloke/baichuan-vicuna-{model_size}B-GGML/raw/main/"
        f"baichuan-vicuna-{model_size}b.ggmlv3.{quantization}.bin"
    )

    MODEL_FAMILIES.append(
        ModelFamily(
            model_name="baichuan-chat",
            model_format="ggmlv3",
            model_sizes_in_billions=[7],
            quantizations=[
                "q2_K",
                "q3_K_L",
                "q3_K_M",
                "q3_K_S",
                "q4_0",
                "q4_1",
                "q4_K_M",
                "q4_K_S",
                "q5_0",
                "q5_1",
                "q5_K_M",
                "q5_K_S",
                "q6_K",
                "q8_0",
            ],
            url_generator=baichuan_chat_url_generator,
            rp_url_generator=baichuan_chat_url_raw_generator,
            cls=VicunaCensoredGgml,
        )
    )

    wizardlm_v1_0_url_generator = lambda model_size, quantization: (
        f"https://huggingface.co/TheBloke/WizardLM-{model_size}B-V1.0-Uncensored-GGML/resolve/main/"
        f"wizardlm-{model_size}b-v1.0-uncensored.ggmlv3.{quantization}.bin"
    )

    wizardlm_v1_0_url_raw_generator = lambda model_size, quantization: (
        f"https://huggingface.co/TheBloke/WizardLM-{model_size}B-V1.0-Uncensored-GGML/raw/main/"
        f"wizardlm-{model_size}b-v1.0-uncensored.ggmlv3.{quantization}.bin"
    )

    MODEL_FAMILIES.append(
        ModelFamily(
            model_name="wizardlm-v1.0",
            model_sizes_in_billions=[7, 13, 33],
            model_format="ggmlv3",
            quantizations=[
                "q2_K",
                "q3_K_L",
                "q3_K_M",
                "q3_K_S",
                "q4_0",
                "q4_1",
                "q4_K_M",
                "q4_K_S",
                "q5_0",
                "q5_1",
                "q5_K_M",
                "q5_K_S",
                "q6_K",
                "q8_0",
            ],
            url_generator=wizardlm_v1_0_url_generator,
            rp_url_generator=wizardlm_v1_0_url_raw_generator,
            cls=WizardlmGgml,
        ),
    )

    wizardlm_v1_1_url_generator = lambda model_size, quantization: (
        f"https://huggingface.co/TheBloke/WizardLM-{model_size}B-V1.1-GGML/resolve/main/"
        f"wizardlm-{model_size}b-v1.1.ggmlv3.{quantization}.bin"
    )
    wizardlm_v1_1_url_raw_generator = lambda model_size, quantization: (
        f"https://huggingface.co/TheBloke/WizardLM-{model_size}B-V1.1-GGML/raw/main/"
        f"wizardlm-{model_size}b-v1.1.ggmlv3.{quantization}.bin"
    )
    MODEL_FAMILIES.append(
        ModelFamily(
            model_name="wizardlm-v1.1",
            model_sizes_in_billions=[13],
            model_format="ggmlv3",
            quantizations=[
                "q2_K",
                "q3_K_L",
                "q3_K_M",
                "q3_K_S",
                "q4_0",
                "q4_1",
                "q4_K_M",
                "q4_K_S",
                "q5_0",
                "q5_1",
                "q5_K_M",
                "q5_K_S",
                "q6_K",
                "q8_0",
            ],
            url_generator=wizardlm_v1_1_url_generator,
            rp_url_generator=wizardlm_v1_1_url_raw_generator,
            cls=VicunaCensoredGgml,  # according to https://huggingface.co/TheBloke/WizardLM-13B-V1.1-GGML
        ),
    )

    vicuna_v1_3_url_generator = lambda model_size, quantization: (
        "https://huggingface.co/TheBloke/vicuna-7B-v1.3-GGML/resolve/main/"
        f"vicuna-7b-v1.3.ggmlv3.{quantization}.bin"
        if model_size == 7
        else (
            "https://huggingface.co/TheBloke/vicuna-13b-v1.3.0-GGML/resolve/main/"
            f"vicuna-13b-v1.3.0.ggmlv3.{quantization}.bin"
            if model_size == 13
            else f"https://huggingface.co/TheBloke/vicuna-33B-GGML/resolve/main/vicuna-33b.ggmlv3.{quantization}.bin"
        )
    )

    vicuna_v1_3__url_raw_generator = lambda model_size, quantization: (
        "https://huggingface.co/TheBloke/vicuna-7B-v1.3-GGML/raw/main/"
        f"vicuna-7b-v1.3.ggmlv3.{quantization}.bin"
        if model_size == 7
        else (
            "https://huggingface.co/TheBloke/vicuna-13b-v1.3.0-GGML/raw/main/"
            f"vicuna-13b-v1.3.0.ggmlv3.{quantization}.bin"
        )
    )

    MODEL_FAMILIES.append(
        ModelFamily(
            model_name="vicuna-v1.3",
            model_sizes_in_billions=[7, 13, 33],
            model_format="ggmlv3",
            quantizations=[
                "q2_K",
                "q3_K_L",
                "q3_K_M",
                "q3_K_S",
                "q4_0",
                "q4_1",
                "q4_K_M",
                "q4_K_S",
                "q5_0",
                "q5_1",
                "q5_K_M",
                "q5_K_S",
                "q6_K",
                "q8_0",
            ],
            url_generator=vicuna_v1_3_url_generator,
            rp_url_generator=vicuna_v1_3__url_raw_generator,
            cls=VicunaCensoredGgml,
        ),
    )

    orca_url_generator = lambda model_size, quantization: (
        f"https://huggingface.co/TheBloke/orca_mini_{model_size}B-GGML/resolve/main/orca-mini-"
        f"{model_size}b.ggmlv3.{quantization}.bin"
    )
    orca_url_raw_generator = lambda model_size, quantization: (
        f"https://huggingface.co/TheBloke/orca_mini_{model_size}B-GGML/raw/main/orca-mini-"
        f"{model_size}b.ggmlv3.{quantization}.bin"
    )

    MODEL_FAMILIES.append(
        ModelFamily(
            model_name="orca",
            model_sizes_in_billions=[3, 7, 13],
            model_format="ggmlv3",
            quantizations=[
                "q4_0",
                "q4_1",
                "q5_0",
                "q5_1",
                "q8_0",
            ],
            url_generator=orca_url_generator,
            rp_url_generator=orca_url_raw_generator,
            cls=OrcaMiniGgml,
        )
    )

    chatglm_url_generator = lambda model_size, quantization: (
        f"https://huggingface.co/Xorbits/chatglm-{model_size}B-GGML/resolve/main/"
        f"chatglm-ggml-{quantization}.bin"
    )
    chatglm_url_raw_generator = lambda model_size, quantization: (
        f"https://huggingface.co/Xorbits/chatglm-{model_size}B-GGML/raw/main/"
        f"chatglm-ggml-{quantization}.bin"
    )
    MODEL_FAMILIES.append(
        ModelFamily(
            model_name="chatglm",
            model_sizes_in_billions=[6],
            model_format="ggmlv3",
            quantizations=[
                "q4_0",
                "q4_1",
                "q5_0",
                "q5_1",
                "q8_0",
            ],
            url_generator=chatglm_url_generator,
            rp_url_generator=chatglm_url_raw_generator,
            cls=ChatglmCppChatModel,
        )
    )

    chatglm2_url_generator = lambda model_size, quantization: (
        f"https://huggingface.co/Xorbits/chatglm2-{model_size}B-GGML/resolve/main/"
        f"chatglm2-ggml-{quantization}.bin"
    )
    chatglm2_url_raw_generator = lambda model_size, quantization: (
        f"https://huggingface.co/Xorbits/chatglm2-{model_size}B-GGML/raw/main/"
        f"chatglm2-ggml-{quantization}.bin"
    )
    MODEL_FAMILIES.append(
        ModelFamily(
            model_name="chatglm2",
            model_sizes_in_billions=[6],
            model_format="ggmlv3",
            quantizations=[
                "q4_0",
                "q4_1",
                "q5_0",
                "q5_1",
                "q8_0",
            ],
            url_generator=chatglm2_url_generator,
            rp_url_generator=chatglm2_url_raw_generator,
            cls=ChatglmCppChatModel,
        )
    )

    llama2_chat_url_generator = lambda model_size, quantization: (
        f"https://huggingface.co/TheBloke/Llama-2-{model_size}B-chat-GGML/resolve/main/llama-2-"
        f"{model_size}b-chat.ggmlv3.{quantization}.bin"
    )
    llama2_chat_url_raw_generator = lambda model_size, quantization: (
        f"https://huggingface.co/TheBloke/Llama-2-{model_size}B-chat-GGML/raw/main/llama-2-"
        f"{model_size}b-chat.ggmlv3.{quantization}.bin"
    )
    MODEL_FAMILIES.append(
        ModelFamily(
            model_name="llama-2-chat",
            model_sizes_in_billions=[7, 13],
            model_format="ggmlv3",
            quantizations=[
                "q2_K",
                "q3_K_L",
                "q3_K_M",
                "q3_K_S",
                "q4_0",
                "q4_1",
                "q4_K_M",
                "q4_K_S",
                "q5_0",
                "q5_1",
                "q5_K_M",
                "q5_K_S",
                "q6_K",
                "q8_0",
            ],
            url_generator=llama2_chat_url_generator,
            rp_url_generator=llama2_chat_url_raw_generator,
            cls=Llama2ChatGgml,
        )
    )

    llama2_url_generator = lambda model_size, quantization: (
        f"https://huggingface.co/TheBloke/Llama-2-{model_size}B-GGML/resolve/main/llama-2-"
        f"{model_size}b.ggmlv3.{quantization}.bin"
    )
    llama2_url_raw_generator = lambda model_size, quantization: (
        f"https://huggingface.co/TheBloke/Llama-2-{model_size}B-GGML/raw/main/llama-2-"
        f"{model_size}b.ggmlv3.{quantization}.bin"
    )
    MODEL_FAMILIES.append(
        ModelFamily(
            model_name="llama-2",
            model_sizes_in_billions=[7, 13],
            model_format="ggmlv3",
            quantizations=[
                "q2_K",
                "q3_K_L",
                "q3_K_M",
                "q3_K_S",
                "q4_0",
                "q4_1",
                "q4_K_M",
                "q4_K_S",
                "q5_0",
                "q5_1",
                "q5_K_M",
                "q5_K_S",
                "q6_K",
                "q8_0",
            ],
            url_generator=llama2_url_generator,
            rp_url_generator=llama2_url_raw_generator,
            cls=LlamaCppModel,
        )
    )

    pytorch_baichuan_name_generator = lambda model_size, quantization: (
        f"baichuan-inc/Baichuan-{model_size}B"
    )
    MODEL_FAMILIES.append(
        ModelFamily(
            model_name="baichuan",
            model_sizes_in_billions=[7],
            model_format="pytorch",
            quantizations=["8-bit", "4-bit", "none"],
            url_generator=pytorch_baichuan_name_generator,
            rp_url_generator=lambda model_size, quantization: "",
            cls=BaichuanPytorch,
        ),
    )

    pytorch_baichuan_base_name_generator = lambda model_size, quantization: (
        f"baichuan-inc/Baichuan-{model_size}B-Base"
    )
    MODEL_FAMILIES.append(
        ModelFamily(
            model_name="baichuan-base",
            model_sizes_in_billions=[13],
            model_format="pytorch",
            quantizations=["8-bit", "4-bit", "none"],
            url_generator=pytorch_baichuan_base_name_generator,
            rp_url_generator=lambda model_size, quantization: "",
            cls=BaichuanPytorch,
        ),
    )

    pytorch_baichuan_chat_name_generator = lambda model_size, quantization: (
        f"baichuan-inc/Baichuan-{model_size}B-Chat"
    )
    MODEL_FAMILIES.append(
        ModelFamily(
            model_name="baichuan-chat",
            model_sizes_in_billions=[13],
            model_format="pytorch",
            quantizations=["8-bit", "4-bit", "none"],
            url_generator=pytorch_baichuan_chat_name_generator,
            rp_url_generator=lambda model_size, quantization: "",
            cls=BaichuanPytorchChat,
        ),
    )

    # """
    # ggmlv3 model has the same model_name as pytorch model, comment it out in temporary
    pytorch_vicuna_v1_3_name_generator = lambda model_size, quantization: (
        f"lmsys/vicuna-{model_size}b-v1.3"
    )
    MODEL_FAMILIES.append(
        ModelFamily(
            model_name="vicuna-v1.3",
            model_sizes_in_billions=[7, 13, 33],
            model_format="pytorch",
            quantizations=["8-bit", "4-bit", "none"],
            url_generator=pytorch_vicuna_v1_3_name_generator,
            rp_url_generator=lambda model_size, quantization: "",
            cls=VicunaCensoredPytorch,
        ),
    )
    # """
