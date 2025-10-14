.. _models_builtin_qwen-image:

==========
Qwen-Image
==========

- **Model Name:** Qwen-Image
- **Model Family:** stable_diffusion
- **Abilities:** text2image, image2image, inpainting
- **Available ControlNet:** None

Specifications
^^^^^^^^^^^^^^

- **Model ID:** Qwen/Qwen-Image
- **GGUF Model ID**: city96/Qwen-Image-gguf
- **GGUF Quantizations**: F16, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_0, Q5_1, Q5_K_M, Q5_K_S, Q6_K, Q8_0

- **Lightning Model ID**: lightx2v/Qwen-Image-Lightning
- **Lightning Versions**: 4steps-V1.0-bf16, 4steps-V1.0, 8steps-V1.0, 8steps-V1.1-bf16, 8steps-V1.1


Execute the following command to launch the model::

   xinference launch --model-name Qwen-Image --model-type image


For GGUF quantization, using below command::

    xinference launch --model-name Qwen-Image --model-type image --gguf_quantization ${gguf_quantization} --cpu_offload True



For Lightning LoRA acceleration, using below command::

    xinference launch --model-name Qwen-Image --model-type image --lightning_version ${lightning_version}
