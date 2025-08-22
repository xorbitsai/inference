.. _models_builtin_qwen-image-edit:

===============
Qwen-Image-Edit
===============

- **Model Name:** Qwen-Image-Edit
- **Model Family:** stable_diffusion
- **Abilities:** image2image
- **Available ControlNet:** None

Specifications
^^^^^^^^^^^^^^

- **Model ID:** Qwen/Qwen-Image-Edit
- **GGUF Model ID**: QuantStack/Qwen-Image-Edit-GGUF
- **GGUF Quantizations**: Q2_K, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_0, Q5_1, Q5_K_M, Q5_K_S, Q6_K, Q8_0


Execute the following command to launch the model::

   xinference launch --model-name Qwen-Image-Edit --model-type image


For GGUF quantization, using below command:

    xinference launch --model-name Qwen-Image-Edit --model-type image --gguf_quantization ${gguf_quantization} --cpu_offload True
