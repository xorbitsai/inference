.. _models_builtin_sd3.5-medium:

============
sd3.5-medium
============

- **Model Name:** sd3.5-medium
- **Model Family:** stable_diffusion
- **Abilities:** text2image, image2image, inpainting
- **Available ControlNet:** None

Specifications
^^^^^^^^^^^^^^

- **Model ID:** stabilityai/stable-diffusion-3.5-medium
- **GGUF Model ID**: city96/stable-diffusion-3.5-medium-gguf
- **GGUF Quantizations**: F16, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_0, Q5_1, Q5_K_M, Q5_K_S, Q6_K, Q8_0


Execute the following command to launch the model::

   xinference launch --model-name sd3.5-medium --model-type image


For GGUF quantization, using below command:

    xinference launch --model-name sd3.5-medium --model-type image --gguf_quantization ${gguf_quantization} --cpu_offload True
