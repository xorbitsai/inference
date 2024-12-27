.. _models_builtin_sd3.5-large:

===========
sd3.5-large
===========

- **Model Name:** sd3.5-large
- **Model Family:** stable_diffusion
- **Abilities:** text2image, image2image, inpainting
- **Available ControlNet:** None

Specifications
^^^^^^^^^^^^^^

- **Model ID:** stabilityai/stable-diffusion-3.5-large
- **GGUF Model ID**: city96/stable-diffusion-3.5-large-gguf
- **GGUF Quantizations**: F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0


Execute the following command to launch the model::

   xinference launch --model-name sd3.5-large --model-type image


For GGUF quantization, using below command:

    xinference launch --model-name sd3.5-large --model-type image --gguf_quantization ${gguf_quantization} --cpu_offload True
