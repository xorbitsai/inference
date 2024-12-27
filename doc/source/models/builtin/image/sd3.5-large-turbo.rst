.. _models_builtin_sd3.5-large-turbo:

=================
sd3.5-large-turbo
=================

- **Model Name:** sd3.5-large-turbo
- **Model Family:** stable_diffusion
- **Abilities:** text2image, image2image, inpainting
- **Available ControlNet:** None

Specifications
^^^^^^^^^^^^^^

- **Model ID:** stabilityai/stable-diffusion-3.5-large-turbo
- **GGUF Model ID**: city96/stable-diffusion-3.5-large-turbo-gguf
- **GGUF Quantizations**: F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0


Execute the following command to launch the model::

   xinference launch --model-name sd3.5-large-turbo --model-type image


For GGUF quantization, using below command:

    xinference launch --model-name sd3.5-large-turbo --model-type image --gguf_quantization ${gguf_quantization} --cpu_offload True
