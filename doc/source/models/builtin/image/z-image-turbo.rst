.. _models_builtin_z-image-turbo:

=============
Z-Image-Turbo
=============

- **Model Name:** Z-Image-Turbo
- **Model Family:** stable_diffusion
- **Abilities:** text2image, image2image
- **Available ControlNet:** None

Specifications
^^^^^^^^^^^^^^

- **Model ID:** Tongyi-MAI/Z-Image-Turbo
- **GGUF Model ID**: unsloth/Z-Image-Turbo-GGUF
- **GGUF Quantizations**: ['BF16', 'F16', 'Q2_K', 'Q3_K_L', 'Q3_K_M', 'Q3_K_S', 'Q4_0', 'Q4_1', 'Q4_K_M', 'Q4_K_S', 'Q5_0', 'Q5_1', 'Q5_K_M', 'Q5_K_S', 'Q6_K', 'Q8_0']


Execute the following command to launch the model::

   xinference launch --model-name Z-Image-Turbo --model-type image


For GGUF quantization, using below command::

    xinference launch --model-name Z-Image-Turbo --model-type image --gguf_quantization ${gguf_quantization} --cpu_offload True


