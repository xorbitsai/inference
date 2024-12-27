.. _models_builtin_flux.1-dev:

==========
FLUX.1-dev
==========

- **Model Name:** FLUX.1-dev
- **Model Family:** stable_diffusion
- **Abilities:** text2image, image2image, inpainting
- **Available ControlNet:** None

Specifications
^^^^^^^^^^^^^^

- **Model ID:** black-forest-labs/FLUX.1-dev
- **GGUF Model ID**: city96/FLUX.1-dev-gguf
- **GGUF Quantizations**: F16, Q2_K, Q3_K_S, Q4_0, Q4_1, Q4_K_S, Q5_0, Q5_1, Q5_K_S, Q6_K, Q8_0


Execute the following command to launch the model::

   xinference launch --model-name FLUX.1-dev --model-type image


For GGUF quantization, using below command:

    xinference launch --model-name FLUX.1-dev --model-type image --gguf_quantization ${gguf_quantization} --cpu_offload True
